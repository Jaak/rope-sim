use crate::config::{ConfigError, SimulationConfig};
use crate::dynamics::RopeDynamics;
use crate::integrators::{
    BackwardEuler, DynamicalSystem, PredictorCorrection, StepError, TimeIntegrator,
    create_integrator,
};
use crate::kinematics::{KinematicMotion, KinematicTarget};
use crate::materials::AxialMaterial;
use crate::math::Vec2;
use crate::state::State;
use crate::xpbd::XpbdRopeRelaxer;

const MAXIMUM_MANIPULATION_THROW_SPEED: f64 = 20.0;
const MANIPULATION_CORRECTION_INTERVAL: f64 = 1.0 / 120.0;
const MANIPULATION_NEWTON_BUDGET: usize = 4;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Diagnostics {
    pub simulation_time: f64,
    pub kinetic_energy: f64,
    pub gravitational_energy: f64,
    pub elastic_energy: f64,
    pub total_mechanical_energy: f64,
    pub maximum_absolute_strain: f64,
    pub maximum_tensile_strain: f64,
    pub minimum_segment_length: f64,
    pub maximum_segment_length: f64,
    pub maximum_node_speed: f64,
    pub prescribed_endpoint_power: f64,
    pub cumulative_prescribed_work: f64,
    pub rejected_steps: u64,
    pub linear_solves: u64,
    pub nonlinear_iterations: u64,
    pub adaptive_retries: u64,
    pub residual_evaluations: u64,
    pub jacobian_assemblies: u64,
    pub block_factorizations: u64,
    pub sparse_factorizations: u64,
    pub line_search_backtracks: u64,
    pub manipulation_corrections: u64,
    pub manipulation_correction_fallbacks: u64,
    pub manipulation_release_handoffs: u64,
    pub manipulation_last_fallback_residual: f64,
    pub explicit_stable_timestep: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconfigureOutcome {
    Updated,
    Reset,
}

#[derive(Clone, Copy)]
struct ManipulationStep {
    dt: f64,
    start_target: KinematicTarget,
    end_target: KinematicTarget,
}

/// A lumped-mass axial rope simulation.
pub struct Simulation {
    config: SimulationConfig,
    state: State,
    masses: Vec<f64>,
    rest_length: f64,
    time: f64,
    payload_target: Option<KinematicTarget>,
    payload_motion: Option<KinematicMotion>,
    manipulation_target: Option<KinematicTarget>,
    manipulation_relaxer: XpbdRopeRelaxer,
    interaction_integrator: BackwardEuler,
    interaction_initial: State,
    interaction_candidate: State,
    last_manipulation_step: Option<ManipulationStep>,
    manipulation_correction_accumulator: f64,
    manipulation_corrections: u64,
    manipulation_correction_fallbacks: u64,
    manipulation_release_handoffs: u64,
    manipulation_last_fallback_residual: f64,
    integrator: Box<dyn TimeIntegrator>,
    cumulative_prescribed_work: f64,
}

impl Simulation {
    pub fn new(config: SimulationConfig) -> Result<Self, ConfigError> {
        let config = config.validate()?;
        let node_count = config.segment_count + 1;
        let rest_length = config.rope_length / config.segment_count as f64;
        let mut positions = Vec::with_capacity(node_count);

        for index in 0..node_count {
            positions.push(config.anchor + Vec2::new(0.0, -(index as f64) * rest_length));
        }
        let scratch_positions = vec![Vec2::ZERO; node_count];

        let mut simulation = Self {
            config,
            state: State::new(positions),
            masses: vec![0.0; node_count],
            rest_length,
            time: 0.0,
            payload_target: None,
            payload_motion: None,
            manipulation_target: None,
            manipulation_relaxer: XpbdRopeRelaxer::new(node_count),
            interaction_integrator: BackwardEuler::new(node_count),
            interaction_initial: State::new(scratch_positions.clone()),
            interaction_candidate: State::new(scratch_positions),
            last_manipulation_step: None,
            manipulation_correction_accumulator: 0.0,
            manipulation_corrections: 0,
            manipulation_correction_fallbacks: 0,
            manipulation_release_handoffs: 0,
            manipulation_last_fallback_residual: 0.0,
            integrator: create_integrator(config.integrator, node_count),
            cumulative_prescribed_work: 0.0,
        };
        simulation.rebuild_masses();
        Ok(simulation)
    }

    pub fn config(&self) -> SimulationConfig {
        self.config
    }

    pub fn positions(&self) -> &[Vec2] {
        &self.state.positions
    }

    pub fn velocities(&self) -> &[Vec2] {
        &self.state.velocities
    }

    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    pub fn rest_length(&self) -> f64 {
        self.rest_length
    }

    pub fn payload_position(&self) -> Vec2 {
        self.state.positions[self.payload_index()]
    }

    pub fn payload_velocity(&self) -> Vec2 {
        self.state.velocities[self.payload_index()]
    }

    /// Signed axial force in one rope element, in newtons.
    ///
    /// Positive values are tensile and negative values are compressive. This
    /// exposes the constitutive response for diagnostics without permitting
    /// callers to mutate the simulation state.
    pub fn segment_tension(&self, left: usize) -> Option<f64> {
        if left >= self.config.segment_count {
            return None;
        }
        let right = left + 1;
        let delta = self.state.positions[right] - self.state.positions[left];
        let length = delta.length();
        if length <= f64::EPSILON {
            return Some(0.0);
        }
        let direction = delta / length;
        let relative_velocity = self.state.velocities[right] - self.state.velocities[left];
        let material = AxialMaterial::from_config(self.config);
        Some(
            material
                .response(
                    crate::materials::AxialKinematics {
                        extension: length - self.rest_length,
                        extension_rate: direction.dot(relative_velocity),
                    },
                    self.state.material_state[left],
                    self.rest_length,
                )
                .force,
        )
    }

    /// Whether the interaction-only XPBD solver currently owns the state.
    pub fn manipulation_active(&self) -> bool {
        self.manipulation_target.is_some()
    }

    /// Recommend a conservative number of substeps for the selected integrator.
    ///
    /// Refining an axial rope simultaneously increases each element's spring
    /// stiffness and decreases its lumped node mass. Consequently the highest
    /// modal frequency grows approximately in proportion to the segment count.
    /// Explicit methods use the fastest local spring-mass mode directly;
    /// linearly implicit methods may use a relaxed version as a nonlinear
    /// linearization limit.
    pub fn recommended_substeps(&self, outer_dt: f64) -> Result<usize, StepError> {
        if self.manipulation_active() {
            if !outer_dt.is_finite() || outer_dt <= 0.0 {
                return Err(StepError::InvalidTimeStep(outer_dt));
            }
            return Ok(1);
        }
        let system = RopeDynamics::new(
            &self.config,
            &self.masses,
            self.rest_length,
            self.payload_target,
            self.payload_motion,
            self.kinematic_speed(),
        );
        self.integrator
            .recommended_substeps(&system, &self.state, outer_dt)
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.config).expect("an existing simulation has valid configuration");
    }

    pub fn reconfigure(
        &mut self,
        config: SimulationConfig,
    ) -> Result<ReconfigureOutcome, ConfigError> {
        let config = config.validate()?;
        let topology_changed = config.segment_count != self.config.segment_count
            || config.rope_length != self.config.rope_length
            || config.anchor != self.config.anchor;
        let integrator_changed = config.integrator != self.config.integrator;
        let material_model_changed = config.rope_model != self.config.rope_model;

        if topology_changed {
            *self = Self::new(config)?;
            Ok(ReconfigureOutcome::Reset)
        } else {
            self.config = config;
            self.rebuild_masses();
            if material_model_changed {
                self.state.material_state.fill(0.0);
            }
            if integrator_changed {
                self.integrator =
                    create_integrator(self.config.integrator, self.state.node_count());
            }
            Ok(ReconfigureOutcome::Updated)
        }
    }

    /// Prescribe the payload state. Passing `None` returns it to dynamic motion.
    ///
    /// The position is applied immediately so dragging remains responsive while
    /// the simulation is paused.
    pub fn set_payload_target(&mut self, target: Option<KinematicTarget>) {
        self.manipulation_target = None;
        self.last_manipulation_step = None;
        self.manipulation_correction_accumulator = 0.0;
        self.payload_motion = None;
        self.payload_target = target;
        if let Some(target) = target {
            let index = self.payload_index();
            self.state.positions[index] = target.position;
            self.state.velocities[index] = target.velocity;
        }
    }

    /// Move a held payload to a new target along a bounded linear trajectory.
    ///
    /// The trajectory is sampled by each physics substep, avoiding the
    /// render-frame position jumps that would otherwise excite high-frequency
    /// rope modes.
    pub fn interpolate_payload_target(
        &mut self,
        target: KinematicTarget,
        duration: f64,
    ) -> Result<(), StepError> {
        if !duration.is_finite() || duration <= 0.0 {
            return Err(StepError::InvalidTimeStep(duration));
        }

        self.manipulation_target = None;
        self.last_manipulation_step = None;
        self.manipulation_correction_accumulator = 0.0;
        let start = KinematicTarget::new(self.payload_position(), self.payload_velocity());
        self.payload_target = Some(start);
        self.payload_motion = Some(KinematicMotion::new(start, target, duration));
        Ok(())
    }

    /// Release a kinematically held payload with an explicit world-space velocity.
    pub fn release_payload(&mut self, velocity: Vec2) {
        self.manipulation_target = None;
        self.last_manipulation_step = None;
        self.manipulation_correction_accumulator = 0.0;
        self.payload_motion = None;
        self.payload_target = None;
        let index = self.payload_index();
        self.state.velocities[index] = velocity;
    }

    /// Move the payload directly while relaxing the rope with XPBD.
    ///
    /// The position is exact while the supplied velocity is retained for a
    /// continuous, throwable handoff to the selected physical integrator.
    pub fn set_manipulation_target(&mut self, target: KinematicTarget) {
        self.payload_motion = None;
        self.payload_target = None;
        if self.manipulation_target.is_none() {
            self.last_manipulation_step = None;
            self.manipulation_correction_accumulator = 0.0;
        }
        let target = KinematicTarget::new(
            target.position,
            limit_magnitude(target.velocity, MAXIMUM_MANIPULATION_THROW_SPEED),
        );
        self.manipulation_target = Some(target);
        let payload = self.payload_index();
        self.state.velocities[payload] = target.velocity;
    }

    /// Immediately return from XPBD manipulation to the selected physical
    /// integrator, preserving internal motion and applying the throw velocity.
    pub fn release_manipulation(&mut self, velocity: Vec2) {
        let velocity = self.manipulation_target.map_or_else(
            || limit_magnitude(velocity, MAXIMUM_MANIPULATION_THROW_SPEED),
            |target| target.velocity,
        );
        self.finish_manipulation_handoff();
        self.manipulation_target = None;
        self.last_manipulation_step = None;
        self.manipulation_correction_accumulator = 0.0;
        let payload = self.payload_index();
        self.state.velocities[payload] = velocity;
    }

    pub fn step(&mut self, dt: f64) -> Result<Diagnostics, StepError> {
        self.step_without_diagnostics(dt)?;
        Ok(self.diagnostics())
    }

    /// Advance the simulation without traversing the rope to compute diagnostics.
    ///
    /// This is useful when an outer frame is split into several physics steps and
    /// only the state after the final step will be displayed or inspected.
    pub fn step_without_diagnostics(&mut self, dt: f64) -> Result<(), StepError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(StepError::InvalidTimeStep(dt));
        }
        if let Some(target) = self.manipulation_target {
            let start_target =
                KinematicTarget::new(self.payload_position(), self.payload_velocity());
            self.interaction_initial.clone_from(&self.state);
            self.manipulation_relaxer.step_held(
                &self.config,
                &mut self.state,
                &self.masses,
                self.rest_length,
                target,
                dt,
            );
            let manipulation_step = ManipulationStep {
                dt,
                start_target,
                end_target: target,
            };
            self.manipulation_correction_accumulator += dt;
            if self.manipulation_correction_accumulator + f64::EPSILON
                >= MANIPULATION_CORRECTION_INTERVAL
            {
                self.manipulation_correction_accumulator %= MANIPULATION_CORRECTION_INTERVAL;
                self.try_bounded_manipulation_correction(manipulation_step);
            }
            self.last_manipulation_step = Some(manipulation_step);
            self.time += dt;
            return Ok(());
        }
        let system = RopeDynamics::new(
            &self.config,
            &self.masses,
            self.rest_length,
            self.payload_target,
            self.payload_motion,
            self.kinematic_speed(),
        );
        self.integrator.step(&system, &mut self.state, dt)?;
        self.advance_payload_motion(dt);
        self.cumulative_prescribed_work += self.prescribed_endpoint_power() * dt;
        self.time += dt;
        Ok(())
    }

    fn try_bounded_manipulation_correction(&mut self, step: ManipulationStep) {
        let motion = KinematicMotion::new(step.start_target, step.end_target, step.dt);
        let kinematic_speed = manipulation_speed(step);
        let system = RopeDynamics::new(
            &self.config,
            &self.masses,
            self.rest_length,
            Some(step.start_target),
            Some(motion),
            kinematic_speed,
        );
        let correction = self.interaction_integrator.correct_from_predictor(
            &system,
            &self.interaction_initial,
            &self.state,
            &mut self.interaction_candidate,
            step.dt,
            MANIPULATION_NEWTON_BUDGET,
        );
        match correction {
            Ok(PredictorCorrection::Converged) => {
                self.state.clone_from(&self.interaction_candidate);
                self.manipulation_corrections += 1;
                self.manipulation_last_fallback_residual = 0.0;
            }
            Ok(PredictorCorrection::BudgetExceeded { residual, .. }) => {
                self.manipulation_correction_fallbacks += 1;
                self.manipulation_last_fallback_residual = residual;
            }
            Err(_) => {
                self.manipulation_correction_fallbacks += 1;
                self.manipulation_last_fallback_residual = f64::INFINITY;
            }
        }
    }

    fn finish_manipulation_handoff(&mut self) {
        let Some(step) = self.last_manipulation_step else {
            return;
        };
        let motion = KinematicMotion::new(step.start_target, step.end_target, step.dt);
        let system = RopeDynamics::new(
            &self.config,
            &self.masses,
            self.rest_length,
            Some(step.start_target),
            Some(motion),
            manipulation_speed(step),
        );
        let correction = self.interaction_integrator.correct_from_predictor_fully(
            &system,
            &self.interaction_initial,
            &self.state,
            &mut self.interaction_candidate,
            step.dt,
        );
        if matches!(correction, Ok(PredictorCorrection::Converged)) {
            self.state.clone_from(&self.interaction_candidate);
            self.manipulation_release_handoffs += 1;
            return;
        }

        // A difficult handoff gets the normal adaptive backward-Euler path as
        // a second chance. Both attempts are transactional; if this also fails,
        // retain the safe XPBD state and let the selected integrator continue.
        self.interaction_candidate
            .clone_from(&self.interaction_initial);
        if self
            .interaction_integrator
            .step(&system, &mut self.interaction_candidate, step.dt)
            .is_ok()
        {
            self.state.clone_from(&self.interaction_candidate);
            self.manipulation_release_handoffs += 1;
        } else {
            self.manipulation_correction_fallbacks += 1;
        }
    }

    pub fn diagnostics(&self) -> Diagnostics {
        let mut kinetic_energy = 0.0;
        let mut gravitational_energy = 0.0;
        let mut elastic_energy = 0.0;
        let mut maximum_absolute_strain: f64 = 0.0;
        let mut maximum_tensile_strain: f64 = 0.0;
        let mut minimum_segment_length = f64::INFINITY;
        let mut maximum_segment_length: f64 = 0.0;
        let mut maximum_node_speed: f64 = 0.0;

        for index in 0..self.state.node_count() {
            kinetic_energy +=
                0.5 * self.masses[index] * self.state.velocities[index].length_squared();
            gravitational_energy -= self.masses[index]
                * self
                    .config
                    .gravity
                    .dot(self.state.positions[index] - self.config.anchor);
            maximum_node_speed = maximum_node_speed.max(self.state.velocities[index].length());
        }

        let material = AxialMaterial::from_config(self.config);
        for index in 0..self.config.segment_count {
            let length = (self.state.positions[index + 1] - self.state.positions[index]).length();
            let extension = length - self.rest_length;
            minimum_segment_length = minimum_segment_length.min(length);
            maximum_segment_length = maximum_segment_length.max(length);
            elastic_energy += material.stored_energy(
                extension,
                self.state.material_state[index],
                self.rest_length,
            );
            maximum_absolute_strain =
                maximum_absolute_strain.max((extension / self.rest_length).abs());
            maximum_tensile_strain =
                maximum_tensile_strain.max((extension / self.rest_length).max(0.0));
        }

        let system = RopeDynamics::new(
            &self.config,
            &self.masses,
            self.rest_length,
            self.payload_target,
            self.payload_motion,
            self.kinematic_speed(),
        );
        let statistics = self.integrator.statistics();

        Diagnostics {
            simulation_time: self.time,
            kinetic_energy,
            gravitational_energy,
            elastic_energy,
            total_mechanical_energy: kinetic_energy + gravitational_energy + elastic_energy,
            maximum_absolute_strain,
            maximum_tensile_strain,
            minimum_segment_length,
            maximum_segment_length,
            maximum_node_speed,
            prescribed_endpoint_power: self.prescribed_endpoint_power(),
            cumulative_prescribed_work: self.cumulative_prescribed_work,
            rejected_steps: statistics.rejected_steps,
            linear_solves: statistics.linear_solves,
            nonlinear_iterations: statistics.nonlinear_iterations,
            adaptive_retries: statistics.adaptive_retries,
            residual_evaluations: statistics.residual_evaluations,
            jacobian_assemblies: statistics.jacobian_assemblies,
            block_factorizations: statistics.block_factorizations,
            sparse_factorizations: statistics.sparse_factorizations,
            line_search_backtracks: statistics.line_search_backtracks,
            manipulation_corrections: self.manipulation_corrections,
            manipulation_correction_fallbacks: self.manipulation_correction_fallbacks,
            manipulation_release_handoffs: self.manipulation_release_handoffs,
            manipulation_last_fallback_residual: self.manipulation_last_fallback_residual,
            explicit_stable_timestep: system.explicit_stable_timestep(&self.state),
        }
    }

    fn prescribed_endpoint_power(&self) -> f64 {
        let Some(target) = self.payload_target else {
            return 0.0;
        };
        let right = self.payload_index();
        let left = right - 1;
        let delta = self.state.positions[right] - self.state.positions[left];
        let length = delta.length();
        if length <= f64::EPSILON {
            return 0.0;
        }
        let direction = delta / length;
        let tension = self.segment_tension(left).unwrap_or(0.0);
        (direction * tension).dot(target.velocity)
    }

    fn payload_index(&self) -> usize {
        self.state.node_count() - 1
    }

    fn rebuild_masses(&mut self) {
        self.masses.fill(0.0);
        let element_mass = self.config.rope_mass / self.config.segment_count as f64;

        self.masses[0] = 0.5 * element_mass;
        let payload_index = self.payload_index();
        self.masses[payload_index] = 0.5 * element_mass + self.config.payload_mass;
        for index in 1..payload_index {
            self.masses[index] = element_mass;
        }
    }

    fn advance_payload_motion(&mut self, dt: f64) {
        let Some(motion) = &mut self.payload_motion else {
            return;
        };

        let (target, finished) = motion.advance(dt);
        self.payload_target = Some(target);
        if finished {
            self.payload_motion = None;
        }
    }

    fn kinematic_speed(&self) -> f64 {
        let target_speed = self
            .payload_target
            .map_or(0.0, |target| target.velocity.length());
        self.payload_motion.map_or(target_speed, |motion| {
            target_speed.max(motion.maximum_speed())
        })
    }
}

fn limit_magnitude(value: Vec2, maximum: f64) -> Vec2 {
    let magnitude = value.length();
    if magnitude > maximum {
        value * (maximum / magnitude)
    } else {
        value
    }
}

fn manipulation_speed(step: ManipulationStep) -> f64 {
    let chord_speed = (step.end_target.position - step.start_target.position).length() / step.dt;
    chord_speed
        .max(step.start_target.velocity.length())
        .max(step.end_target.velocity.length())
}

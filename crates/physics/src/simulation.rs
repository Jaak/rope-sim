use crate::config::{ConfigError, SimulationConfig};
use crate::dynamics::RopeDynamics;
use crate::integrators::{DynamicalSystem, StepError, TimeIntegrator, create_integrator};
use crate::kinematics::{KinematicMotion, KinematicTarget};
use crate::materials::AxialMaterial;
use crate::math::Vec2;
use crate::state::State;

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
    pub explicit_stable_timestep: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconfigureOutcome {
    Updated,
    Reset,
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

        let mut simulation = Self {
            config,
            state: State::new(positions),
            masses: vec![0.0; node_count],
            rest_length,
            time: 0.0,
            payload_target: None,
            payload_motion: None,
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

    /// Recommend a conservative number of substeps for the selected integrator.
    ///
    /// Refining an axial rope simultaneously increases each element's spring
    /// stiffness and decreases its lumped node mass. Consequently the highest
    /// modal frequency grows approximately in proportion to the segment count.
    /// Explicit methods use the fastest local spring-mass mode directly;
    /// linearly implicit methods may use a relaxed version as a nonlinear
    /// linearization limit.
    pub fn recommended_substeps(&self, outer_dt: f64) -> Result<usize, StepError> {
        let system = RopeDynamics::new(
            &self.config,
            &self.masses,
            self.rest_length,
            self.payload_target,
            self.payload_motion,
            self.kinematic_speed(),
        );
        self.integrator.recommended_substeps(&system, outer_dt)
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

        let start = KinematicTarget::new(self.payload_position(), self.payload_velocity());
        self.payload_target = Some(start);
        self.payload_motion = Some(KinematicMotion::new(start, target, duration));
        Ok(())
    }

    /// Release a kinematically held payload with an explicit world-space velocity.
    pub fn release_payload(&mut self, velocity: Vec2) {
        self.payload_motion = None;
        self.payload_target = None;
        let index = self.payload_index();
        self.state.velocities[index] = velocity;
    }

    pub fn step(&mut self, dt: f64) -> Result<Diagnostics, StepError> {
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
        Ok(self.diagnostics())
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
            explicit_stable_timestep: system.explicit_stable_timestep(),
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
        let relative_velocity = self.state.velocities[right] - self.state.velocities[left];
        let material = AxialMaterial::from_config(self.config);
        let response = material.response(
            crate::materials::AxialKinematics {
                extension: length - self.rest_length,
                extension_rate: direction.dot(relative_velocity),
            },
            self.state.material_state[left],
            self.rest_length,
        );
        (direction * response.force).dot(target.velocity)
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

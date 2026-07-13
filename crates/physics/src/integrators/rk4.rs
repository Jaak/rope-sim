use crate::math::Vec2;
use crate::state::State;

use super::{DynamicalSystem, StepError, TimeIntegrator, validate_timestep};

/// Classical explicit fourth-order Runge-Kutta integration.
///
/// RK4 has a wider stability interval and much lower local error than
/// semi-implicit Euler, but remains explicit and therefore uses automatic
/// substeps for the rope's fastest axial modes.
pub(super) struct RungeKutta4 {
    stage: State,
    k1: StateDerivative,
    k2: StateDerivative,
    k3: StateDerivative,
    k4: StateDerivative,
}

impl RungeKutta4 {
    pub fn new(node_count: usize) -> Self {
        let zero_positions = vec![Vec2::ZERO; node_count];
        Self {
            stage: State::new(zero_positions),
            k1: StateDerivative::new(node_count),
            k2: StateDerivative::new(node_count),
            k3: StateDerivative::new(node_count),
            k4: StateDerivative::new(node_count),
        }
    }
}

impl TimeIntegrator for RungeKutta4 {
    fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;
        system.enforce_kinematics(state, 0.0);

        evaluate_derivative(system, state, &mut self.k1);
        set_stage(system, &mut self.stage, state, &self.k1, 0.5 * dt, 0.5 * dt);
        evaluate_derivative(system, &self.stage, &mut self.k2);
        set_stage(system, &mut self.stage, state, &self.k2, 0.5 * dt, 0.5 * dt);
        evaluate_derivative(system, &self.stage, &mut self.k3);
        set_stage(system, &mut self.stage, state, &self.k3, dt, dt);
        evaluate_derivative(system, &self.stage, &mut self.k4);

        let scale = dt / 6.0;
        for node in 0..state.node_count() {
            if !system.is_dynamic_node(node) {
                continue;
            }
            state.positions[node] += (self.k1.positions[node]
                + self.k2.positions[node] * 2.0
                + self.k3.positions[node] * 2.0
                + self.k4.positions[node])
                * scale;
            state.velocities[node] += (self.k1.velocities[node]
                + self.k2.velocities[node] * 2.0
                + self.k3.velocities[node] * 2.0
                + self.k4.velocities[node])
                * scale;
        }
        for element in 0..state.material_state.len() {
            state.material_state[element] += (self.k1.material_state[element]
                + 2.0 * self.k2.material_state[element]
                + 2.0 * self.k3.material_state[element]
                + self.k4.material_state[element])
                * scale;
        }

        system.enforce_kinematics(state, dt);
        if state.is_finite() {
            Ok(())
        } else {
            Err(StepError::NonFiniteState)
        }
    }

    fn recommended_substeps(
        &self,
        system: &dyn DynamicalSystem,
        state: &State,
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        // Classical RK4 reaches farther along the imaginary stability axis
        // than semi-implicit Euler. This factor retains margin below that
        // theoretical boundary for the nonlinear, nonuniform rope.
        let maximum_dt = 1.65 * system.explicit_stable_timestep(state);
        Ok((outer_dt / maximum_dt).ceil().max(1.0) as usize)
    }
}

struct StateDerivative {
    positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
    material_state: Vec<f64>,
}

impl StateDerivative {
    fn new(node_count: usize) -> Self {
        Self {
            positions: vec![Vec2::ZERO; node_count],
            velocities: vec![Vec2::ZERO; node_count],
            material_state: vec![0.0; node_count.saturating_sub(1)],
        }
    }

    fn resize_and_clear(&mut self, node_count: usize) {
        self.positions.resize(node_count, Vec2::ZERO);
        self.velocities.resize(node_count, Vec2::ZERO);
        self.material_state
            .resize(node_count.saturating_sub(1), 0.0);
        self.positions.fill(Vec2::ZERO);
        self.velocities.fill(Vec2::ZERO);
        self.material_state.fill(0.0);
    }
}

fn evaluate_derivative(system: &dyn DynamicalSystem, state: &State, output: &mut StateDerivative) {
    output.resize_and_clear(state.node_count());
    system.accelerations(state, &mut output.velocities);
    system.material_state_derivatives(state, &mut output.material_state);
    for node in 0..state.node_count() {
        if system.is_dynamic_node(node) {
            output.positions[node] = state.velocities[node];
        }
    }
}

fn set_stage(
    system: &dyn DynamicalSystem,
    stage: &mut State,
    initial: &State,
    derivative: &StateDerivative,
    dt: f64,
    stage_time: f64,
) {
    stage.clone_from(initial);
    for node in 0..initial.node_count() {
        if system.is_dynamic_node(node) {
            stage.positions[node] += derivative.positions[node] * dt;
            stage.velocities[node] += derivative.velocities[node] * dt;
        }
    }
    for element in 0..initial.material_state.len() {
        stage.material_state[element] += derivative.material_state[element] * dt;
    }
    system.enforce_kinematics(stage, stage_time);
}

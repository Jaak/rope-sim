use crate::math::Vec2;
use crate::state::State;

use super::block_tridiagonal::{
    BLOCK_SIZE, BlockThomasSolver, BlockTridiagonalMatrix, BlockVector,
};
use super::{DynamicalSystem, IntegratorStatistics, StepError, TimeIntegrator, validate_timestep};

// L-stable ROS2 branch with bounded stiff-mode internal stages.
const GAMMA: f64 = 1.0 + std::f64::consts::FRAC_1_SQRT_2;
const FREE_MOTION_LIMIT_MULTIPLIER: f64 = 8.0;
const KINEMATIC_MOTION_LIMIT_MULTIPLIER: f64 = 1.0;

pub(super) struct Rosenbrock2 {
    backup: State,
    stage: State,
    accelerations: Vec<Vec2>,
    material_derivatives: Vec<f64>,
    matrix: BlockTridiagonalMatrix,
    solver: BlockThomasSolver,
    k1: Vec<BlockVector>,
    k2: Vec<BlockVector>,
    statistics: IntegratorStatistics,
}

impl Rosenbrock2 {
    pub fn new(node_count: usize) -> Self {
        let positions = vec![Vec2::ZERO; node_count];
        Self {
            backup: State::new(positions.clone()),
            stage: State::new(positions),
            accelerations: vec![Vec2::ZERO; node_count],
            material_derivatives: vec![0.0; node_count.saturating_sub(1)],
            matrix: BlockTridiagonalMatrix::new(node_count),
            solver: BlockThomasSolver::new(node_count),
            k1: vec![[0.0; BLOCK_SIZE]; node_count],
            k2: vec![[0.0; BLOCK_SIZE]; node_count],
            statistics: IntegratorStatistics::default(),
        }
    }

    fn prepare(&mut self, state: &State) {
        let node_count = state.node_count();
        self.accelerations.resize(node_count, Vec2::ZERO);
        self.material_derivatives
            .resize(node_count.saturating_sub(1), 0.0);
        self.k1.resize(node_count, [0.0; BLOCK_SIZE]);
        self.k2.resize(node_count, [0.0; BLOCK_SIZE]);
    }
}

impl TimeIntegrator for Rosenbrock2 {
    fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;
        system.enforce_kinematics(state, 0.0);
        self.backup.clone_from(state);
        self.prepare(state);

        system.first_order_jacobian(state, &mut self.matrix);
        self.matrix.shift_and_scale(-GAMMA * dt);
        if self.solver.factorize(&self.matrix).is_err() {
            self.statistics.rejected_steps += 1;
            return Err(StepError::SingularJacobian);
        }

        evaluate_derivative(
            system,
            state,
            &mut self.accelerations,
            &mut self.material_derivatives,
            &mut self.k1,
        );
        scale_blocks(&mut self.k1, dt);
        self.solver.solve_in_place(&mut self.k1);
        self.statistics.linear_solves += 1;

        self.stage.clone_from(state);
        add_increment(system, &mut self.stage, &self.k1, 1.0, dt);
        if !self.stage.is_finite() {
            state.clone_from(&self.backup);
            self.statistics.rejected_steps += 1;
            return Err(StepError::NonFiniteState);
        }

        evaluate_derivative(
            system,
            &self.stage,
            &mut self.accelerations,
            &mut self.material_derivatives,
            &mut self.k2,
        );
        for (second, first) in self.k2.iter_mut().zip(&self.k1) {
            for component in 0..BLOCK_SIZE {
                second[component] = dt * second[component] - 2.0 * first[component];
            }
        }
        self.solver.solve_in_place(&mut self.k2);
        self.statistics.linear_solves += 1;

        self.stage.clone_from(&self.backup);
        add_increment(system, &mut self.stage, &self.k1, 1.5, dt);
        add_increment(system, &mut self.stage, &self.k2, 0.5, dt);
        if self.stage.is_finite() {
            state.clone_from(&self.stage);
            return Ok(());
        }

        state.clone_from(&self.backup);
        self.statistics.rejected_steps += 1;
        Err(StepError::NonFiniteState)
    }

    fn recommended_substeps(
        &self,
        system: &dyn DynamicalSystem,
        state: &State,
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        let multiplier = if system.has_kinematic_target() {
            KINEMATIC_MOTION_LIMIT_MULTIPLIER
        } else {
            FREE_MOTION_LIMIT_MULTIPLIER
        };
        let linearization_limit = (multiplier * system.elastic_stable_timestep(state))
            .min(system.kinematic_timestep_limit());
        let requested = (outer_dt / linearization_limit).ceil().max(1.0) as usize;
        Ok(requested)
    }

    fn statistics(&self) -> IntegratorStatistics {
        self.statistics
    }
}

fn evaluate_derivative(
    system: &dyn DynamicalSystem,
    state: &State,
    accelerations: &mut [Vec2],
    material_derivatives: &mut [f64],
    output: &mut [BlockVector],
) {
    accelerations.fill(Vec2::ZERO);
    material_derivatives.fill(0.0);
    output.fill([0.0; BLOCK_SIZE]);
    system.accelerations(state, accelerations);
    system.material_state_derivatives(state, material_derivatives);

    for node in 0..state.node_count() {
        if system.is_dynamic_node(node) {
            output[node][0] = state.velocities[node].x;
            output[node][1] = state.velocities[node].y;
            output[node][2] = accelerations[node].x;
            output[node][3] = accelerations[node].y;
        }
        if node < material_derivatives.len() {
            output[node][4] = material_derivatives[node];
        }
    }
}

fn scale_blocks(blocks: &mut [BlockVector], scale: f64) {
    for block in blocks {
        for value in block {
            *value *= scale;
        }
    }
}

fn add_increment(
    system: &dyn DynamicalSystem,
    state: &mut State,
    increment: &[BlockVector],
    scale: f64,
    stage_time: f64,
) {
    for (node, delta) in increment.iter().enumerate().take(state.node_count()) {
        if system.is_dynamic_node(node) {
            state.positions[node].x += scale * delta[0];
            state.positions[node].y += scale * delta[1];
            state.velocities[node].x += scale * delta[2];
            state.velocities[node].y += scale * delta[3];
        }
        if node < state.material_state.len() {
            state.material_state[node] += scale * delta[4];
        }
    }
    system.enforce_kinematics(state, stage_time);
}

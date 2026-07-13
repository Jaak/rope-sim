use faer::Col;
use faer::prelude::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};

use crate::math::Vec2;
use crate::state::State;

use super::{
    AccelerationJacobianBlock, DynamicalSystem, IntegratorStatistics, StepError, TimeIntegrator,
    validate_timestep,
};

const GAMMA: f64 = 2.0 - std::f64::consts::SQRT_2;
const DIAGONAL_COEFFICIENT: f64 = 0.5 * GAMMA;
const BDF_STAGE_WEIGHT: f64 = 1.0 / (GAMMA * (2.0 - GAMMA));
const INITIAL_STAGE_WEIGHT: f64 = 1.0 - BDF_STAGE_WEIGHT;
const MAX_NEWTON_ITERATIONS: usize = 12;
const MAX_LINE_SEARCH_ITERATIONS: usize = 10;
const MAX_ADAPTIVE_RETRY_LEVELS: usize = 4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const NOT_DYNAMIC: usize = usize::MAX;

pub(super) struct TrBdf2 {
    backup: State,
    initial: State,
    trapezoidal: State,
    material_base: State,
    position_reference: Vec<Vec2>,
    predictor: Vec<Vec2>,
    accelerations: Vec<Vec2>,
    material_derivatives: Vec<f64>,
    solver: StageSolver,
    statistics: IntegratorStatistics,
}

impl TrBdf2 {
    pub fn new(node_count: usize) -> Self {
        let positions = vec![Vec2::ZERO; node_count];
        Self {
            backup: State::new(positions.clone()),
            initial: State::new(positions.clone()),
            trapezoidal: State::new(positions.clone()),
            material_base: State::new(positions),
            position_reference: vec![Vec2::ZERO; node_count],
            predictor: vec![Vec2::ZERO; node_count],
            accelerations: vec![Vec2::ZERO; node_count],
            material_derivatives: vec![0.0; node_count.saturating_sub(1)],
            solver: StageSolver::new(node_count),
            statistics: IntegratorStatistics::default(),
        }
    }

    fn solve_once(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
        start_time: f64,
    ) -> Result<(), StepError> {
        system.enforce_kinematics(state, start_time);
        self.initial.clone_from(state);
        let stage_dt = DIAGONAL_COEFFICIENT * dt;

        self.accelerations.resize(state.node_count(), Vec2::ZERO);
        self.accelerations.fill(Vec2::ZERO);
        system.accelerations(&self.initial, &mut self.accelerations);
        self.material_derivatives
            .resize(state.node_count().saturating_sub(1), 0.0);
        self.material_derivatives.fill(0.0);
        system.material_state_derivatives(&self.initial, &mut self.material_derivatives);

        self.position_reference
            .resize(state.node_count(), Vec2::ZERO);
        self.predictor.resize(state.node_count(), Vec2::ZERO);
        self.material_base.clone_from(&self.initial);
        for node in 0..state.node_count() {
            self.position_reference[node] =
                self.initial.positions[node] + self.initial.velocities[node] * stage_dt;
            self.predictor[node] = self.initial.positions[node]
                + self.initial.velocities[node] * (2.0 * stage_dt)
                + self.accelerations[node] * (stage_dt * stage_dt);
        }
        for (base, derivative) in self
            .material_base
            .material_state
            .iter_mut()
            .zip(&self.material_derivatives)
        {
            *base += stage_dt * derivative;
        }

        self.trapezoidal.clone_from(&self.initial);
        self.solver.solve(
            system,
            &mut self.trapezoidal,
            &self.material_base,
            &self.position_reference,
            &self.predictor,
            stage_dt,
            start_time + 2.0 * stage_dt,
            &mut self.statistics,
        )?;

        self.material_base.clone_from(&self.initial);
        for node in 0..state.node_count() {
            self.position_reference[node] = self.initial.positions[node] * INITIAL_STAGE_WEIGHT
                + self.trapezoidal.positions[node] * BDF_STAGE_WEIGHT;
            let base_velocity = self.initial.velocities[node] * INITIAL_STAGE_WEIGHT
                + self.trapezoidal.velocities[node] * BDF_STAGE_WEIGHT;
            self.predictor[node] = self.position_reference[node] + base_velocity * stage_dt;
        }
        for element in 0..self.material_base.material_state.len() {
            self.material_base.material_state[element] = self.initial.material_state[element]
                * INITIAL_STAGE_WEIGHT
                + self.trapezoidal.material_state[element] * BDF_STAGE_WEIGHT;
        }

        state.clone_from(&self.trapezoidal);
        self.solver.solve(
            system,
            state,
            &self.material_base,
            &self.position_reference,
            &self.predictor,
            stage_dt,
            start_time + dt,
            &mut self.statistics,
        )
    }
}

impl TimeIntegrator for TrBdf2 {
    fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;
        system.enforce_kinematics(state, 0.0);
        self.backup.clone_from(state);
        let mut last_error = StepError::NewtonDidNotConverge {
            iterations: 0,
            residual: f64::INFINITY,
        };

        for retry_level in 0..=MAX_ADAPTIVE_RETRY_LEVELS {
            if retry_level > 0 {
                self.statistics.adaptive_retries += 1;
            }
            state.clone_from(&self.backup);
            let subdivisions = 1_usize << retry_level;
            let substep_dt = dt / subdivisions as f64;
            let mut succeeded = true;
            for subdivision in 0..subdivisions {
                let start_time = subdivision as f64 * substep_dt;
                if let Err(error) = self.solve_once(system, state, substep_dt, start_time) {
                    self.statistics.rejected_steps += 1;
                    last_error = error;
                    succeeded = false;
                    break;
                }
            }
            if succeeded {
                return Ok(());
            }
        }

        state.clone_from(&self.backup);
        system.enforce_kinematics(state, 0.0);
        Err(last_error)
    }

    fn statistics(&self) -> IntegratorStatistics {
        self.statistics
    }

    fn recommended_substeps(
        &self,
        system: &dyn DynamicalSystem,
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        let boundary_sampling_limit = 0.75 * system.kinematic_timestep_limit();
        Ok((outer_dt / boundary_sampling_limit).ceil().max(1.0) as usize)
    }
}

struct StageSolver {
    trial: State,
    dynamic_nodes: Vec<usize>,
    factorized_dynamic_nodes: Vec<usize>,
    node_to_unknown: Vec<usize>,
    accelerations: Vec<Vec2>,
    acceleration_jacobian: Vec<AccelerationJacobianBlock>,
    residual: Vec<f64>,
    candidate_residual: Vec<f64>,
    jacobian_triplets: Vec<Triplet<usize, usize, f64>>,
    jacobian_matrix: Option<SparseColMat<usize, f64>>,
    symbolic_lu: Option<SymbolicLu<usize>>,
    delta: Vec<f64>,
    base_unknowns: Vec<f64>,
    block_solver: Block2Solver,
}

impl StageSolver {
    fn new(node_count: usize) -> Self {
        Self {
            trial: State::new(vec![Vec2::ZERO; node_count]),
            dynamic_nodes: Vec::with_capacity(node_count),
            factorized_dynamic_nodes: Vec::with_capacity(node_count),
            node_to_unknown: vec![NOT_DYNAMIC; node_count],
            accelerations: vec![Vec2::ZERO; node_count],
            acceleration_jacobian: Vec::with_capacity(5 * node_count),
            residual: Vec::new(),
            candidate_residual: Vec::new(),
            jacobian_triplets: Vec::with_capacity(20 * node_count),
            jacobian_matrix: None,
            symbolic_lu: None,
            delta: Vec::new(),
            base_unknowns: Vec::new(),
            block_solver: Block2Solver::new(node_count),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &mut self,
        system: &dyn DynamicalSystem,
        output: &mut State,
        material_base: &State,
        position_reference: &[Vec2],
        predictor: &[Vec2],
        dt: f64,
        stage_time: f64,
        statistics: &mut IntegratorStatistics,
    ) -> Result<(), StepError> {
        self.trial.clone_from(output);
        self.dynamic_nodes.clear();
        self.node_to_unknown
            .resize(output.node_count(), NOT_DYNAMIC);
        self.node_to_unknown.fill(NOT_DYNAMIC);
        for (node, &predicted_position) in predictor.iter().enumerate().take(output.node_count()) {
            if system.is_dynamic_node(node) {
                self.node_to_unknown[node] = 2 * self.dynamic_nodes.len();
                self.dynamic_nodes.push(node);
                self.trial.positions[node] = predicted_position;
            }
        }
        system.enforce_kinematics(&mut self.trial, stage_time);

        if self.factorized_dynamic_nodes != self.dynamic_nodes {
            self.factorized_dynamic_nodes
                .clone_from(&self.dynamic_nodes);
            self.jacobian_matrix = None;
            self.symbolic_lu = None;
        }
        let dimension = 2 * self.dynamic_nodes.len();
        self.accelerations.resize(output.node_count(), Vec2::ZERO);
        self.residual.resize(dimension, 0.0);
        self.candidate_residual.resize(dimension, 0.0);
        self.delta.resize(dimension, 0.0);
        self.base_unknowns.resize(dimension, 0.0);

        if dimension == 0 {
            system.prepare_implicit_state(material_base, &mut self.trial, dt);
            output.clone_from(&self.trial);
            return Ok(());
        }

        let mut last_residual = f64::INFINITY;
        for iteration in 0..MAX_NEWTON_ITERATIONS {
            statistics.nonlinear_iterations += 1;
            evaluate_residual(
                system,
                material_base,
                &mut self.trial,
                &self.dynamic_nodes,
                position_reference,
                predictor,
                dt,
                &mut self.accelerations,
                &mut self.residual,
                stage_time,
            );
            last_residual = infinity_norm(&self.residual);
            let scale = 1.0 + maximum_position_magnitude(&self.trial, &self.dynamic_nodes);
            if last_residual <= RESIDUAL_TOLERANCE * scale {
                output.clone_from(&self.trial);
                return Ok(());
            }

            for index in 0..dimension {
                let node = self.dynamic_nodes[index / 2];
                self.base_unknowns[index] = component_value(self.trial.positions[node], index % 2);
            }
            system.implicit_acceleration_jacobian(
                material_base,
                &self.trial,
                dt,
                &mut self.acceleration_jacobian,
            );
            if self
                .block_solver
                .solve(
                    &self.acceleration_jacobian,
                    &self.node_to_unknown,
                    &self.residual,
                    dt,
                    &mut self.delta,
                )
                .is_err()
            {
                assemble_residual_jacobian(
                    &self.acceleration_jacobian,
                    &self.node_to_unknown,
                    dimension,
                    dt,
                    &mut self.jacobian_triplets,
                );
                solve_sparse_system(
                    dimension,
                    &self.jacobian_triplets,
                    &self.residual,
                    &mut self.delta,
                    &mut self.jacobian_matrix,
                    &mut self.symbolic_lu,
                )?;
            }
            statistics.linear_solves += 1;

            let mut accepted = false;
            let mut step_scale = 1.0;
            for _ in 0..MAX_LINE_SEARCH_ITERATIONS {
                for index in 0..dimension {
                    let node = self.dynamic_nodes[index / 2];
                    set_component(
                        &mut self.trial.positions[node],
                        index % 2,
                        self.base_unknowns[index] + step_scale * self.delta[index],
                    );
                }
                evaluate_residual(
                    system,
                    material_base,
                    &mut self.trial,
                    &self.dynamic_nodes,
                    position_reference,
                    predictor,
                    dt,
                    &mut self.accelerations,
                    &mut self.candidate_residual,
                    stage_time,
                );
                if infinity_norm(&self.candidate_residual) < last_residual {
                    accepted = true;
                    break;
                }
                step_scale *= 0.5;
            }
            if !accepted {
                return Err(StepError::NewtonDidNotConverge {
                    iterations: iteration + 1,
                    residual: last_residual,
                });
            }
        }

        Err(StepError::NewtonDidNotConverge {
            iterations: MAX_NEWTON_ITERATIONS,
            residual: last_residual,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_residual(
    system: &dyn DynamicalSystem,
    material_base: &State,
    trial: &mut State,
    dynamic_nodes: &[usize],
    position_reference: &[Vec2],
    predictor: &[Vec2],
    dt: f64,
    accelerations: &mut [Vec2],
    output: &mut [f64],
    stage_time: f64,
) {
    for &node in dynamic_nodes {
        trial.velocities[node] = (trial.positions[node] - position_reference[node]) / dt;
    }
    system.enforce_kinematics(trial, stage_time);
    system.prepare_implicit_state(material_base, trial, dt);
    accelerations.fill(Vec2::ZERO);
    system.accelerations(trial, accelerations);
    let dt_squared = dt * dt;
    for (index, &node) in dynamic_nodes.iter().enumerate() {
        let residual = trial.positions[node] - predictor[node] - accelerations[node] * dt_squared;
        output[2 * index] = residual.x;
        output[2 * index + 1] = residual.y;
    }
}

type Matrix2 = [[f64; 2]; 2];
const ZERO_MATRIX2: Matrix2 = [[0.0; 2]; 2];

struct Block2Solver {
    lower: Vec<Matrix2>,
    diagonal: Vec<Matrix2>,
    reduced_upper: Vec<Matrix2>,
    right_hand_side: Vec<Vec2>,
}

impl Block2Solver {
    fn new(block_count: usize) -> Self {
        Self {
            lower: vec![ZERO_MATRIX2; block_count.saturating_sub(1)],
            diagonal: vec![ZERO_MATRIX2; block_count],
            reduced_upper: vec![ZERO_MATRIX2; block_count.saturating_sub(1)],
            right_hand_side: vec![Vec2::ZERO; block_count],
        }
    }

    fn solve(
        &mut self,
        acceleration_blocks: &[AccelerationJacobianBlock],
        node_to_unknown: &[usize],
        residual: &[f64],
        dt: f64,
        output: &mut [f64],
    ) -> Result<(), StepError> {
        let block_count = residual.len() / 2;
        self.lower
            .resize(block_count.saturating_sub(1), ZERO_MATRIX2);
        self.diagonal.resize(block_count, ZERO_MATRIX2);
        self.reduced_upper
            .resize(block_count.saturating_sub(1), ZERO_MATRIX2);
        self.right_hand_side.resize(block_count, Vec2::ZERO);
        self.lower.fill(ZERO_MATRIX2);
        self.diagonal.fill(ZERO_MATRIX2);
        self.reduced_upper.fill(ZERO_MATRIX2);
        for (block, diagonal) in self.diagonal.iter_mut().enumerate() {
            diagonal[0][0] = 1.0;
            diagonal[1][1] = 1.0;
            self.right_hand_side[block] = Vec2::new(-residual[2 * block], -residual[2 * block + 1]);
        }

        let dt_squared = dt * dt;
        for block in acceleration_blocks {
            let row_offset = node_to_unknown[block.row_node];
            let column_offset = node_to_unknown[block.column_node];
            if row_offset == NOT_DYNAMIC || column_offset == NOT_DYNAMIC {
                continue;
            }
            let row_block = row_offset / 2;
            let column_block = column_offset / 2;
            let target = if row_block == column_block {
                &mut self.diagonal[row_block]
            } else if row_block == column_block + 1 {
                &mut self.lower[column_block]
            } else if column_block == row_block + 1 {
                &mut self.reduced_upper[row_block]
            } else {
                return Err(StepError::SingularJacobian);
            };
            for (row, target_row) in target.iter_mut().enumerate() {
                for (column, target_value) in target_row.iter_mut().enumerate() {
                    *target_value += -dt_squared * block.position[row][column]
                        - dt * block.velocity[row][column];
                }
            }
        }

        // reduced_upper initially stores the original upper blocks. It is
        // overwritten in place with D_i^{-1} U_i during elimination.
        for index in 0..block_count {
            if index > 0 {
                self.diagonal[index] = subtract_matrix_product(
                    self.diagonal[index],
                    self.lower[index - 1],
                    self.reduced_upper[index - 1],
                );
                let previous = self.right_hand_side[index - 1];
                self.right_hand_side[index] -=
                    multiply_matrix_vector(self.lower[index - 1], previous);
            }
            self.right_hand_side[index] =
                solve_matrix2(self.diagonal[index], self.right_hand_side[index])?;
            if index + 1 < block_count {
                self.reduced_upper[index] =
                    solve_matrix2_columns(self.diagonal[index], self.reduced_upper[index])?;
            }
        }

        for index in (0..block_count.saturating_sub(1)).rev() {
            let next = self.right_hand_side[index + 1];
            self.right_hand_side[index] -= multiply_matrix_vector(self.reduced_upper[index], next);
        }
        for (block, solution) in self.right_hand_side.iter().enumerate() {
            output[2 * block] = solution.x;
            output[2 * block + 1] = solution.y;
        }
        if output.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(StepError::SingularJacobian)
        }
    }
}

fn solve_matrix2(matrix: Matrix2, rhs: Vec2) -> Result<Vec2, StepError> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let scale = matrix
        .iter()
        .flatten()
        .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
    if !determinant.is_finite() || determinant.abs() <= 64.0 * f64::EPSILON * scale * scale {
        return Err(StepError::SingularJacobian);
    }
    Ok(Vec2::new(
        (matrix[1][1] * rhs.x - matrix[0][1] * rhs.y) / determinant,
        (-matrix[1][0] * rhs.x + matrix[0][0] * rhs.y) / determinant,
    ))
}

fn solve_matrix2_columns(matrix: Matrix2, rhs: Matrix2) -> Result<Matrix2, StepError> {
    let first = solve_matrix2(matrix, Vec2::new(rhs[0][0], rhs[1][0]))?;
    let second = solve_matrix2(matrix, Vec2::new(rhs[0][1], rhs[1][1]))?;
    Ok([[first.x, second.x], [first.y, second.y]])
}

fn multiply_matrix_vector(matrix: Matrix2, vector: Vec2) -> Vec2 {
    Vec2::new(
        matrix[0][0] * vector.x + matrix[0][1] * vector.y,
        matrix[1][0] * vector.x + matrix[1][1] * vector.y,
    )
}

fn subtract_matrix_product(base: Matrix2, left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut output = base;
    for row in 0..2 {
        for column in 0..2 {
            output[row][column] -=
                left[row][0] * right[0][column] + left[row][1] * right[1][column];
        }
    }
    output
}

fn assemble_residual_jacobian(
    blocks: &[AccelerationJacobianBlock],
    node_to_unknown: &[usize],
    dimension: usize,
    dt: f64,
    output: &mut Vec<Triplet<usize, usize, f64>>,
) {
    output.clear();
    for index in 0..dimension {
        output.push(Triplet::new(index, index, 1.0));
    }
    let dt_squared = dt * dt;
    for block in blocks {
        let row_offset = node_to_unknown[block.row_node];
        let column_offset = node_to_unknown[block.column_node];
        if row_offset == NOT_DYNAMIC || column_offset == NOT_DYNAMIC {
            continue;
        }
        for row in 0..2 {
            for column in 0..2 {
                output.push(Triplet::new(
                    row_offset + row,
                    column_offset + column,
                    -dt_squared * block.position[row][column] - dt * block.velocity[row][column],
                ));
            }
        }
    }
}

fn solve_sparse_system(
    dimension: usize,
    triplets: &[Triplet<usize, usize, f64>],
    residual: &[f64],
    output: &mut [f64],
    matrix_cache: &mut Option<SparseColMat<usize, f64>>,
    symbolic_cache: &mut Option<SymbolicLu<usize>>,
) -> Result<(), StepError> {
    if matrix_cache.is_none() {
        let matrix =
            SparseColMat::<usize, f64>::try_new_from_triplets(dimension, dimension, triplets)
                .map_err(|_| StepError::SingularJacobian)?;
        let symbolic =
            SymbolicLu::try_new(matrix.symbolic()).map_err(|_| StepError::SingularJacobian)?;
        *matrix_cache = Some(matrix);
        *symbolic_cache = Some(symbolic);
    } else {
        let matrix = matrix_cache.as_mut().expect("matrix cache initialized");
        matrix.val_mut().fill(0.0);
        for triplet in triplets {
            matrix[(triplet.row, triplet.col)] += triplet.val;
        }
    }
    let matrix = matrix_cache.as_ref().expect("matrix cache initialized");
    let symbolic = symbolic_cache.as_ref().expect("symbolic cache initialized");
    let factorization = Lu::try_new_with_symbolic(symbolic.clone(), matrix.as_ref())
        .map_err(|_| StepError::SingularJacobian)?;
    let rhs = Col::from_fn(dimension, |index| -residual[index]);
    let solution = factorization.solve(&rhs);
    for index in 0..dimension {
        output[index] = solution[index];
    }
    if output.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(StepError::SingularJacobian)
    }
}

fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |norm, value| norm.max(value.abs()))
}

fn maximum_position_magnitude(state: &State, dynamic_nodes: &[usize]) -> f64 {
    dynamic_nodes.iter().fold(0.0, |maximum, &node| {
        maximum.max(state.positions[node].length())
    })
}

fn component_value(vector: Vec2, component: usize) -> f64 {
    if component == 0 { vector.x } else { vector.y }
}

fn set_component(vector: &mut Vec2, component: usize, value: f64) {
    if component == 0 {
        vector.x = value;
    } else {
        vector.y = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block2_solver_recovers_a_known_solution() {
        let diagonal = [[4.0, 0.2], [-0.1, 3.0]];
        let lower = [[-0.4, 0.1], [0.05, -0.3]];
        let upper = [[-0.2, -0.05], [0.1, -0.25]];
        let expected = [
            Vec2::new(0.5, -0.25),
            Vec2::new(1.0, 0.75),
            Vec2::new(-0.4, 1.2),
        ];
        let mut residual = vec![0.0; 2 * expected.len()];
        for row in 0..expected.len() {
            let mut value = multiply_matrix_vector(diagonal, expected[row]);
            if row > 0 {
                value += multiply_matrix_vector(lower, expected[row - 1]);
            }
            if row + 1 < expected.len() {
                value += multiply_matrix_vector(upper, expected[row + 1]);
            }
            residual[2 * row] = -value.x;
            residual[2 * row + 1] = -value.y;
        }

        let identity = [[1.0, 0.0], [0.0, 1.0]];
        let mut acceleration_blocks = Vec::new();
        for node in 0..expected.len() {
            acceleration_blocks.push(AccelerationJacobianBlock {
                row_node: node,
                column_node: node,
                position: subtract_matrix(identity, diagonal),
                velocity: ZERO_MATRIX2,
            });
            if node + 1 < expected.len() {
                acceleration_blocks.push(AccelerationJacobianBlock {
                    row_node: node,
                    column_node: node + 1,
                    position: negate_matrix(upper),
                    velocity: ZERO_MATRIX2,
                });
                acceleration_blocks.push(AccelerationJacobianBlock {
                    row_node: node + 1,
                    column_node: node,
                    position: negate_matrix(lower),
                    velocity: ZERO_MATRIX2,
                });
            }
        }

        let node_to_unknown = [0, 2, 4];
        let mut actual = vec![0.0; 2 * expected.len()];
        Block2Solver::new(expected.len())
            .solve(
                &acceleration_blocks,
                &node_to_unknown,
                &residual,
                1.0,
                &mut actual,
            )
            .unwrap();

        for (block, expected) in expected.iter().enumerate() {
            assert!((actual[2 * block] - expected.x).abs() < 1.0e-12);
            assert!((actual[2 * block + 1] - expected.y).abs() < 1.0e-12);
        }
    }

    fn subtract_matrix(left: Matrix2, right: Matrix2) -> Matrix2 {
        let mut output = ZERO_MATRIX2;
        for row in 0..2 {
            for column in 0..2 {
                output[row][column] = left[row][column] - right[row][column];
            }
        }
        output
    }

    fn negate_matrix(matrix: Matrix2) -> Matrix2 {
        let mut output = ZERO_MATRIX2;
        for row in 0..2 {
            for column in 0..2 {
                output[row][column] = -matrix[row][column];
            }
        }
        output
    }
}

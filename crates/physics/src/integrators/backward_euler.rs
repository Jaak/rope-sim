use faer::Col;
use faer::prelude::Solve;
use faer::sparse::linalg::solvers::{Lu, SymbolicLu};
use faer::sparse::{SparseColMat, Triplet};

use crate::math::Vec2;
use crate::state::State;

use super::newton_block_pentadiagonal::NewtonBlockPentadiagonalSolver;
use super::{
    AccelerationJacobianBlock, DynamicalSystem, IntegratorStatistics, StepError, TimeIntegrator,
    validate_timestep,
};

const MAX_NEWTON_ITERATIONS: usize = 12;
const MAX_LINE_SEARCH_ITERATIONS: usize = 10;
const MAX_ADAPTIVE_RETRY_LEVELS: usize = 4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const INTERACTION_RESIDUAL_TOLERANCE: f64 = 1.0e-5;
const NOT_DYNAMIC: usize = usize::MAX;

pub(crate) struct BackwardEuler {
    backup: State,
    initial: State,
    trial: State,
    alternative_trial: State,
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
    block_solver: NewtonBlockPentadiagonalSolver,
    delta: Vec<f64>,
    base_unknowns: Vec<f64>,
    statistics: IntegratorStatistics,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PredictorCorrection {
    Converged,
    BudgetExceeded { iterations: usize, residual: f64 },
}

impl BackwardEuler {
    pub fn new(node_count: usize) -> Self {
        let zero_positions = vec![Vec2::ZERO; node_count];
        Self {
            backup: State::new(zero_positions.clone()),
            initial: State::new(zero_positions.clone()),
            trial: State::new(zero_positions.clone()),
            alternative_trial: State::new(zero_positions),
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
            block_solver: NewtonBlockPentadiagonalSolver::new(node_count),
            delta: Vec::new(),
            base_unknowns: Vec::new(),
            statistics: IntegratorStatistics::default(),
        }
    }

    fn prepare(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
        start_time: f64,
    ) {
        system.enforce_kinematics(state, start_time);
        self.initial.clone_from(state);
        self.trial.clone_from(state);
        self.dynamic_nodes.clear();
        self.node_to_unknown.fill(NOT_DYNAMIC);

        for index in 0..state.node_count() {
            if system.is_dynamic_node(index) {
                self.node_to_unknown[index] = 2 * self.dynamic_nodes.len();
                self.dynamic_nodes.push(index);
                self.trial.positions[index] += self.initial.velocities[index] * dt;
            }
        }
        system.enforce_kinematics(&mut self.trial, start_time + dt);

        if self.factorized_dynamic_nodes != self.dynamic_nodes {
            self.factorized_dynamic_nodes
                .clone_from(&self.dynamic_nodes);
            self.jacobian_matrix = None;
            self.symbolic_lu = None;
        }

        let dimension = 2 * self.dynamic_nodes.len();
        self.accelerations.resize(state.node_count(), Vec2::ZERO);
        self.residual.resize(dimension, 0.0);
        self.candidate_residual.resize(dimension, 0.0);
        self.delta.resize(dimension, 0.0);
        self.base_unknowns.resize(dimension, 0.0);
    }

    fn prepare_from_predictor(
        &mut self,
        system: &dyn DynamicalSystem,
        initial: &State,
        predictor: &State,
        dt: f64,
        start_time: f64,
    ) {
        self.initial.clone_from(initial);
        system.enforce_kinematics(&mut self.initial, start_time);
        self.trial.clone_from(predictor);
        system.enforce_kinematics(&mut self.trial, start_time + dt);
        self.dynamic_nodes.clear();
        self.node_to_unknown.fill(NOT_DYNAMIC);

        for index in 0..initial.node_count() {
            if system.is_dynamic_node(index) {
                self.node_to_unknown[index] = 2 * self.dynamic_nodes.len();
                self.dynamic_nodes.push(index);
            }
        }

        if self.factorized_dynamic_nodes != self.dynamic_nodes {
            self.factorized_dynamic_nodes
                .clone_from(&self.dynamic_nodes);
            self.jacobian_matrix = None;
            self.symbolic_lu = None;
        }

        let dimension = 2 * self.dynamic_nodes.len();
        self.accelerations.resize(initial.node_count(), Vec2::ZERO);
        self.residual.resize(dimension, 0.0);
        self.candidate_residual.resize(dimension, 0.0);
        self.delta.resize(dimension, 0.0);
        self.base_unknowns.resize(dimension, 0.0);

        // XPBD is usually the better nonlinear seed during manipulation, but
        // at high resolution a partially converged local projection can be
        // farther from the constitutive residual than BE's inertial predictor.
        // Evaluate both transactionally and start Newton from the better one.
        self.alternative_trial.clone_from(&self.initial);
        for &node in &self.dynamic_nodes {
            self.alternative_trial.positions[node] += self.initial.velocities[node] * dt;
        }
        system.enforce_kinematics(&mut self.alternative_trial, start_time + dt);
        self.statistics.residual_evaluations += 2;
        evaluate_residual(
            system,
            &self.initial,
            &mut self.trial,
            &self.dynamic_nodes,
            dt,
            &mut self.accelerations,
            &mut self.residual,
            start_time + dt,
        );
        evaluate_residual(
            system,
            &self.initial,
            &mut self.alternative_trial,
            &self.dynamic_nodes,
            dt,
            &mut self.accelerations,
            &mut self.candidate_residual,
            start_time + dt,
        );
        if infinity_norm(&self.candidate_residual) < infinity_norm(&self.residual) {
            self.trial.clone_from(&self.alternative_trial);
        }
    }

    fn finish(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        end_time: f64,
    ) -> Result<(), StepError> {
        state.clone_from(&self.trial);
        system.enforce_kinematics(state, end_time);
        if state.is_finite() {
            Ok(())
        } else {
            Err(StepError::NonFiniteState)
        }
    }
}

impl BackwardEuler {
    pub(crate) fn correct_from_predictor(
        &mut self,
        system: &dyn DynamicalSystem,
        initial: &State,
        predictor: &State,
        output: &mut State,
        dt: f64,
        maximum_iterations: usize,
    ) -> Result<PredictorCorrection, StepError> {
        validate_timestep(dt)?;
        self.prepare_from_predictor(system, initial, predictor, dt, 0.0);
        self.solve_prepared(
            system,
            output,
            dt,
            0.0,
            maximum_iterations,
            INTERACTION_RESIDUAL_TOLERANCE,
        )
    }

    pub(crate) fn correct_from_predictor_fully(
        &mut self,
        system: &dyn DynamicalSystem,
        initial: &State,
        predictor: &State,
        output: &mut State,
        dt: f64,
    ) -> Result<PredictorCorrection, StepError> {
        validate_timestep(dt)?;
        self.prepare_from_predictor(system, initial, predictor, dt, 0.0);
        self.solve_prepared(
            system,
            output,
            dt,
            0.0,
            MAX_NEWTON_ITERATIONS,
            RESIDUAL_TOLERANCE,
        )
    }

    fn solve_once(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
        start_time: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;
        self.prepare(system, state, dt, start_time);
        match self.solve_prepared(
            system,
            state,
            dt,
            start_time,
            MAX_NEWTON_ITERATIONS,
            RESIDUAL_TOLERANCE,
        )? {
            PredictorCorrection::Converged => Ok(()),
            PredictorCorrection::BudgetExceeded {
                iterations,
                residual,
            } => Err(StepError::NewtonDidNotConverge {
                iterations,
                residual,
            }),
        }
    }

    fn solve_prepared(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
        start_time: f64,
        maximum_iterations: usize,
        residual_tolerance: f64,
    ) -> Result<PredictorCorrection, StepError> {
        let end_time = start_time + dt;

        if self.dynamic_nodes.is_empty() {
            system.update_implicit_trial_state(&self.initial, &mut self.trial, dt);
            self.finish(system, state, end_time)?;
            return Ok(PredictorCorrection::Converged);
        }

        let dimension = self.residual.len();
        let mut last_residual = f64::INFINITY;
        for iteration in 0..maximum_iterations {
            self.statistics.nonlinear_iterations += 1;
            self.statistics.residual_evaluations += 1;
            evaluate_residual(
                system,
                &self.initial,
                &mut self.trial,
                &self.dynamic_nodes,
                dt,
                &mut self.accelerations,
                &mut self.residual,
                end_time,
            );
            last_residual = infinity_norm(&self.residual);
            let scale = 1.0 + maximum_position_magnitude(&self.trial, &self.dynamic_nodes);
            if last_residual <= residual_tolerance * scale {
                self.finish(system, state, end_time)?;
                return Ok(PredictorCorrection::Converged);
            }

            for index in 0..dimension {
                let node = self.dynamic_nodes[index / 2];
                self.base_unknowns[index] = component_value(self.trial.positions[node], index % 2);
            }

            system.implicit_acceleration_jacobian(
                &self.initial,
                &self.trial,
                dt,
                &mut self.acceleration_jacobian,
            );
            self.statistics.jacobian_assemblies += 1;
            if self
                .block_solver
                .factorize(
                    &self.acceleration_jacobian,
                    &self.node_to_unknown,
                    self.dynamic_nodes.len(),
                    dt,
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
                self.statistics.sparse_factorizations += 1;
            } else {
                self.block_solver.solve(&self.residual, &mut self.delta)?;
                self.statistics.block_factorizations += 1;
            }
            self.statistics.linear_solves += 1;

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

                self.statistics.residual_evaluations += 1;
                evaluate_residual(
                    system,
                    &self.initial,
                    &mut self.trial,
                    &self.dynamic_nodes,
                    dt,
                    &mut self.accelerations,
                    &mut self.candidate_residual,
                    end_time,
                );
                if infinity_norm(&self.candidate_residual) < last_residual {
                    accepted = true;
                    break;
                }
                self.statistics.line_search_backtracks += 1;
                step_scale *= 0.5;
            }

            if !accepted {
                return Ok(PredictorCorrection::BudgetExceeded {
                    iterations: iteration + 1,
                    residual: last_residual,
                });
            }

            last_residual = infinity_norm(&self.candidate_residual);
            let candidate_scale =
                1.0 + maximum_position_magnitude(&self.trial, &self.dynamic_nodes);
            if last_residual <= residual_tolerance * candidate_scale {
                self.finish(system, state, end_time)?;
                return Ok(PredictorCorrection::Converged);
            }
        }

        Ok(PredictorCorrection::BudgetExceeded {
            iterations: maximum_iterations,
            residual: last_residual,
        })
    }
}

impl TimeIntegrator for BackwardEuler {
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
            let subdivision_count = 1_usize << retry_level;
            let substep_dt = dt / subdivision_count as f64;
            let mut succeeded = true;

            for subdivision in 0..subdivision_count {
                let start_time = subdivision as f64 * substep_dt;
                match self.solve_once(system, state, substep_dt, start_time) {
                    Ok(()) => {}
                    Err(StepError::InvalidTimeStep(invalid_dt)) => {
                        state.clone_from(&self.backup);
                        return Err(StepError::InvalidTimeStep(invalid_dt));
                    }
                    Err(error) => {
                        self.statistics.rejected_steps += 1;
                        last_error = error;
                        succeeded = false;
                        break;
                    }
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
        _state: &State,
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        let boundary_sampling_limit = 0.75 * system.kinematic_timestep_limit();
        Ok((outer_dt / boundary_sampling_limit).ceil().max(1.0) as usize)
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_residual(
    system: &dyn DynamicalSystem,
    initial: &State,
    trial: &mut State,
    dynamic_nodes: &[usize],
    dt: f64,
    accelerations: &mut [Vec2],
    output: &mut [f64],
    end_time: f64,
) {
    for &node in dynamic_nodes {
        trial.velocities[node] = (trial.positions[node] - initial.positions[node]) / dt;
    }
    system.enforce_kinematics(trial, end_time);
    system.update_implicit_trial_state(initial, trial, dt);
    accelerations.fill(Vec2::ZERO);
    system.accelerations(trial, accelerations);

    let dt_squared = dt * dt;
    for (index, &node) in dynamic_nodes.iter().enumerate() {
        let residual = trial.positions[node]
            - initial.positions[node]
            - initial.velocities[node] * dt
            - accelerations[node] * dt_squared;
        output[2 * index] = residual.x;
        output[2 * index + 1] = residual.y;
    }
}

fn assemble_residual_jacobian(
    acceleration_blocks: &[AccelerationJacobianBlock],
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
    for block in acceleration_blocks {
        let row_offset = node_to_unknown[block.row_node];
        let column_offset = node_to_unknown[block.column_node];
        if row_offset == NOT_DYNAMIC || column_offset == NOT_DYNAMIC {
            continue;
        }

        for row in 0..2 {
            for column in 0..2 {
                let value =
                    -dt_squared * block.position[row][column] - dt * block.velocity[row][column];
                output.push(Triplet::new(
                    row_offset + row,
                    column_offset + column,
                    value,
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
        let matrix = matrix_cache
            .as_mut()
            .expect("the sparse matrix cache was checked above");
        matrix.val_mut().fill(0.0);
        for triplet in triplets {
            matrix[(triplet.row, triplet.col)] += triplet.val;
        }
    }

    let matrix = matrix_cache
        .as_ref()
        .expect("the sparse matrix is initialized above");
    let symbolic = symbolic_cache
        .as_ref()
        .expect("the symbolic factorization is initialized with the matrix");
    let factorization = Lu::try_new_with_symbolic(symbolic.clone(), matrix.as_ref())
        .map_err(|_| StepError::SingularJacobian)?;
    let right_hand_side = Col::from_fn(dimension, |index| -residual[index]);
    let solution = factorization.solve(&right_hand_side);

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

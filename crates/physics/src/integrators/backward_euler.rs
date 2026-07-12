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

const MAX_NEWTON_ITERATIONS: usize = 12;
const MAX_LINE_SEARCH_ITERATIONS: usize = 10;
const MAX_ADAPTIVE_RETRY_LEVELS: usize = 4;
const RESIDUAL_TOLERANCE: f64 = 1.0e-8;
const NOT_DYNAMIC: usize = usize::MAX;

pub(super) struct BackwardEuler {
    backup: State,
    initial: State,
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
    statistics: IntegratorStatistics,
}

impl BackwardEuler {
    pub fn new(node_count: usize) -> Self {
        let zero_positions = vec![Vec2::ZERO; node_count];
        Self {
            backup: State::new(zero_positions.clone()),
            initial: State::new(zero_positions.clone()),
            trial: State::new(zero_positions),
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
            statistics: IntegratorStatistics::default(),
        }
    }

    fn prepare(&mut self, system: &dyn DynamicalSystem, state: &mut State, dt: f64) {
        system.enforce_kinematics(state);
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
        system.enforce_kinematics(&mut self.trial);

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

    fn finish(&mut self, system: &dyn DynamicalSystem, state: &mut State) -> Result<(), StepError> {
        state.clone_from(&self.trial);
        system.enforce_kinematics(state);
        if state.is_finite() {
            Ok(())
        } else {
            Err(StepError::NonFiniteState)
        }
    }
}

impl BackwardEuler {
    fn solve_once(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;
        self.prepare(system, state, dt);

        if self.dynamic_nodes.is_empty() {
            system.prepare_implicit_state(&self.initial, &mut self.trial, dt);
            return self.finish(system, state);
        }

        let dimension = self.residual.len();
        let mut last_residual = f64::INFINITY;
        for iteration in 0..MAX_NEWTON_ITERATIONS {
            self.statistics.nonlinear_iterations += 1;
            evaluate_residual(
                system,
                &self.initial,
                &mut self.trial,
                &self.dynamic_nodes,
                dt,
                &mut self.accelerations,
                &mut self.residual,
            );
            last_residual = infinity_norm(&self.residual);
            let scale = 1.0 + maximum_position_magnitude(&self.trial, &self.dynamic_nodes);
            if last_residual <= RESIDUAL_TOLERANCE * scale {
                return self.finish(system, state);
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

                evaluate_residual(
                    system,
                    &self.initial,
                    &mut self.trial,
                    &self.dynamic_nodes,
                    dt,
                    &mut self.accelerations,
                    &mut self.candidate_residual,
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

impl TimeIntegrator for BackwardEuler {
    fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;
        system.enforce_kinematics(state);
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

            for _ in 0..subdivision_count {
                match self.solve_once(system, state, substep_dt) {
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
        system.enforce_kinematics(state);
        Err(last_error)
    }

    fn statistics(&self) -> IntegratorStatistics {
        self.statistics
    }
}

fn evaluate_residual(
    system: &dyn DynamicalSystem,
    initial: &State,
    trial: &mut State,
    dynamic_nodes: &[usize],
    dt: f64,
    accelerations: &mut [Vec2],
    output: &mut [f64],
) {
    for &node in dynamic_nodes {
        trial.velocities[node] = (trial.positions[node] - initial.positions[node]) / dt;
    }
    system.enforce_kinematics(trial);
    system.prepare_implicit_state(initial, trial, dt);
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

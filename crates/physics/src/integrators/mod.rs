mod backward_euler;
mod newton_block_pentadiagonal;
mod rk4;
mod semi_implicit_euler;
mod tr_bdf2;

use std::error::Error;
use std::fmt;

use crate::materials::StandardLinearSolidStateDerivative;
use crate::math::{Matrix2, Vec2};
use crate::state::State;

pub(crate) use backward_euler::{BackwardEuler, PredictorCorrection};
use rk4::RungeKutta4;
use semi_implicit_euler::SemiImplicitEuler;
use tr_bdf2::TrBdf2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AccelerationJacobianBlock {
    pub row_node: usize,
    pub column_node: usize,
    /// Derivative of row-node acceleration with respect to column-node position.
    pub position: Matrix2,
    /// Derivative of row-node acceleration with respect to column-node velocity.
    pub velocity: Matrix2,
}

pub(crate) trait DynamicalSystem {
    /// Evaluate node accelerations at `state`.
    fn accelerations(&self, state: &State, output: &mut [Vec2]);

    /// Evaluate derivatives of per-element constitutive state for explicit
    /// integrators.
    fn sls_state_derivatives(
        &self,
        state: &State,
        output: &mut Vec<StandardLinearSolidStateDerivative>,
    );

    /// Derive constitutive state for a backward-Euler trial from the last
    /// committed stage. This may be called repeatedly during Newton and must
    /// not mutate `committed`; `trial` is committed only when the stage succeeds.
    fn update_implicit_trial_state(&self, committed: &State, trial: &mut State, dt: f64);

    /// Acceleration Jacobian after eliminating implicit constitutive state.
    fn implicit_acceleration_jacobian(
        &self,
        initial: &State,
        state: &State,
        dt: f64,
        output: &mut Vec<AccelerationJacobianBlock>,
    );

    /// Return whether the integrator should advance this node.
    fn is_dynamic_node(&self, index: usize) -> bool;

    /// Reapply fixed and prescribed state at a stage time measured from the
    /// beginning of the current integration step.
    fn enforce_kinematics(&self, state: &mut State, stage_time: f64);

    /// Conservative stability limit for explicit integration of this system.
    fn explicit_stable_timestep(&self, state: &State) -> f64;

    /// Conservative elastic timescale, excluding viscous and constitutive
    /// relaxation restrictions that a linearly implicit method handles directly.
    fn elastic_stable_timestep(&self, state: &State) -> f64;

    /// Maximum step over which prescribed endpoint travel remains local to an
    /// element. Returns infinity when no endpoint is moving kinematically.
    fn kinematic_timestep_limit(&self) -> f64;
}

pub(crate) trait TimeIntegrator {
    fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError>;

    fn recommended_substeps(
        &self,
        _system: &dyn DynamicalSystem,
        _state: &State,
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        Ok(1)
    }

    fn statistics(&self) -> IntegratorStatistics {
        IntegratorStatistics::default()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct IntegratorStatistics {
    pub rejected_steps: u64,
    pub linear_solves: u64,
    pub nonlinear_iterations: u64,
    pub adaptive_retries: u64,
    pub residual_evaluations: u64,
    pub jacobian_assemblies: u64,
    pub block_factorizations: u64,
    pub sparse_factorizations: u64,
    pub line_search_backtracks: u64,
    pub failed_line_searches: u64,
    pub stagnation_acceptances: u64,
    pub maximum_retry_level: u64,
    pub last_velocity_residual: f64,
    pub last_normalized_residual: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegratorKind {
    SemiImplicitEuler,
    RungeKutta4,
    TrBdf2,
    #[default]
    BackwardEuler,
}

impl IntegratorKind {
    pub const ALL: [Self; 4] = [
        Self::SemiImplicitEuler,
        Self::RungeKutta4,
        Self::TrBdf2,
        Self::BackwardEuler,
    ];

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::SemiImplicitEuler => "Semi-implicit Euler",
            Self::RungeKutta4 => "Runge-Kutta 4",
            Self::TrBdf2 => "TR-BDF2",
            Self::BackwardEuler => "Backward Euler",
        }
    }
}

pub(crate) fn create_integrator(
    kind: IntegratorKind,
    node_count: usize,
) -> Box<dyn TimeIntegrator> {
    match kind {
        IntegratorKind::SemiImplicitEuler => Box::new(SemiImplicitEuler::new(node_count)),
        IntegratorKind::RungeKutta4 => Box::new(RungeKutta4::new(node_count)),
        IntegratorKind::TrBdf2 => Box::new(TrBdf2::new(node_count)),
        IntegratorKind::BackwardEuler => Box::new(BackwardEuler::new(node_count)),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepError {
    InvalidTimeStep(f64),
    NonFiniteState,
    SingularJacobian,
    NewtonDidNotConverge { iterations: usize, residual: f64 },
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTimeStep(dt) => {
                write!(f, "time step must be finite and positive (received {dt})")
            }
            Self::NonFiniteState => write!(f, "the simulation produced a non-finite state"),
            Self::SingularJacobian => write!(f, "the implicit Jacobian is singular"),
            Self::NewtonDidNotConverge {
                iterations,
                residual,
            } => write!(
                f,
                "Newton solve did not converge after {iterations} iterations (residual {residual:.3e})"
            ),
        }
    }
}

impl Error for StepError {}

pub(super) fn validate_timestep(dt: f64) -> Result<(), StepError> {
    if !dt.is_finite() || dt <= 0.0 {
        Err(StepError::InvalidTimeStep(dt))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VelocityResidualTolerance {
    pub absolute: f64,
    pub relative: f64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VelocityResidualMetrics {
    /// Maximum two-dimensional node defect, in metres per second.
    pub absolute: f64,
    /// Maximum defect divided by that node's absolute/relative tolerance.
    pub normalized: f64,
}

/// Measure a position-form implicit residual as a per-node velocity defect.
///
/// Scaling the residual and Jacobian by `1 / dt` leaves the Newton correction
/// unchanged. Each 2D node is weighted independently so a fast payload cannot
/// loosen convergence for stationary nodes elsewhere in the rope.
pub(super) fn velocity_residual_metrics(
    position_residual: &[f64],
    dt: f64,
    reference: &State,
    trial: &State,
    dynamic_nodes: &[usize],
    tolerance: VelocityResidualTolerance,
) -> VelocityResidualMetrics {
    debug_assert_eq!(position_residual.len(), 2 * dynamic_nodes.len());
    let mut absolute: f64 = 0.0;
    let mut normalized: f64 = 0.0;

    for (components, &node) in position_residual.chunks_exact(2).zip(dynamic_nodes) {
        let defect = Vec2::new(components[0], components[1]).length() / dt;
        let speed = reference.velocities[node]
            .length()
            .max(trial.velocities[node].length());
        let scale = tolerance.absolute + tolerance.relative * speed;
        absolute = absolute.max(defect);
        normalized = normalized.max(defect / scale);
    }

    VelocityResidualMetrics {
        absolute,
        normalized,
    }
}

pub(super) fn infinity_norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |norm, value| norm.max(value.abs()))
}

pub(super) fn record_converged_residual(
    statistics: &mut IntegratorStatistics,
    residual: VelocityResidualMetrics,
) {
    statistics.last_velocity_residual = residual.absolute;
    statistics.last_normalized_residual = residual.normalized;
}

#[cfg(test)]
mod tests {
    use super::{VelocityResidualTolerance, velocity_residual_metrics};
    use crate::math::Vec2;
    use crate::state::State;

    #[test]
    fn velocity_residual_is_weighted_per_two_dimensional_node() {
        let mut reference = State::new(vec![Vec2::ZERO; 2]);
        reference.velocities[1] = Vec2::new(100.0, 0.0);
        let trial = reference.clone();
        let tolerance = VelocityResidualTolerance {
            absolute: 1.0e-6,
            relative: 1.0e-6,
        };

        // The stationary node has a 3-4-5 micrometre-per-second defect. The
        // unrelated fast node must not loosen its tolerance.
        let metrics = velocity_residual_metrics(
            &[3.0e-6, 4.0e-6, 0.0, 0.0],
            1.0,
            &reference,
            &trial,
            &[0, 1],
            tolerance,
        );

        assert!((metrics.absolute - 5.0e-6).abs() < 1.0e-15);
        assert!((metrics.normalized - 5.0).abs() < 1.0e-12);
    }
}

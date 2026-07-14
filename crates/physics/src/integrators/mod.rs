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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

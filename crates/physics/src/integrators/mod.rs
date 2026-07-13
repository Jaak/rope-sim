mod backward_euler;
pub(crate) mod block_tridiagonal;
mod rk4;
mod rosenbrock;
mod semi_implicit_euler;
mod tr_bdf2;

use std::error::Error;
use std::fmt;

use crate::math::Vec2;
use crate::state::State;

use block_tridiagonal::BlockTridiagonalMatrix;

use backward_euler::BackwardEuler;
use rk4::RungeKutta4;
use rosenbrock::Rosenbrock2;
use semi_implicit_euler::SemiImplicitEuler;
use tr_bdf2::TrBdf2;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AccelerationJacobianBlock {
    pub row_node: usize,
    pub column_node: usize,
    /// Derivative of row-node acceleration with respect to column-node position.
    pub position: [[f64; 2]; 2],
    /// Derivative of row-node acceleration with respect to column-node velocity.
    pub velocity: [[f64; 2]; 2],
}

pub(crate) trait DynamicalSystem {
    /// Evaluate node accelerations at `state`.
    fn accelerations(&self, state: &State, output: &mut [Vec2]);

    /// Evaluate derivatives of per-element constitutive state for explicit
    /// integrators.
    fn material_state_derivatives(&self, state: &State, output: &mut [f64]);

    /// Assemble the Jacobian of the complete first-order state derivative in
    /// node/element blocks for linearly implicit integration.
    fn first_order_jacobian(&self, state: &State, output: &mut BlockTridiagonalMatrix);

    /// Update constitutive state consistently with a backward-Euler trial.
    fn prepare_implicit_state(&self, initial: &State, trial: &mut State, dt: f64);

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
    fn explicit_stable_timestep(&self) -> f64;

    /// Conservative elastic timescale, excluding viscous and constitutive
    /// relaxation restrictions that a linearly implicit method handles directly.
    fn elastic_stable_timestep(&self) -> f64;

    /// Whether an endpoint is currently following a prescribed trajectory.
    fn has_kinematic_target(&self) -> bool;

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
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntegratorKind {
    #[default]
    SemiImplicitEuler,
    RungeKutta4,
    Rosenbrock2,
    TrBdf2,
    BackwardEuler,
}

impl IntegratorKind {
    pub const ALL: [Self; 5] = [
        Self::SemiImplicitEuler,
        Self::RungeKutta4,
        Self::Rosenbrock2,
        Self::TrBdf2,
        Self::BackwardEuler,
    ];

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::SemiImplicitEuler => "Semi-implicit Euler",
            Self::RungeKutta4 => "Runge-Kutta 4",
            Self::Rosenbrock2 => "Rosenbrock ROS2 (experimental)",
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
        IntegratorKind::Rosenbrock2 => Box::new(Rosenbrock2::new(node_count)),
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

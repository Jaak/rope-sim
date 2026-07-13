//! Platform-independent physics for the RopeSim application.

#![forbid(unsafe_code)]

mod config;
mod dynamics;
mod integrators;
mod kinematics;
mod materials;
mod math;
mod recorded_scenario;
mod simulation;
mod state;

pub use config::{ConfigError, RopeModelKind, SimulationConfig};
pub use integrators::{IntegratorKind, StepError};
pub use kinematics::KinematicTarget;
pub use math::Vec2;
pub use recorded_scenario::{
    MotionCommand, RecordedScenario, ScenarioFormatError, TimedMotionCommand,
};
pub use simulation::{Diagnostics, ReconfigureOutcome, Simulation};

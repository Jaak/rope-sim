//! Platform-independent physics for the RopeSim application.

#![forbid(unsafe_code)]

mod calibration;
mod config;
mod dynamics;
mod integrators;
mod kinematics;
mod materials;
mod math;
mod recorded_scenario;
mod simulation;
mod state;
mod xpbd;

pub use calibration::{
    CalibrationError, CalibrationMeasurements, CalibrationSettings, DynamicRopeReference,
    VOLTA_GUIDE_9MM, run_dynamic_rope_calibration,
};
pub use config::{ConfigError, RopeModelKind, SimulationConfig};
pub use integrators::{IntegratorKind, StepError};
pub use kinematics::KinematicTarget;
pub use math::Vec2;
pub use recorded_scenario::{
    MotionCommand, RecordedScenario, ScenarioFormatError, TimedMotionCommand,
};
pub use simulation::{Diagnostics, ReconfigureOutcome, Simulation};

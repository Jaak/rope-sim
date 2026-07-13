use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{IntegratorKind, KinematicTarget, SimulationConfig, Vec2};

pub const CURRENT_SCENARIO_FORMAT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionCommand {
    SetTarget(KinematicTarget),
    InterpolateTarget {
        target: KinematicTarget,
        duration: f64,
    },
    Release {
        velocity: Vec2,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimedMotionCommand {
    pub time: f64,
    pub command: MotionCommand,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedScenario {
    pub format_version: u32,
    pub config: SimulationConfig,
    pub fixed_time_step: f64,
    pub duration: f64,
    pub test_integrators: Vec<IntegratorKind>,
    pub commands: Vec<TimedMotionCommand>,
}

impl RecordedScenario {
    pub fn new(config: SimulationConfig, fixed_time_step: f64) -> Self {
        Self {
            format_version: CURRENT_SCENARIO_FORMAT_VERSION,
            config,
            fixed_time_step,
            duration: 0.0,
            test_integrators: vec![IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2],
            commands: Vec::new(),
        }
    }

    pub fn to_json_pretty(&self) -> Result<String, ScenarioFormatError> {
        self.validate()?;
        serde_json::to_string_pretty(self).map_err(ScenarioFormatError::Json)
    }

    pub fn from_json(json: &str) -> Result<Self, ScenarioFormatError> {
        let scenario: Self = serde_json::from_str(json).map_err(ScenarioFormatError::Json)?;
        scenario.validate()?;
        Ok(scenario)
    }

    pub fn validate(&self) -> Result<(), ScenarioFormatError> {
        if self.format_version != CURRENT_SCENARIO_FORMAT_VERSION {
            return Err(ScenarioFormatError::Validation(format!(
                "unsupported scenario format version {} (expected {})",
                self.format_version, CURRENT_SCENARIO_FORMAT_VERSION
            )));
        }
        self.config
            .validate()
            .map_err(|error| ScenarioFormatError::Validation(error.to_string()))?;
        if !self.fixed_time_step.is_finite() || self.fixed_time_step <= 0.0 {
            return Err(ScenarioFormatError::Validation(
                "fixed_time_step must be finite and positive".to_owned(),
            ));
        }
        if !self.duration.is_finite() || self.duration < 0.0 {
            return Err(ScenarioFormatError::Validation(
                "duration must be finite and nonnegative".to_owned(),
            ));
        }
        if self.test_integrators.is_empty() {
            return Err(ScenarioFormatError::Validation(
                "test_integrators must contain at least one integrator".to_owned(),
            ));
        }

        let mut previous_time = 0.0;
        for (index, timed) in self.commands.iter().enumerate() {
            if !timed.time.is_finite() || timed.time < 0.0 || timed.time > self.duration + 1.0e-12 {
                return Err(ScenarioFormatError::Validation(format!(
                    "command {index} has a time outside the scenario duration"
                )));
            }
            if index > 0 && timed.time < previous_time {
                return Err(ScenarioFormatError::Validation(format!(
                    "command {index} is out of chronological order"
                )));
            }
            previous_time = timed.time;
            validate_command(index, timed.command)?;
        }

        Ok(())
    }
}

fn validate_command(index: usize, command: MotionCommand) -> Result<(), ScenarioFormatError> {
    let invalid =
        |message: &str| ScenarioFormatError::Validation(format!("command {index} {message}"));
    match command {
        MotionCommand::SetTarget(target) => {
            if !target.position.is_finite() || !target.velocity.is_finite() {
                return Err(invalid("contains a non-finite target"));
            }
        }
        MotionCommand::InterpolateTarget { target, duration } => {
            if !target.position.is_finite() || !target.velocity.is_finite() {
                return Err(invalid("contains a non-finite target"));
            }
            if !duration.is_finite() || duration <= 0.0 {
                return Err(invalid("has an invalid interpolation duration"));
            }
        }
        MotionCommand::Release { velocity } => {
            if !velocity.is_finite() {
                return Err(invalid("contains a non-finite release velocity"));
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum ScenarioFormatError {
    Json(serde_json::Error),
    Validation(String),
}

impl fmt::Display for ScenarioFormatError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(formatter, "invalid scenario JSON: {error}"),
            Self::Validation(message) => formatter.write_str(message),
        }
    }
}

impl Error for ScenarioFormatError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            Self::Validation(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{MotionCommand, RecordedScenario, ScenarioFormatError, TimedMotionCommand};
    use crate::{KinematicTarget, SimulationConfig, Vec2};

    #[test]
    fn scenario_json_round_trip_preserves_commands_and_configuration() {
        let mut scenario = RecordedScenario::new(SimulationConfig::default(), 1.0 / 240.0);
        scenario.duration = 0.5;
        scenario.commands.push(TimedMotionCommand {
            time: 0.25,
            command: MotionCommand::SetTarget(KinematicTarget::new(
                Vec2::new(1.0, -2.0),
                Vec2::new(3.0, 4.0),
            )),
        });

        let json = scenario.to_json_pretty().unwrap();
        assert_eq!(RecordedScenario::from_json(&json).unwrap(), scenario);
    }

    #[test]
    fn scenario_rejects_commands_after_its_duration() {
        let mut scenario = RecordedScenario::new(SimulationConfig::default(), 1.0 / 240.0);
        scenario.duration = 0.5;
        scenario.commands.push(TimedMotionCommand {
            time: 0.6,
            command: MotionCommand::Release {
                velocity: Vec2::ZERO,
            },
        });

        assert!(matches!(
            scenario.validate(),
            Err(ScenarioFormatError::Validation(_))
        ));
    }
}

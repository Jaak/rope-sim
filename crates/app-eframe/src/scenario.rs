use ropesim_physics::{
    IntegratorKind, MotionCommand, RecordedScenario, ScenarioFormatError, SimulationConfig,
    TimedMotionCommand,
};

const TIME_EPSILON: f64 = 1.0e-12;

#[derive(Clone, Debug, Default)]
enum Activity {
    #[default]
    Idle,
    Recording(RecordedScenario),
    Replaying {
        scenario: RecordedScenario,
        next_command: usize,
    },
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ScenarioController {
    saved: Option<RecordedScenario>,
    activity: Activity,
}

impl ScenarioController {
    pub(crate) fn begin_recording(&mut self, config: SimulationConfig, fixed_time_step: f64) {
        self.activity = Activity::Recording(RecordedScenario::new(config, fixed_time_step));
    }

    pub(crate) fn record(&mut self, time: f64, command: MotionCommand) {
        if let Activity::Recording(scenario) = &mut self.activity {
            scenario.commands.push(TimedMotionCommand { time, command });
        }
    }

    pub(crate) fn stop(&mut self, time: f64) {
        match std::mem::take(&mut self.activity) {
            Activity::Recording(mut scenario) => {
                scenario.duration = time.max(0.0);
                self.saved = Some(scenario);
            }
            Activity::Idle | Activity::Replaying { .. } => {}
        }
    }

    pub(crate) fn recorded_config_for(
        &self,
        integrator: IntegratorKind,
    ) -> Option<SimulationConfig> {
        self.saved.as_ref().map(|scenario| SimulationConfig {
            integrator,
            ..scenario.config
        })
    }

    pub(crate) fn begin_replay(&mut self) -> bool {
        let Some(scenario) = self.saved.clone() else {
            return false;
        };
        self.activity = Activity::Replaying {
            scenario,
            next_command: 0,
        };
        true
    }

    pub(crate) fn saved_json(&self) -> Result<Option<String>, ScenarioFormatError> {
        self.saved
            .as_ref()
            .map(RecordedScenario::to_json_pretty)
            .transpose()
    }

    pub(crate) fn load_json(&mut self, json: &str) -> Result<(), ScenarioFormatError> {
        let scenario = RecordedScenario::from_json(json)?;
        self.activity = Activity::Idle;
        self.saved = Some(scenario);
        Ok(())
    }

    pub(crate) fn take_due_commands(&mut self, time: f64) -> Vec<MotionCommand> {
        let Activity::Replaying {
            scenario,
            next_command,
        } = &mut self.activity
        else {
            return Vec::new();
        };

        let start = *next_command;
        while *next_command < scenario.commands.len()
            && scenario.commands[*next_command].time <= time + TIME_EPSILON
        {
            *next_command += 1;
        }

        scenario.commands[start..*next_command]
            .iter()
            .map(|timed| timed.command)
            .collect()
    }

    pub(crate) fn finish_replay_if_complete(&mut self, time: f64) -> bool {
        let complete = matches!(
            &self.activity,
            Activity::Replaying { scenario, .. }
                if time + TIME_EPSILON >= scenario.duration
        );
        if complete {
            self.activity = Activity::Idle;
        }
        complete
    }

    pub(crate) fn is_recording(&self) -> bool {
        matches!(self.activity, Activity::Recording(_))
    }

    pub(crate) fn is_replaying(&self) -> bool {
        matches!(self.activity, Activity::Replaying { .. })
    }

    pub(crate) fn is_active(&self) -> bool {
        !matches!(self.activity, Activity::Idle)
    }

    pub(crate) fn has_recording(&self) -> bool {
        self.saved.is_some()
    }

    pub(crate) fn recording_summary(&self) -> Option<(f64, usize)> {
        self.saved
            .as_ref()
            .map(|scenario| (scenario.duration, scenario.commands.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::ScenarioController;
    use ropesim_physics::{IntegratorKind, KinematicTarget, MotionCommand, SimulationConfig, Vec2};

    #[test]
    fn replay_preserves_configuration_except_for_integrator() {
        let config = SimulationConfig {
            segment_count: 37,
            integrator: IntegratorKind::BackwardEuler,
            ..SimulationConfig::default()
        };
        let mut controller = ScenarioController::default();
        controller.begin_recording(config, 1.0 / 240.0);
        controller.stop(2.0);

        let replay = controller
            .recorded_config_for(IntegratorKind::TrBdf2)
            .unwrap();
        assert_eq!(replay.segment_count, 37);
        assert_eq!(replay.integrator, IntegratorKind::TrBdf2);
    }

    #[test]
    fn replay_commands_are_released_at_their_recorded_physics_times() {
        let mut controller = ScenarioController::default();
        controller.begin_recording(SimulationConfig::default(), 1.0 / 240.0);
        let first = MotionCommand::SetTarget(KinematicTarget::new(Vec2::new(1.0, 2.0), Vec2::ZERO));
        let second = MotionCommand::Release {
            velocity: Vec2::new(3.0, 4.0),
        };
        controller.record(0.25, first);
        controller.record(0.5, second);
        controller.stop(0.75);
        assert!(controller.begin_replay());

        assert!(controller.take_due_commands(0.2).is_empty());
        assert_eq!(controller.take_due_commands(0.25), vec![first]);
        assert_eq!(controller.take_due_commands(0.49), Vec::new());
        assert_eq!(controller.take_due_commands(0.5), vec![second]);
        assert!(!controller.finish_replay_if_complete(0.74));
        assert!(controller.finish_replay_if_complete(0.75));
    }

    #[test]
    fn saved_json_can_be_loaded_into_a_fresh_controller() {
        let mut recorded = ScenarioController::default();
        recorded.begin_recording(SimulationConfig::default(), 1.0 / 240.0);
        recorded.record(
            0.1,
            MotionCommand::Release {
                velocity: Vec2::new(1.0, 0.0),
            },
        );
        recorded.stop(0.2);
        let json = recorded.saved_json().unwrap().unwrap();

        let mut loaded = ScenarioController::default();
        loaded.load_json(&json).unwrap();
        assert_eq!(loaded.recording_summary(), Some((0.2, 1)));
        assert!(loaded.begin_replay());
    }
}

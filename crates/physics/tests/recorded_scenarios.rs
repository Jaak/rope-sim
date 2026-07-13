use std::fs;
use std::path::{Path, PathBuf};

use ropesim_physics::{MotionCommand, RecordedScenario, Simulation};

const TIME_EPSILON: f64 = 1.0e-12;

#[test]
fn saved_scenarios_replay_without_solver_failures_or_non_finite_state() {
    for path in scenario_paths() {
        let json = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
        let scenario = RecordedScenario::from_json(&json)
            .unwrap_or_else(|error| panic!("invalid fixture {}: {error}", path.display()));

        for &integrator in &scenario.test_integrators {
            replay_scenario(&path, &scenario, integrator);
        }
    }
}

fn scenario_paths() -> Vec<PathBuf> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios");
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    paths
}

fn replay_scenario(
    path: &Path,
    scenario: &RecordedScenario,
    integrator: ropesim_physics::IntegratorKind,
) {
    let mut config = scenario.config;
    config.integrator = integrator;
    let mut simulation = Simulation::new(config).unwrap_or_else(|error| {
        panic!(
            "{} could not start with {}: {error}",
            path.display(),
            integrator.display_name()
        )
    });
    let mut next_command = 0;

    while simulation.diagnostics().simulation_time + TIME_EPSILON < scenario.duration {
        apply_due_commands(
            scenario,
            &mut simulation,
            &mut next_command,
            path,
            integrator,
        );

        let time = simulation.diagnostics().simulation_time;
        let base_dt = scenario.fixed_time_step.min(scenario.duration - time);
        let substeps = simulation
            .recommended_substeps(base_dt)
            .unwrap_or_else(|error| replay_panic(path, integrator, time, &error.to_string()));
        let substep_dt = base_dt / substeps as f64;
        for _ in 0..substeps {
            simulation
                .step(substep_dt)
                .unwrap_or_else(|error| replay_panic(path, integrator, time, &error.to_string()));
        }
    }
    apply_due_commands(
        scenario,
        &mut simulation,
        &mut next_command,
        path,
        integrator,
    );

    let diagnostics = simulation.diagnostics();
    assert!(
        diagnostics.total_mechanical_energy.is_finite()
            && diagnostics.maximum_node_speed.is_finite()
            && diagnostics.maximum_absolute_strain.is_finite()
            && simulation
                .positions()
                .iter()
                .all(|position| position.is_finite()),
        "{} produced non-finite state with {}",
        path.display(),
        integrator.display_name()
    );
}

fn apply_due_commands(
    scenario: &RecordedScenario,
    simulation: &mut Simulation,
    next_command: &mut usize,
    path: &Path,
    integrator: ropesim_physics::IntegratorKind,
) {
    let time = simulation.diagnostics().simulation_time;
    while *next_command < scenario.commands.len()
        && scenario.commands[*next_command].time <= time + TIME_EPSILON
    {
        match scenario.commands[*next_command].command {
            MotionCommand::SetTarget(target) => simulation.set_payload_target(Some(target)),
            MotionCommand::InterpolateTarget { target, duration } => simulation
                .interpolate_payload_target(target, duration)
                .unwrap_or_else(|error| replay_panic(path, integrator, time, &error.to_string())),
            MotionCommand::Release { velocity } => simulation.release_payload(velocity),
        }
        *next_command += 1;
    }
}

fn replay_panic(
    path: &Path,
    integrator: ropesim_physics::IntegratorKind,
    time: f64,
    error: &str,
) -> ! {
    panic!(
        "{} failed with {} at t={time:.6}: {error}",
        path.display(),
        integrator.display_name()
    )
}

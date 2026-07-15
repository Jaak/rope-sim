use std::fs;
use std::path::{Path, PathBuf};

use ropesim_physics::{MotionCommand, RecordedScenario, Simulation};

const TIME_EPSILON: f64 = 1.0e-12;

#[test]
fn saved_scenarios_replay_without_solver_failures_or_non_finite_state() {
    let mut failures = Vec::new();
    for path in scenario_paths() {
        let json = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
        let scenario = RecordedScenario::from_json(&json)
            .unwrap_or_else(|error| panic!("invalid fixture {}: {error}", path.display()));

        for &integrator in &scenario.test_integrators {
            if let Err(error) = replay_scenario(&path, &scenario, integrator) {
                failures.push(error);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "saved scenario failures:\n{}",
        failures.join("\n")
    );
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
) -> Result<(), String> {
    let mut config = scenario.config;
    config.integrator = integrator;
    let mut simulation = Simulation::new(config).map_err(|error| {
        format!(
            "{} could not start with {}: {error}",
            path.display(),
            integrator.display_name()
        )
    })?;
    let mut next_command = 0;
    let mut maximum_node_speed = 0.0_f64;
    let mut maximum_speed_time = 0.0_f64;
    let mut maximum_speed_node = 0_usize;
    let settling_probe_time = (path.file_stem().and_then(|name| name.to_str()) == Some("bug-5"))
        .then(|| {
            scenario
                .commands
                .iter()
                .find_map(|timed| {
                    matches!(timed.command, MotionCommand::Release { .. })
                        .then_some(timed.time - 0.2)
                })
                .unwrap_or(scenario.duration)
        });
    let mut settling_probe_speed = None;
    let taut_probe_time =
        (path.file_stem().and_then(|name| name.to_str()) == Some("bug-7")).then(|| {
            scenario
                .commands
                .iter()
                .find_map(|timed| {
                    matches!(timed.command, MotionCommand::Release { .. }).then_some(timed.time)
                })
                .unwrap_or(scenario.duration)
        });
    let mut taut_probe = None;
    let smooth_slack_probe_time =
        (path.file_stem().and_then(|name| name.to_str()) == Some("bug-8-1")).then(|| {
            scenario
                .commands
                .iter()
                .find_map(|timed| {
                    matches!(timed.command, MotionCommand::Release { .. }).then_some(timed.time)
                })
                .unwrap_or(scenario.duration)
        });
    let mut smooth_slack_probe = None;
    // bug-4-1 records the slow-moving-boundary regression. Its final command
    // releases an 80 kg payload with roughly 11 m of slack below it; the later
    // free-fall/catch is a separate integrator stress case, not drag validation.
    let replay_duration = if path.file_stem().and_then(|name| name.to_str()) == Some("bug-4-1") {
        scenario
            .commands
            .iter()
            .find_map(|timed| {
                matches!(timed.command, MotionCommand::Release { .. }).then_some(timed.time)
            })
            .unwrap_or(scenario.duration)
    } else {
        scenario.duration
    };

    while simulation.diagnostics().simulation_time + TIME_EPSILON < replay_duration {
        apply_due_commands(scenario, &mut simulation, &mut next_command)?;

        let time = simulation.diagnostics().simulation_time;
        let base_dt = scenario.fixed_time_step.min(replay_duration - time);
        let substeps = simulation
            .recommended_substeps(base_dt)
            .map_err(|error| replay_error(path, integrator, time, &error.to_string()))?;
        let substep_dt = base_dt / substeps as f64;
        for _ in 0..substeps {
            simulation
                .step(substep_dt)
                .map_err(|error| replay_error(path, integrator, time, &error.to_string()))?;
            let (current_speed_node, current_speed) = simulation
                .velocities()
                .iter()
                .take(simulation.velocities().len().saturating_sub(1))
                .enumerate()
                .map(|(index, velocity)| (index, velocity.length()))
                .fold((0, 0.0_f64), |maximum, current| {
                    if current.1 > maximum.1 {
                        current
                    } else {
                        maximum
                    }
                });
            if current_speed > maximum_node_speed {
                maximum_node_speed = current_speed;
                maximum_speed_time = simulation.diagnostics().simulation_time;
                maximum_speed_node = current_speed_node;
            }
            if settling_probe_speed.is_none()
                && settling_probe_time.is_some_and(|probe| {
                    simulation.diagnostics().simulation_time + TIME_EPSILON >= probe
                })
            {
                settling_probe_speed = Some(current_speed);
            }
            if taut_probe.is_none()
                && taut_probe_time.is_some_and(|probe| {
                    simulation.diagnostics().simulation_time + TIME_EPSILON >= probe
                })
            {
                let shape = rope_shape(&simulation);
                taut_probe = Some(shape);
            }
            if smooth_slack_probe.is_none()
                && smooth_slack_probe_time.is_some_and(|probe| {
                    simulation.diagnostics().simulation_time + TIME_EPSILON >= probe
                })
            {
                let diagnostics = simulation.diagnostics();
                smooth_slack_probe = Some((
                    rope_shape(&simulation).2,
                    diagnostics.maximum_curvature,
                    diagnostics.minimum_segment_length,
                    diagnostics.maximum_node_speed,
                ));
            }
        }
    }
    apply_due_commands(scenario, &mut simulation, &mut next_command)?;

    let diagnostics = simulation.diagnostics();
    if !(diagnostics.total_mechanical_energy.is_finite()
        && diagnostics.maximum_node_speed.is_finite()
        && diagnostics.maximum_absolute_strain.is_finite()
        && simulation
            .positions()
            .iter()
            .all(|position| position.is_finite()))
    {
        return Err(format!(
            "{} produced non-finite state with {}",
            path.display(),
            integrator.display_name()
        ));
    }
    if path.file_stem().and_then(|name| name.to_str()) == Some("bug-3")
        && maximum_node_speed >= 25.0
    {
        return Err(format!(
            "{} reproduced its runaway slack-transition speed of {maximum_node_speed:.3} m/s at node {maximum_speed_node}, t={maximum_speed_time:.3} with {}",
            path.display(),
            integrator.display_name()
        ));
    }
    if path.file_stem().and_then(|name| name.to_str()) == Some("bug-4-1")
        && maximum_node_speed >= 10.0
    {
        return Err(format!(
            "{} reproduced its excessive boundary-driven speed of {maximum_node_speed:.3} m/s at node {maximum_speed_node}, t={maximum_speed_time:.3} with {}",
            path.display(),
            integrator.display_name()
        ));
    }
    if let Some(speed) = settling_probe_speed
        && speed >= 2.0
    {
        return Err(format!(
            "{} retained an excessive node speed of {speed:.3} m/s after the final stationary hold with {}",
            path.display(),
            integrator.display_name()
        ));
    }
    if let Some((geometric_slack, maximum_deviation, _)) = taut_probe
        && (geometric_slack >= 0.001 || maximum_deviation >= 0.02)
    {
        return Err(format!(
            "{} retained {:.3} m of geometric slack and {:.3} m maximum deviation while held taut with {}",
            path.display(),
            geometric_slack,
            maximum_deviation,
            integrator.display_name()
        ));
    }
    if let Some((contour_length, maximum_curvature, minimum_length, maximum_speed)) =
        smooth_slack_probe
        && ((contour_length - scenario.config.rope_length).abs() >= 0.02
            || maximum_curvature >= 40.0
            || minimum_length <= 0.95 * simulation.rest_length()
            || maximum_speed >= 3.0)
    {
        return Err(format!(
            "{} retained a kinked held state with {}: contour {:.3} m, curvature {:.1} 1/m, minimum element {:.4} m, speed {:.3} m/s",
            path.display(),
            integrator.display_name(),
            contour_length,
            maximum_curvature,
            minimum_length,
            maximum_speed,
        ));
    }
    Ok(())
}

fn rope_shape(simulation: &Simulation) -> (f64, f64, f64) {
    let positions = simulation.positions();
    let anchor = positions[0];
    let chord = positions[positions.len() - 1] - anchor;
    let chord_length = chord.length();
    let direction = if chord_length > f64::EPSILON {
        chord / chord_length
    } else {
        chord
    };
    let contour_length: f64 = positions
        .windows(2)
        .map(|nodes| (nodes[1] - nodes[0]).length())
        .sum();
    let maximum_deviation = positions
        .iter()
        .map(|position| {
            let relative = *position - anchor;
            (relative - direction * relative.dot(direction)).length()
        })
        .fold(0.0_f64, f64::max);
    (
        contour_length - chord_length,
        maximum_deviation,
        contour_length,
    )
}

fn apply_due_commands(
    scenario: &RecordedScenario,
    simulation: &mut Simulation,
    next_command: &mut usize,
) -> Result<(), String> {
    let time = simulation.diagnostics().simulation_time;
    while *next_command < scenario.commands.len()
        && scenario.commands[*next_command].time <= time + TIME_EPSILON
    {
        match scenario.commands[*next_command].command {
            MotionCommand::SetTarget(target) => {
                simulation.set_manipulation_target(target);
            }
            MotionCommand::InterpolateTarget { target, .. } => {
                simulation.set_manipulation_target(target);
            }
            MotionCommand::Release { velocity } => {
                simulation.release_manipulation(velocity);
            }
        }
        *next_command += 1;
    }
    Ok(())
}

fn replay_error(
    path: &Path,
    integrator: ropesim_physics::IntegratorKind,
    time: f64,
    error: &str,
) -> String {
    format!(
        "{} failed with {} at t={time:.6}: {error}",
        path.display(),
        integrator.display_name()
    )
}

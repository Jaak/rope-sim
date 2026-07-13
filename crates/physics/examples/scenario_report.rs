use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ropesim_physics::{Diagnostics, IntegratorKind, MotionCommand, RecordedScenario, Simulation};
use ropesim_physics::{KinematicTarget, RopeModelKind, SimulationConfig, Vec2};

const TIME_EPSILON: f64 = 1.0e-12;
const TIMING_RUNS: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut paths: Vec<_> = env::args_os().skip(1).map(PathBuf::from).collect();
    if paths.is_empty() {
        paths = default_fixture_paths()?;
    }

    for path in paths {
        report_fixture(&path)?;
        println!();
    }
    report_moving_boundary_convergence()?;
    Ok(())
}

fn report_fixture(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let scenario = RecordedScenario::from_json(&fs::read_to_string(path)?)?;

    println!("fixture: {}", path.display());
    println!(
        "{:<18} {:>7} {:>6} {:>9} {:>12} {:>10} {:>9} {:>9} {:>9} {:>8} {:>7} {:>7} {:>7} {:>7}",
        "integrator",
        "dt (ms)",
        "air",
        "wall ms",
        "final E",
        "speed max",
        "Newton",
        "R eval",
        "J build",
        "blk LU",
        "sp LU",
        "reject",
        "retry",
        "backtrk",
    );
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        for timestep_scale in [1.0, 0.5, 0.25] {
            report_run(
                &scenario,
                integrator,
                timestep_scale,
                scenario.config.air_damping_rate,
            )?;
        }
    }
    for air_damping_rate in [0.5, 1.0] {
        report_run(&scenario, IntegratorKind::TrBdf2, 1.0, air_damping_rate)?;
    }
    Ok(())
}

fn report_run(
    scenario: &RecordedScenario,
    integrator: IntegratorKind,
    timestep_scale: f64,
    air_damping_rate: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Keep allocation and code-page warmup outside the reported median.
    replay(scenario, integrator, timestep_scale, air_damping_rate)?;
    let mut runs = Vec::with_capacity(TIMING_RUNS);
    for _ in 0..TIMING_RUNS {
        runs.push(replay(
            scenario,
            integrator,
            timestep_scale,
            air_damping_rate,
        )?);
    }
    runs.sort_by_key(|metrics| metrics.elapsed);
    let metrics = runs.swap_remove(TIMING_RUNS / 2);
    println!(
        "{:<18} {:>7.3} {:>6.2} {:>9.3} {:>12.4e} {:>10.4} {:>9} {:>9} {:>9} {:>8} {:>7} {:>7} {:>7} {:>7}",
        integrator.display_name(),
        1_000.0 * scenario.fixed_time_step * timestep_scale,
        air_damping_rate,
        metrics.elapsed.as_secs_f64() * 1_000.0,
        metrics.final_diagnostics.total_mechanical_energy,
        metrics.maximum_node_speed,
        metrics.final_diagnostics.nonlinear_iterations,
        metrics.final_diagnostics.residual_evaluations,
        metrics.final_diagnostics.jacobian_assemblies,
        metrics.final_diagnostics.block_factorizations,
        metrics.final_diagnostics.sparse_factorizations,
        metrics.final_diagnostics.rejected_steps,
        metrics.final_diagnostics.adaptive_retries,
        metrics.final_diagnostics.line_search_backtracks,
    );
    Ok(())
}

fn default_fixture_paths() -> Result<Vec<PathBuf>, std::io::Error> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scenarios");
    let mut paths: Vec<_> = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "json")
        })
        .collect();
    paths.sort();
    Ok(paths)
}

struct ReplayMetrics {
    final_diagnostics: Diagnostics,
    maximum_node_speed: f64,
    elapsed: Duration,
}

fn replay(
    scenario: &RecordedScenario,
    integrator: IntegratorKind,
    timestep_scale: f64,
    air_damping_rate: f64,
) -> Result<ReplayMetrics, Box<dyn std::error::Error>> {
    let mut config = scenario.config;
    config.integrator = integrator;
    config.air_damping_rate = air_damping_rate;
    let mut simulation = Simulation::new(config)?;
    let base_timestep = scenario.fixed_time_step * timestep_scale;
    let mut next_command = 0;
    let mut maximum_node_speed = 0.0_f64;
    let start = Instant::now();

    while simulation.diagnostics().simulation_time + TIME_EPSILON < scenario.duration {
        apply_due_commands(scenario, &mut simulation, &mut next_command)?;
        let time = simulation.diagnostics().simulation_time;
        let outer_dt = base_timestep.min(scenario.duration - time);
        let substeps = simulation.recommended_substeps(outer_dt)?;
        let dt = outer_dt / substeps as f64;
        for _ in 0..substeps {
            simulation.step_without_diagnostics(dt)?;
        }
        let diagnostics = simulation.diagnostics();
        maximum_node_speed = maximum_node_speed.max(diagnostics.maximum_node_speed);
    }

    Ok(ReplayMetrics {
        final_diagnostics: simulation.diagnostics(),
        maximum_node_speed,
        elapsed: start.elapsed(),
    })
}

fn apply_due_commands(
    scenario: &RecordedScenario,
    simulation: &mut Simulation,
    next_command: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
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
            MotionCommand::Release { velocity } => simulation.release_manipulation(velocity),
        }
        *next_command += 1;
    }
    Ok(())
}

fn report_moving_boundary_convergence() -> Result<(), Box<dyn std::error::Error>> {
    println!("linear moving-boundary convergence (error against analytic solution):");
    let exact = exact_moving_boundary_state();
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        let error20 = state_error(moving_boundary_state(integrator, 0.02)?, exact);
        let error10 = state_error(moving_boundary_state(integrator, 0.01)?, exact);
        let error5 = state_error(moving_boundary_state(integrator, 0.005)?, exact);
        let error2_5 = state_error(moving_boundary_state(integrator, 0.0025)?, exact);
        println!(
            "  {:<18} e20={error20:.3e} e10={error10:.3e} e5={error5:.3e} e2.5={error2_5:.3e} ratios={:.2},{:.2},{:.2}",
            integrator.display_name(),
            error20 / error10,
            error10 / error5,
            error5 / error2_5,
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct NodeState {
    position: Vec2,
    velocity: Vec2,
}

fn moving_boundary_state(
    integrator: IntegratorKind,
    dt: f64,
) -> Result<NodeState, Box<dyn std::error::Error>> {
    let config = SimulationConfig {
        segment_count: 2,
        rope_length: 2.0,
        rope_mass: 1.0,
        payload_mass: 1.0,
        axial_rigidity: 1_000.0,
        rope_model: RopeModelKind::HookeSpring,
        gravity: Vec2::ZERO,
        air_damping_rate: 0.0,
        integrator,
        ..SimulationConfig::default()
    };
    let mut simulation = Simulation::new(config)?;
    let duration = 0.5;
    simulation.interpolate_payload_target(
        KinematicTarget::new(Vec2::new(0.0, -2.2), Vec2::new(0.0, -0.4)),
        duration,
    )?;
    let steps = (duration / dt).round() as usize;
    for _ in 0..steps {
        simulation.step(dt)?;
    }
    Ok(NodeState {
        position: simulation.positions()[1],
        velocity: simulation.velocities()[1],
    })
}

fn exact_moving_boundary_state() -> NodeState {
    let duration = 0.5;
    let boundary_speed = 0.4;
    let frequency = 4_000.0_f64.sqrt();
    let position = 1.0 + 0.5 * boundary_speed * duration
        - 0.5 * boundary_speed / frequency * (frequency * duration).sin();
    let velocity = 0.5 * boundary_speed - 0.5 * boundary_speed * (frequency * duration).cos();
    NodeState {
        position: Vec2::new(0.0, -position),
        velocity: Vec2::new(0.0, -velocity),
    }
}

fn state_error(actual: NodeState, expected: NodeState) -> f64 {
    ((actual.position - expected.position).length_squared()
        + (actual.velocity - expected.velocity).length_squared())
    .sqrt()
}

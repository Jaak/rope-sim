use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use ropesim_physics::{
    IntegratorKind, KinematicTarget, RopeModelKind, Simulation, SimulationConfig, Vec2,
};

const DT: f64 = 1.0 / 240.0;
const WARMUP_STEPS: usize = 60;
const TIMED_STEPS: usize = 180;
const PROFILE_STEPS: usize = 2_400;
const SEGMENT_COUNTS: [usize; 4] = [64, 256, 512, 1_024];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Some(argument) = env::args().nth(1) {
        match argument.as_str() {
            "--profile-hybrid" => return profile_hybrid(false),
            "--profile-bending" => return profile_hybrid(true),
            _ => return Err(format!("unrecognized argument: {argument}").into()),
        }
    }

    println!("Run this benchmark with --release. Times are per 240 Hz physics step.");
    println!(
        "{:<11} {:<10} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9} {:>11} {:>11}",
        "phase",
        "model",
        "links",
        "mean us",
        "p50 us",
        "p99 us",
        "XPBD us",
        "corr us",
        "substeps",
        "correct",
        "fallback",
        "last R",
        "release us",
    );

    for rope_model in [
        RopeModelKind::StandardLinearSolid,
        RopeModelKind::QuadraticKelvinVoigt,
    ] {
        for segment_count in SEGMENT_COUNTS {
            let result = benchmark_hybrid_drag(segment_count, rope_model, 0.0)?;
            print_result("hybrid", rope_model, segment_count, &result);
        }
    }

    for segment_count in SEGMENT_COUNTS {
        let result =
            benchmark_hybrid_drag(segment_count, RopeModelKind::StandardLinearSolid, 0.01)?;
        print_result(
            "bend hybrid",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }

    for segment_count in SEGMENT_COUNTS {
        let result = benchmark_free(segment_count, 0.0, IntegratorKind::BackwardEuler)?;
        print_result(
            "free BE",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }
    for segment_count in SEGMENT_COUNTS {
        let result = benchmark_free(segment_count, 0.01, IntegratorKind::BackwardEuler)?;
        print_result(
            "bend BE",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }
    for segment_count in SEGMENT_COUNTS {
        let result = benchmark_free(segment_count, 0.0, IntegratorKind::TrBdf2)?;
        print_result(
            "free TR2",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }
    for segment_count in SEGMENT_COUNTS {
        let result = benchmark_free(segment_count, 0.01, IntegratorKind::TrBdf2)?;
        print_result(
            "bend TR2",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }
    for segment_count in [20, 40, 64] {
        let result = benchmark_free(segment_count, 0.01, IntegratorKind::RungeKutta4)?;
        print_result(
            "bend RK4",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }
    Ok(())
}

fn profile_hybrid(bending: bool) -> Result<(), Box<dyn std::error::Error>> {
    let bending_rigidity = if bending { 0.01 } else { 0.0 };
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 1_024,
        rope_model: RopeModelKind::StandardLinearSolid,
        bending_rigidity,
        bending_viscosity: 0.1 * bending_rigidity,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })?;

    for step in 0..WARMUP_STEPS + PROFILE_STEPS {
        simulation.set_manipulation_target(drag_target(step));
        simulation.step_without_diagnostics(black_box(DT))?;
        black_box(simulation.payload_position());
    }

    println!(
        "profiled {PROFILE_STEPS} hybrid steps at 1,024 links ({})",
        if bending {
            "with bending"
        } else {
            "axial only"
        },
    );
    Ok(())
}

struct BenchmarkResult {
    samples: Vec<Duration>,
    xpbd_only_samples: Vec<Duration>,
    correction_samples: Vec<Duration>,
    corrections: u64,
    fallbacks: u64,
    release: Option<Duration>,
    last_fallback_residual: f64,
    maximum_substeps: usize,
}

fn benchmark_hybrid_drag(
    segment_count: usize,
    rope_model: RopeModelKind,
    bending_rigidity: f64,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count,
        rope_model,
        bending_rigidity,
        bending_viscosity: 0.1 * bending_rigidity,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })?;

    for step in 0..WARMUP_STEPS {
        simulation.set_manipulation_target(drag_target(step));
        simulation.step_without_diagnostics(DT)?;
    }
    let before = simulation.diagnostics();
    let mut samples = Vec::with_capacity(TIMED_STEPS);
    let mut xpbd_only_samples = Vec::with_capacity(TIMED_STEPS / 2);
    let mut correction_samples = Vec::with_capacity(TIMED_STEPS / 2);
    for step in WARMUP_STEPS..WARMUP_STEPS + TIMED_STEPS {
        simulation.set_manipulation_target(drag_target(step));
        let start = Instant::now();
        simulation.step_without_diagnostics(black_box(DT))?;
        let elapsed = start.elapsed();
        samples.push(elapsed);
        if (step + 1).is_multiple_of(2) {
            correction_samples.push(elapsed);
        } else {
            xpbd_only_samples.push(elapsed);
        }
    }
    let after = simulation.diagnostics();
    assert_eq!(
        after.manipulation_corrections - before.manipulation_corrections
            + after.manipulation_correction_fallbacks
            - before.manipulation_correction_fallbacks,
        correction_samples.len() as u64,
        "the benchmark's correction-step classification is out of date",
    );
    let release_start = Instant::now();
    simulation.release_manipulation(drag_target(WARMUP_STEPS + TIMED_STEPS).velocity);
    let release = release_start.elapsed();

    Ok(BenchmarkResult {
        samples,
        xpbd_only_samples,
        correction_samples,
        corrections: after.manipulation_corrections - before.manipulation_corrections,
        fallbacks: after.manipulation_correction_fallbacks
            - before.manipulation_correction_fallbacks,
        release: Some(release),
        last_fallback_residual: after.manipulation_last_fallback_residual,
        maximum_substeps: 1,
    })
}

fn benchmark_free(
    segment_count: usize,
    bending_rigidity: f64,
    integrator: IntegratorKind,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count,
        rope_model: RopeModelKind::StandardLinearSolid,
        bending_rigidity,
        bending_viscosity: 0.1 * bending_rigidity,
        integrator,
        ..SimulationConfig::default()
    })?;
    let mut maximum_substeps = 1;
    for _ in 0..WARMUP_STEPS {
        advance_outer_step(&mut simulation, &mut maximum_substeps)?;
    }

    let mut samples = Vec::with_capacity(TIMED_STEPS);
    for _ in 0..TIMED_STEPS {
        let start = Instant::now();
        advance_outer_step(&mut simulation, &mut maximum_substeps)?;
        samples.push(start.elapsed());
    }
    Ok(BenchmarkResult {
        samples,
        xpbd_only_samples: Vec::new(),
        correction_samples: Vec::new(),
        corrections: 0,
        fallbacks: 0,
        release: None,
        last_fallback_residual: 0.0,
        maximum_substeps,
    })
}

fn advance_outer_step(
    simulation: &mut Simulation,
    maximum_substeps: &mut usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let substeps = simulation.recommended_substeps(black_box(DT))?;
    *maximum_substeps = (*maximum_substeps).max(substeps);
    let dt = DT / substeps as f64;
    for _ in 0..substeps {
        simulation.step_without_diagnostics(dt)?;
    }
    Ok(())
}

fn drag_target(step: usize) -> KinematicTarget {
    let time = step as f64 * DT;
    let angular_speed = std::f64::consts::PI;
    let phase = angular_speed * time;
    KinematicTarget::new(
        Vec2::new(1.5 * phase.sin(), -11.4 + 0.2 * phase.cos()),
        Vec2::new(
            1.5 * angular_speed * phase.cos(),
            -0.2 * angular_speed * phase.sin(),
        ),
    )
}

fn print_result(
    phase: &str,
    rope_model: RopeModelKind,
    segment_count: usize,
    result: &BenchmarkResult,
) {
    let mut micros: Vec<_> = result
        .samples
        .iter()
        .map(|duration| duration.as_secs_f64() * 1.0e6)
        .collect();
    micros.sort_by(f64::total_cmp);
    let mean = micros.iter().sum::<f64>() / micros.len() as f64;
    let percentile = |fraction: f64| {
        let index = (fraction * (micros.len() - 1) as f64).ceil() as usize;
        micros[index]
    };
    let release = result
        .release
        .map_or(0.0, |duration| duration.as_secs_f64() * 1.0e6);
    let mean_duration = |samples: &[Duration]| {
        if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(Duration::as_secs_f64).sum::<f64>() * 1.0e6 / samples.len() as f64
        }
    };
    println!(
        "{phase:<11} {:<10} {segment_count:>6} {mean:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>10.1} {:>8} {:>9} {:>9} {:>11.2e} {release:>11.1}",
        short_model_name(rope_model),
        percentile(0.50),
        percentile(0.99),
        mean_duration(&result.xpbd_only_samples),
        mean_duration(&result.correction_samples),
        result.maximum_substeps,
        result.corrections,
        result.fallbacks,
        result.last_fallback_residual,
    );
}

fn short_model_name(model: RopeModelKind) -> &'static str {
    match model {
        RopeModelKind::HookeSpring => "Hooke",
        RopeModelKind::KelvinVoigt => "KV",
        RopeModelKind::QuadraticKelvinVoigt => "QKV",
        RopeModelKind::StandardLinearSolid => "SLS",
    }
}

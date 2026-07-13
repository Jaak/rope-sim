use std::hint::black_box;
use std::time::{Duration, Instant};

use ropesim_physics::{
    IntegratorKind, KinematicTarget, RopeModelKind, Simulation, SimulationConfig, Vec2,
};

const DT: f64 = 1.0 / 240.0;
const WARMUP_STEPS: usize = 60;
const TIMED_STEPS: usize = 180;
const SEGMENT_COUNTS: [usize; 4] = [64, 256, 512, 1_024];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Run this benchmark with --release. Times are per 240 Hz physics step.");
    println!(
        "{:<11} {:<10} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9} {:>11} {:>11}",
        "phase",
        "model",
        "links",
        "mean us",
        "p50 us",
        "p99 us",
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
            let result = benchmark_hybrid_drag(segment_count, rope_model)?;
            print_result("hybrid", rope_model, segment_count, &result);
        }
    }

    for segment_count in SEGMENT_COUNTS {
        let result = benchmark_free_backward_euler(segment_count)?;
        print_result(
            "free BE",
            RopeModelKind::StandardLinearSolid,
            segment_count,
            &result,
        );
    }
    Ok(())
}

struct BenchmarkResult {
    samples: Vec<Duration>,
    corrections: u64,
    fallbacks: u64,
    release: Option<Duration>,
    last_fallback_residual: f64,
}

fn benchmark_hybrid_drag(
    segment_count: usize,
    rope_model: RopeModelKind,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count,
        rope_model,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })?;

    for step in 0..WARMUP_STEPS {
        simulation.set_manipulation_target(drag_target(step));
        simulation.step_without_diagnostics(DT)?;
    }
    let before = simulation.diagnostics();
    let mut samples = Vec::with_capacity(TIMED_STEPS);
    for step in WARMUP_STEPS..WARMUP_STEPS + TIMED_STEPS {
        simulation.set_manipulation_target(drag_target(step));
        let start = Instant::now();
        simulation.step_without_diagnostics(black_box(DT))?;
        samples.push(start.elapsed());
    }
    let after = simulation.diagnostics();
    let release_start = Instant::now();
    simulation.release_manipulation(drag_target(WARMUP_STEPS + TIMED_STEPS).velocity);
    let release = release_start.elapsed();

    Ok(BenchmarkResult {
        samples,
        corrections: after.manipulation_corrections - before.manipulation_corrections,
        fallbacks: after.manipulation_correction_fallbacks
            - before.manipulation_correction_fallbacks,
        release: Some(release),
        last_fallback_residual: after.manipulation_last_fallback_residual,
    })
}

fn benchmark_free_backward_euler(
    segment_count: usize,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count,
        rope_model: RopeModelKind::StandardLinearSolid,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })?;
    for _ in 0..WARMUP_STEPS {
        simulation.step_without_diagnostics(DT)?;
    }

    let mut samples = Vec::with_capacity(TIMED_STEPS);
    for _ in 0..TIMED_STEPS {
        let start = Instant::now();
        simulation.step_without_diagnostics(black_box(DT))?;
        samples.push(start.elapsed());
    }
    Ok(BenchmarkResult {
        samples,
        corrections: 0,
        fallbacks: 0,
        release: None,
        last_fallback_residual: 0.0,
    })
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
    println!(
        "{phase:<11} {:<10} {segment_count:>6} {mean:>10.1} {:>10.1} {:>10.1} {:>9} {:>9} {:>11.2e} {release:>11.1}",
        short_model_name(rope_model),
        percentile(0.50),
        percentile(0.99),
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

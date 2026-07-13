use ropesim_physics::{
    CalibrationMeasurements, CalibrationSettings, DynamicRopeReference, RopeModelKind,
    SimulationConfig, VOLTA_GUIDE_9MM, run_dynamic_rope_calibration,
};

const SEGMENT_COUNT: usize = 64;

fn main() {
    let reference = VOLTA_GUIDE_9MM;
    let relaxed_rigidity = relaxed_linear_rigidity(reference);
    println!("Fitting {} at {} elements", reference.name, SEGMENT_COUNT);
    println!("static-constrained relaxed EA = {relaxed_rigidity:.3} N\n");

    let (hooke_rigidity, _) = fit_one_dimension(30_000.0_f64.ln(), |log_rigidity| {
        evaluate(SimulationConfig {
            rope_model: RopeModelKind::HookeSpring,
            axial_rigidity: log_rigidity.exp(),
            ..SimulationConfig::default()
        })
    });
    let hooke_config = SimulationConfig {
        rope_model: RopeModelKind::HookeSpring,
        axial_rigidity: hooke_rigidity.exp(),
        ..SimulationConfig::default()
    };

    let (kv_viscosity, _) = fit_one_dimension(1_000.0_f64.ln(), |log_viscosity| {
        evaluate(SimulationConfig {
            rope_model: RopeModelKind::KelvinVoigt,
            axial_rigidity: relaxed_rigidity,
            axial_viscosity: log_viscosity.exp(),
            ..SimulationConfig::default()
        })
    });
    let kv_config = SimulationConfig {
        rope_model: RopeModelKind::KelvinVoigt,
        axial_rigidity: relaxed_rigidity,
        axial_viscosity: kv_viscosity.exp(),
        ..SimulationConfig::default()
    };

    let ([sls_transient, sls_viscosity], _) = fit_two_dimensions(
        [22_000.0_f64.ln(), 4_400.0_f64.ln()],
        |[log_transient, log_viscosity]| {
            evaluate(SimulationConfig {
                rope_model: RopeModelKind::StandardLinearSolid,
                axial_rigidity: relaxed_rigidity,
                transient_axial_rigidity: log_transient.exp(),
                axial_viscosity: log_viscosity.exp(),
                ..SimulationConfig::default()
            })
        },
    );
    let sls_config = SimulationConfig {
        rope_model: RopeModelKind::StandardLinearSolid,
        axial_rigidity: relaxed_rigidity,
        transient_axial_rigidity: sls_transient.exp(),
        axial_viscosity: sls_viscosity.exp(),
        ..SimulationConfig::default()
    };

    let qkv_seed = coarse_qkv_seed(reference);
    let ([quadratic_rigidity, qkv_viscosity], _) =
        fit_two_dimensions(qkv_seed, |[log_quadratic, log_viscosity]| {
            let quadratic = log_quadratic.exp();
            let Some(linear) = qkv_linear_rigidity(reference, quadratic) else {
                return Evaluation::failed();
            };
            evaluate(SimulationConfig {
                rope_model: RopeModelKind::QuadraticKelvinVoigt,
                axial_rigidity: linear,
                quadratic_axial_rigidity: quadratic,
                axial_viscosity: log_viscosity.exp(),
                ..SimulationConfig::default()
            })
        });
    let quadratic_rigidity = quadratic_rigidity.exp();
    let qkv_config = SimulationConfig {
        rope_model: RopeModelKind::QuadraticKelvinVoigt,
        axial_rigidity: qkv_linear_rigidity(reference, quadratic_rigidity)
            .expect("fitted QKV rigidity has a static solution"),
        quadratic_axial_rigidity: quadratic_rigidity,
        axial_viscosity: qkv_viscosity.exp(),
        ..SimulationConfig::default()
    };

    println!("Selected parameter sets:");
    print_config("Hooke", hooke_config);
    print_config("Kelvin-Voigt", kv_config);
    print_config("SLS", sls_config);
    print_config("Quadratic Kelvin-Voigt", qkv_config);
    println!("\nKelvin-Voigt tradeoff at the static-constrained EA:");
    println!("{:>12} {:>9} {:>11}", "eta*A", "dynamic", "impact");
    for viscosity in [0.0, 250.0, 500.0, 1_000.0, 1_500.0, 2_000.0, 3_000.0] {
        let config = SimulationConfig {
            rope_model: RopeModelKind::KelvinVoigt,
            axial_rigidity: relaxed_rigidity,
            axial_viscosity: viscosity,
            ..SimulationConfig::default()
        };
        let settings = CalibrationSettings {
            segment_count: SEGMENT_COUNT,
            timestep: 1.0 / 240.0,
            static_duration: 1.0e-6,
            dynamic_duration: 1.0,
        };
        match run_dynamic_rope_calibration(config, reference, settings) {
            Ok(result) => println!(
                "{viscosity:>12.1} {:>8.2}% {:>9.3} kN",
                100.0 * result.maximum_dynamic_elongation,
                result.peak_payload_tension / 1_000.0,
            ),
            Err(error) => println!("{viscosity:>12.1} failed: {error}"),
        }
    }
    println!("\nFull fixture verification:");
    println!(
        "{:<26} {:>9} {:>9} {:>11} {:>10}",
        "model", "static", "dynamic", "impact", "score"
    );
    for (name, config) in [
        ("Hooke", hooke_config),
        ("Kelvin-Voigt", kv_config),
        ("SLS", sls_config),
        ("Quadratic Kelvin-Voigt", qkv_config),
    ] {
        match run_dynamic_rope_calibration(config, reference, CalibrationSettings::default()) {
            Ok(result) => println!(
                "{name:<26} {:>8.2}% {:>8.2}% {:>9.3} kN {:>10.3}",
                100.0 * result.static_elongation,
                100.0 * result.maximum_dynamic_elongation,
                result.peak_payload_tension / 1_000.0,
                score(result, reference),
            ),
            Err(error) => println!("{name:<26} failed: {error}"),
        }
    }
    println!("\nSLS discretization check:");
    println!(
        "{:>8} {:>9} {:>9} {:>11}",
        "pieces", "dt (ms)", "dynamic", "impact"
    );
    for segment_count in [20, 32, 64, 128] {
        for frequency in [120.0, 240.0, 480.0] {
            let settings = CalibrationSettings {
                segment_count,
                timestep: 1.0 / frequency,
                static_duration: 1.0e-6,
                dynamic_duration: 1.0,
            };
            match run_dynamic_rope_calibration(sls_config, reference, settings) {
                Ok(result) => println!(
                    "{segment_count:>8} {:>9.3} {:>8.2}% {:>9.3} kN",
                    1_000.0 / frequency,
                    100.0 * result.maximum_dynamic_elongation,
                    result.peak_payload_tension / 1_000.0,
                ),
                Err(error) => println!(
                    "{segment_count:>8} {:>9.3} failed: {error}",
                    1_000.0 / frequency
                ),
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Evaluation {
    score: f64,
}

impl Evaluation {
    const fn failed() -> Self {
        Self { score: 1.0e100 }
    }
}

fn evaluate(config: SimulationConfig) -> Evaluation {
    let settings = CalibrationSettings {
        segment_count: SEGMENT_COUNT,
        timestep: 1.0 / 240.0,
        static_duration: 1.0e-6,
        dynamic_duration: 1.0,
    };
    run_dynamic_rope_calibration(config, VOLTA_GUIDE_9MM, settings).map_or_else(
        |_| Evaluation::failed(),
        |measurements| Evaluation {
            score: score(measurements, VOLTA_GUIDE_9MM),
        },
    )
}

fn score(measurements: CalibrationMeasurements, reference: DynamicRopeReference) -> f64 {
    let elongation_error =
        (measurements.maximum_dynamic_elongation - reference.maximum_dynamic_elongation) / 0.01;
    let force_error = (measurements.peak_payload_tension - reference.maximum_impact_force) / 250.0;
    elongation_error * elongation_error + force_error * force_error
}

fn fit_one_dimension(
    mut center: f64,
    mut objective: impl FnMut(f64) -> Evaluation,
) -> (f64, Evaluation) {
    let mut step = 2.0_f64.ln();
    let mut best = objective(center);
    for _ in 0..24 {
        let candidates = [center - step, center, center + step];
        let mut next_center = center;
        let mut next_best = best;
        for candidate in candidates {
            let evaluation = objective(candidate);
            if evaluation.score < next_best.score {
                next_center = candidate;
                next_best = evaluation;
            }
        }
        if next_center == center {
            step *= 0.5;
        } else {
            center = next_center;
            best = next_best;
        }
    }
    (center, best)
}

fn fit_two_dimensions(
    mut center: [f64; 2],
    mut objective: impl FnMut([f64; 2]) -> Evaluation,
) -> ([f64; 2], Evaluation) {
    let mut step = [2.0_f64.ln(), 2.0_f64.ln()];
    let mut best = objective(center);
    for _ in 0..28 {
        let mut next_center = center;
        let mut next_best = best;
        for first in -1..=1 {
            for second in -1..=1 {
                let candidate = [
                    center[0] + first as f64 * step[0],
                    center[1] + second as f64 * step[1],
                ];
                let evaluation = objective(candidate);
                if evaluation.score < next_best.score {
                    next_center = candidate;
                    next_best = evaluation;
                }
            }
        }
        if next_center == center {
            step[0] *= 0.5;
            step[1] *= 0.5;
        } else {
            center = next_center;
            best = next_best;
        }
    }
    (center, best)
}

fn relaxed_linear_rigidity(reference: DynamicRopeReference) -> f64 {
    let average_supported_mass = reference.static_test_mass + 0.5 * reference.drop_test_rope_mass();
    9.81 * average_supported_mass / reference.static_elongation
}

fn qkv_linear_rigidity(reference: DynamicRopeReference, quadratic_rigidity: f64) -> Option<f64> {
    let mean_strain = |linear_rigidity: f64| {
        (0..SEGMENT_COUNT)
            .map(|left| {
                let supported_rope_fraction = (SEGMENT_COUNT - left) as f64 / SEGMENT_COUNT as f64
                    - 0.5 / SEGMENT_COUNT as f64;
                let tension = 9.81
                    * (reference.static_test_mass
                        + reference.drop_test_rope_mass() * supported_rope_fraction);
                2.0 * tension
                    / (linear_rigidity
                        + (linear_rigidity * linear_rigidity + 4.0 * quadratic_rigidity * tension)
                            .sqrt())
            })
            .sum::<f64>()
            / SEGMENT_COUNT as f64
    };

    if mean_strain(0.0) < reference.static_elongation {
        return None;
    }
    let mut lower = 0.0;
    let mut upper = 100_000.0;
    for _ in 0..80 {
        let middle = 0.5 * (lower + upper);
        if mean_strain(middle) > reference.static_elongation {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    Some(0.5 * (lower + upper))
}

fn coarse_qkv_seed(reference: DynamicRopeReference) -> [f64; 2] {
    let mut best_parameters = [40_000.0_f64.ln(), 0.1_f64.ln()];
    let mut best_score = f64::INFINITY;
    for quadratic in [
        5_000.0, 10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0, 60_000.0, 80_000.0, 100_000.0,
        120_000.0,
    ] {
        let Some(linear) = qkv_linear_rigidity(reference, quadratic) else {
            continue;
        };
        let mut row_best_score = f64::INFINITY;
        let mut row_best_viscosity = 0.0;
        for viscosity in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0] {
            let evaluation = evaluate(SimulationConfig {
                rope_model: RopeModelKind::QuadraticKelvinVoigt,
                axial_rigidity: linear,
                quadratic_axial_rigidity: quadratic,
                axial_viscosity: viscosity,
                ..SimulationConfig::default()
            });
            if evaluation.score < best_score {
                best_score = evaluation.score;
                best_parameters = [quadratic.ln(), viscosity.ln()];
            }
            if evaluation.score < row_best_score {
                row_best_score = evaluation.score;
                row_best_viscosity = viscosity;
            }
        }
        println!(
            "  QKV grid A2={quadratic:>8.1}: eta={row_best_viscosity:>8.4}, score={row_best_score:.3}"
        );
    }
    println!(
        "QKV coarse seed A2={:.1} N, eta*A={:.4} N*s, score={best_score:.3}",
        best_parameters[0].exp(),
        best_parameters[1].exp(),
    );
    best_parameters
}

fn print_config(name: &str, config: SimulationConfig) {
    match config.rope_model {
        RopeModelKind::KelvinVoigt => println!(
            "  {name:<24} EA={:.1} N, eta*A={:.1} N*s",
            config.axial_rigidity, config.axial_viscosity
        ),
        RopeModelKind::StandardLinearSolid => println!(
            "  {name:<24} EA_inf={:.1} N, EA_1={:.1} N, eta*A={:.1} N*s, tau={:.4} s",
            config.axial_rigidity,
            config.transient_axial_rigidity,
            config.axial_viscosity,
            config.axial_viscosity / config.transient_axial_rigidity,
        ),
        RopeModelKind::QuadraticKelvinVoigt => println!(
            "  {name:<24} A1={:.1} N, A2={:.1} N, eta*A={:.1} N*s",
            config.axial_rigidity, config.quadratic_axial_rigidity, config.axial_viscosity
        ),
        RopeModelKind::HookeSpring => println!(
            "  {name:<24} effective dynamic EA={:.1} N",
            config.axial_rigidity
        ),
    }
}

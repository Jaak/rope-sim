use ropesim_physics::{
    CalibrationSettings, KinematicTarget, Simulation, SimulationConfig, VOLTA_GUIDE_9MM, Vec2,
    run_dynamic_rope_calibration,
};

const DT: f64 = 1.0 / 240.0;
const RAMP_STEPS: usize = 480;
const HOLD_STEPS: usize = 480;
const RIGIDITIES: [f64; 5] = [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Experimental bending sensitivity probe");
    println!("Bending viscosity is B * 0.1 s. B=0 exactly selects axial-only behavior.\n");
    println!(
        "{:<8} {:>6} {:>12} {:>12} {:>12} {:>12} {:>10}",
        "B N*m^2", "links", "curv 1/m", "bend J", "end axial N", "max strain", "fallback"
    );

    for segment_count in [20, 64] {
        for rigidity in RIGIDITIES {
            let result = end_shortening_probe(segment_count, rigidity)?;
            println!(
                "{rigidity:<8.1e} {segment_count:>6} {:>12.4} {:>12.4} {:>12.2} {:>11.3}% {:>10}",
                result.maximum_curvature,
                result.bending_energy,
                result.end_axial_force,
                100.0 * result.maximum_tensile_strain,
                result.fallbacks,
            );
        }
    }

    println!("\nIdeal vertical first-arrest fixture (32 links):");
    println!(
        "{:<8} {:>11} {:>11} {:>12}",
        "B N*m^2", "static", "dynamic", "impact kN"
    );
    let settings = CalibrationSettings {
        segment_count: 32,
        ..CalibrationSettings::default()
    };
    for rigidity in RIGIDITIES {
        let config = SimulationConfig {
            bending_rigidity: rigidity,
            bending_viscosity: 0.1 * rigidity,
            ..SimulationConfig::default()
        };
        match run_dynamic_rope_calibration(config, VOLTA_GUIDE_9MM, settings) {
            Ok(result) => println!(
                "{rigidity:<8.1e} {:>10.3}% {:>10.3}% {:>12.3}",
                100.0 * result.static_elongation,
                100.0 * result.maximum_dynamic_elongation,
                result.peak_payload_tension / 1_000.0,
            ),
            Err(error) => println!("{rigidity:<8.1e} failed: {error}"),
        }
    }
    Ok(())
}

struct EndShorteningResult {
    maximum_curvature: f64,
    bending_energy: f64,
    end_axial_force: f64,
    maximum_tensile_strain: f64,
    fallbacks: u64,
}

fn end_shortening_probe(
    segment_count: usize,
    bending_rigidity: f64,
) -> Result<EndShorteningResult, Box<dyn std::error::Error>> {
    let config = SimulationConfig {
        segment_count,
        bending_rigidity,
        bending_viscosity: 0.1 * bending_rigidity,
        ..SimulationConfig::default()
    };
    let mut simulation = Simulation::new(config)?;
    let start = simulation.payload_position();
    // The lateral offset breaks the perfectly straight, numerically symmetric
    // compressed branch without prescribing where the rope should fold.
    let end = Vec2::new(0.5, -6.0);
    let ramp_velocity = (end - start) / (RAMP_STEPS as f64 * DT);

    for step in 1..=RAMP_STEPS {
        let fraction = step as f64 / RAMP_STEPS as f64;
        simulation.set_manipulation_target(KinematicTarget::new(
            start + (end - start) * fraction,
            ramp_velocity,
        ));
        simulation.step_without_diagnostics(DT)?;
    }
    for _ in 0..HOLD_STEPS {
        simulation.set_manipulation_target(KinematicTarget::new(end, Vec2::ZERO));
        simulation.step_without_diagnostics(DT)?;
    }

    let diagnostics = simulation.diagnostics();
    Ok(EndShorteningResult {
        maximum_curvature: diagnostics.maximum_curvature,
        bending_energy: diagnostics.bending_energy,
        end_axial_force: simulation.segment_tension(segment_count - 1).unwrap_or(0.0),
        maximum_tensile_strain: diagnostics.maximum_tensile_strain,
        fallbacks: diagnostics.manipulation_correction_fallbacks,
    })
}

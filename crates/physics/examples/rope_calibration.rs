use ropesim_physics::{
    CalibrationSettings, RopeModelKind, SimulationConfig, VOLTA_GUIDE_9MM,
    run_dynamic_rope_calibration,
};

fn main() {
    let reference = VOLTA_GUIDE_9MM;
    let settings = CalibrationSettings::default();
    println!("Reference: {}", reference.name);
    println!(
        "Fixture: {:.1} kg, {:.2} m rope, {:.2} m free fall, factor {:.3}, {} elements, dt {:.3} ms",
        reference.drop_test_mass,
        reference.drop_test_rope_length,
        reference.free_fall_height,
        reference.fall_factor(),
        settings.segment_count,
        1_000.0 * settings.timestep,
    );
    println!(
        "Targets: static {:.1}%, dynamic {:.1}%, impact {:.2} kN\n",
        100.0 * reference.static_elongation,
        100.0 * reference.maximum_dynamic_elongation,
        reference.maximum_impact_force / 1_000.0,
    );
    println!(
        "{:<34} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "model", "static", "dynamic", "payload kN", "anchor kN", "end m/s"
    );

    for rope_model in RopeModelKind::ALL {
        let config = SimulationConfig::default().with_recommended_rope_model(rope_model);
        match run_dynamic_rope_calibration(config, reference, settings) {
            Ok(result) => println!(
                "{:<34} {:>9.2}% {:>9.2}% {:>12.3} {:>12.3} {:>10.4}",
                rope_model.display_name(),
                100.0 * result.static_elongation,
                100.0 * result.maximum_dynamic_elongation,
                result.peak_payload_tension / 1_000.0,
                result.peak_anchor_tension / 1_000.0,
                result.static_endpoint_speed,
            ),
            Err(error) => println!("{:<34} failed: {error}", rope_model.display_name()),
        }
    }
}

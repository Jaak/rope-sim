use ropesim_physics::{
    ConfigError, IntegratorKind, KinematicTarget, ReconfigureOutcome, RopeModelKind, Simulation,
    SimulationConfig, Vec2,
};

const EPSILON: f64 = 1.0e-10;

#[test]
fn mass_distribution_preserves_total_mass() {
    let config = SimulationConfig::default();
    let simulation = Simulation::new(config).unwrap();
    let total_mass: f64 = simulation.masses().iter().sum();

    assert!((total_mass - config.rope_mass - config.payload_mass).abs() < EPSILON);
}

#[test]
fn initial_segments_have_the_rest_length() {
    let simulation = Simulation::new(SimulationConfig::default()).unwrap();

    for nodes in simulation.positions().windows(2) {
        assert!(((nodes[1] - nodes[0]).length() - simulation.rest_length()).abs() < EPSILON);
    }
}

#[test]
fn fixed_anchor_never_moves() {
    let config = SimulationConfig::default();
    let mut simulation = Simulation::new(config).unwrap();

    for _ in 0..20 {
        simulation.step(1.0 / 1000.0).unwrap();
    }

    assert_eq!(simulation.positions()[0], config.anchor);
    assert_eq!(simulation.velocities()[0], Vec2::ZERO);
}

#[test]
fn stepping_without_diagnostics_preserves_the_simulation_result() {
    let config = SimulationConfig {
        integrator: IntegratorKind::BackwardEuler,
        rope_model: RopeModelKind::StandardLinearSolid,
        ..SimulationConfig::default()
    };
    let mut regular = Simulation::new(config).unwrap();
    let mut deferred = Simulation::new(config).unwrap();
    let target = KinematicTarget::new(Vec2::new(0.4, -11.8), Vec2::new(1.5, 0.3));
    regular.interpolate_payload_target(target, 0.1).unwrap();
    deferred.interpolate_payload_target(target, 0.1).unwrap();

    for _ in 0..12 {
        regular.step(1.0 / 240.0).unwrap();
        deferred.step_without_diagnostics(1.0 / 240.0).unwrap();
    }

    assert_eq!(regular.positions(), deferred.positions());
    assert_eq!(regular.velocities(), deferred.velocities());
    assert_eq!(regular.diagnostics(), deferred.diagnostics());
}

#[test]
fn default_simulation_remains_finite_over_time_with_recommended_substeps() {
    let mut simulation = Simulation::new(SimulationConfig::default()).unwrap();
    let outer_dt = 1.0 / 240.0;
    let substeps = simulation.recommended_substeps(outer_dt).unwrap();
    let substep_dt = outer_dt / substeps as f64;

    for _ in 0..(10 * 240) {
        for _ in 0..substeps {
            simulation.step(substep_dt).unwrap();
        }
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite() && position.length() < 100.0)
    );
}

#[test]
fn recommended_substeps_stabilize_piece_count_changes() {
    let outer_dt = 1.0 / 240.0;

    for segment_count in [1, 2, 8, 20, 32, 64] {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count,
            ..SimulationConfig::default()
        })
        .unwrap();
        let substeps = simulation.recommended_substeps(outer_dt).unwrap();
        let substep_dt = outer_dt / substeps as f64;

        for _ in 0..(4 * 240) {
            for _ in 0..substeps {
                simulation.step(substep_dt).unwrap();
            }
        }

        assert!(
            simulation
                .positions()
                .iter()
                .all(|position| position.is_finite() && position.length() < 100.0),
            "segment count {segment_count} diverged with {substeps} substeps"
        );
    }
}

#[test]
fn refined_ropes_request_more_substeps() {
    let outer_dt = 1.0 / 240.0;
    let coarse = Simulation::new(SimulationConfig {
        segment_count: 4,
        ..SimulationConfig::default()
    })
    .unwrap();
    let refined = Simulation::new(SimulationConfig {
        segment_count: 64,
        ..SimulationConfig::default()
    })
    .unwrap();

    assert!(
        refined.recommended_substeps(outer_dt).unwrap()
            > coarse.recommended_substeps(outer_dt).unwrap()
    );
}

#[test]
fn automatic_substeps_handle_the_stiffest_ui_configuration() {
    let outer_dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_length: 2.0,
        rope_mass: 0.2,
        payload_mass: 10.0,
        axial_rigidity: 100_000.0,
        air_damping_rate: 0.0,
        ..SimulationConfig::default()
    })
    .unwrap();
    let substeps = simulation.recommended_substeps(outer_dt).unwrap();
    let substep_dt = outer_dt / substeps as f64;

    for _ in 0..(2 * 240) {
        for _ in 0..substeps {
            simulation.step(substep_dt).unwrap();
        }
    }

    assert!(substeps > 1);
    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite() && position.length() < 100.0)
    );
}

#[test]
fn kinematic_target_is_exact_while_stepping() {
    let mut simulation = Simulation::new(SimulationConfig::default()).unwrap();
    let target = KinematicTarget::new(Vec2::new(0.25, -0.5), Vec2::new(1.0, 2.0));
    simulation.set_payload_target(Some(target));
    simulation.step(1.0 / 240.0).unwrap();

    assert_eq!(simulation.payload_position(), target.position);
    assert_eq!(simulation.payload_velocity(), target.velocity);
}

#[test]
fn kinematic_target_is_interpolated_across_physics_steps() {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 1,
        gravity: Vec2::ZERO,
        air_damping_rate: 0.0,
        ..SimulationConfig::default()
    })
    .unwrap();
    let start = simulation.payload_position();
    let end = KinematicTarget::new(start + Vec2::new(1.0, 0.0), Vec2::ZERO);
    simulation.interpolate_payload_target(end, 0.1).unwrap();

    simulation.step(0.025).unwrap();
    let intermediate = simulation.payload_position();
    assert!(intermediate.x > start.x);
    assert!(intermediate.x < end.position.x);

    for _ in 0..3 {
        simulation.step(0.025).unwrap();
    }
    assert!((simulation.payload_position() - end.position).length() < EPSILON);
    assert!((simulation.payload_velocity() - end.velocity).length() < EPSILON);
}

#[test]
fn kinematic_interpolation_cannot_overshoot_its_endpoints() {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 1,
        gravity: Vec2::ZERO,
        air_damping_rate: 0.0,
        ..SimulationConfig::default()
    })
    .unwrap();
    let start = simulation.payload_position();
    let end = KinematicTarget::new(start + Vec2::new(0.1, 0.0), Vec2::new(100.0, 0.0));
    simulation.interpolate_payload_target(end, 0.1).unwrap();

    simulation.step(0.05).unwrap();
    let intermediate = simulation.payload_position();
    assert!(intermediate.x >= start.x);
    assert!(intermediate.x <= end.position.x);
}

#[test]
fn diagnostics_capture_scripted_slack_to_taut_interaction_cost() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 12,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let targets = [
        Vec2::new(2.0, -8.0),
        Vec2::new(-3.0, -11.0),
        Vec2::new(1.0, -12.5),
        Vec2::new(0.0, -12.0),
    ];
    let mut peak_energy: f64 = 0.0;
    let mut minimum_segment_length = f64::INFINITY;

    for target in targets {
        simulation
            .interpolate_payload_target(KinematicTarget::new(target, Vec2::ZERO), 0.25)
            .unwrap();
        for _ in 0..60 {
            simulation.step(dt).unwrap();
            let diagnostics = simulation.diagnostics();
            peak_energy = peak_energy.max(diagnostics.total_mechanical_energy.abs());
            minimum_segment_length = minimum_segment_length.min(diagnostics.minimum_segment_length);
        }
    }

    let diagnostics = simulation.diagnostics();
    assert!(peak_energy.is_finite());
    assert!(minimum_segment_length.is_finite() && minimum_segment_length >= 0.0);
    assert!(diagnostics.cumulative_prescribed_work.is_finite());
    assert!(diagnostics.maximum_tensile_strain < 0.5);
    assert!(diagnostics.linear_solves > 0);
    assert!(diagnostics.nonlinear_iterations >= diagnostics.linear_solves);
    assert!(diagnostics.linear_solves < 20_000);
}

#[test]
fn backward_euler_sls_survives_repeated_slack_to_taut_transitions() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 20,
        rope_model: RopeModelKind::StandardLinearSolid,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let targets = [
        Vec2::new(0.5, -11.8),
        Vec2::new(-0.5, -11.9),
        Vec2::new(1.0, -11.5),
        Vec2::new(0.0, -12.0),
    ];

    for _ in 0..3 {
        for target in targets {
            // Keep the prescribed trajectory within the frontend's 20 m/s
            // interaction limit. This test targets repeated constitutive
            // active-set changes, not an unbounded endpoint impulse.
            simulation
                .interpolate_payload_target(KinematicTarget::new(target, Vec2::ZERO), 0.25)
                .unwrap();
            for _ in 0..60 {
                let substeps = simulation.recommended_substeps(dt).unwrap();
                for _ in 0..substeps {
                    simulation.step(dt / substeps as f64).unwrap();
                }
            }
        }
    }

    let diagnostics = simulation.diagnostics();
    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(diagnostics.maximum_tensile_strain < 0.5);
    assert!(diagnostics.nonlinear_iterations > 0);
}

#[test]
fn small_drag_and_release_remains_bounded() {
    let outer_dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig::default()).unwrap();
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(0.25, 0.0),
        Vec2::ZERO,
    );
    simulation.interpolate_payload_target(target, 0.1).unwrap();

    for _ in 0..(6 * 240) {
        let substeps = simulation.recommended_substeps(outer_dt).unwrap();
        for _ in 0..substeps {
            simulation.step(outer_dt / substeps as f64).unwrap();
        }
    }

    let release_velocity = simulation.payload_velocity();
    simulation.release_payload(release_velocity);
    for _ in 0..(6 * 240) {
        let substeps = simulation.recommended_substeps(outer_dt).unwrap();
        for _ in 0..substeps {
            simulation.step(outer_dt / substeps as f64).unwrap();
        }
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite() && position.length() < 24.0)
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.2);
}

#[test]
fn backward_euler_advances_the_default_rope_without_substeps() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    assert_eq!(simulation.recommended_substeps(dt).unwrap(), 1);
    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite())
    );
    assert!(simulation.diagnostics().maximum_tensile_strain < 0.1);
}

#[test]
fn tr_bdf2_advances_the_default_rope_without_substeps() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();

    assert_eq!(simulation.recommended_substeps(dt).unwrap(), 1);
    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
fn tr_bdf2_is_second_order_for_a_single_axial_oscillator() {
    fn error(dt: f64) -> f64 {
        let config = SimulationConfig {
            segment_count: 1,
            rope_length: 1.0,
            rope_mass: 0.001,
            payload_mass: 1.0,
            axial_rigidity: 100.0,
            gravity: Vec2::ZERO,
            air_damping_rate: 0.0,
            integrator: IntegratorKind::TrBdf2,
            ..SimulationConfig::default()
        };
        let mut simulation = Simulation::new(config).unwrap();
        let extension = 0.1;
        simulation.set_payload_target(Some(KinematicTarget::new(
            Vec2::new(0.0, -(config.rope_length + extension)),
            Vec2::ZERO,
        )));
        simulation.release_payload(Vec2::ZERO);

        let duration = 0.5;
        let steps = (duration / dt) as usize;
        for _ in 0..steps {
            simulation.step(dt).unwrap();
        }

        let mass = config.payload_mass + 0.5 * config.rope_mass;
        let frequency = (config.axial_rigidity / config.rope_length / mass).sqrt();
        let exact_extension = extension * (frequency * duration).cos();
        let exact_velocity = -extension * frequency * (frequency * duration).sin();
        let actual_extension = -simulation.payload_position().y - config.rope_length;
        let actual_velocity = -simulation.payload_velocity().y;
        ((actual_extension - exact_extension).powi(2)
            + ((actual_velocity - exact_velocity) / frequency).powi(2))
        .sqrt()
    }

    let coarse = error(0.02);
    let fine = error(0.01);
    assert!(
        coarse > 3.5 * fine,
        "expected second-order convergence, coarse={coarse:e}, fine={fine:e}"
    );
}

#[test]
fn tr_bdf2_retains_second_order_accuracy_with_a_moving_endpoint() {
    fn error(dt: f64) -> f64 {
        let config = SimulationConfig {
            segment_count: 2,
            rope_length: 2.0,
            rope_mass: 1.0,
            payload_mass: 1.0,
            axial_rigidity: 1_000.0,
            rope_model: RopeModelKind::HookeSpring,
            gravity: Vec2::ZERO,
            air_damping_rate: 0.0,
            integrator: IntegratorKind::TrBdf2,
            ..SimulationConfig::default()
        };
        let mut simulation = Simulation::new(config).unwrap();
        let duration = 0.5;
        let boundary_speed = 0.4;
        simulation
            .interpolate_payload_target(
                KinematicTarget::new(Vec2::new(0.0, -2.2), Vec2::new(0.0, -boundary_speed)),
                duration,
            )
            .unwrap();
        for _ in 0..(duration / dt).round() as usize {
            simulation.step(dt).unwrap();
        }

        let frequency = 4_000.0_f64.sqrt();
        let expected_position = 1.0 + 0.5 * boundary_speed * duration
            - 0.5 * boundary_speed / frequency * (frequency * duration).sin();
        let expected_velocity =
            0.5 * boundary_speed - 0.5 * boundary_speed * (frequency * duration).cos();
        let position_error = simulation.positions()[1].y + expected_position;
        let velocity_error = simulation.velocities()[1].y + expected_velocity;
        (position_error * position_error + velocity_error * velocity_error).sqrt()
    }

    let coarse = error(0.01);
    let fine = error(0.005);
    assert!(
        coarse > 2.8 * fine,
        "expected near-second-order moving-boundary convergence, coarse={coarse:e}, fine={fine:e}"
    );
}

#[test]
fn tr_bdf2_damps_an_undamped_oscillator_less_than_backward_euler() {
    fn retained_energy(integrator: IntegratorKind) -> f64 {
        let config = SimulationConfig {
            segment_count: 1,
            rope_length: 1.0,
            rope_mass: 0.001,
            payload_mass: 1.0,
            axial_rigidity: 100.0,
            gravity: Vec2::ZERO,
            air_damping_rate: 0.0,
            integrator,
            ..SimulationConfig::default()
        };
        let mut simulation = Simulation::new(config).unwrap();
        simulation.set_payload_target(Some(KinematicTarget::new(Vec2::new(0.0, -1.1), Vec2::ZERO)));
        simulation.release_payload(Vec2::ZERO);
        for _ in 0..300 {
            simulation.step(1.0 / 60.0).unwrap();
        }
        simulation.diagnostics().total_mechanical_energy
    }

    let backward_euler = retained_energy(IntegratorKind::BackwardEuler);
    let tr_bdf2 = retained_energy(IntegratorKind::TrBdf2);
    assert!(
        tr_bdf2 > 5.0 * backward_euler,
        "expected less numerical damping: TR-BDF2={tr_bdf2:e}, BE={backward_euler:e}"
    );
}

#[test]
#[ignore = "ROS2 is experimental: even free default motion accumulates excessive tensile strain"]
fn rosenbrock_advances_the_default_rope_without_substeps() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();

    assert_eq!(simulation.recommended_substeps(dt).unwrap(), 1);
    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(simulation.diagnostics().maximum_tensile_strain < 0.1);
}

#[test]
fn rosenbrock_is_second_order_for_a_single_axial_oscillator() {
    fn error(dt: f64) -> f64 {
        let config = SimulationConfig {
            segment_count: 1,
            rope_length: 1.0,
            rope_mass: 0.001,
            payload_mass: 1.0,
            axial_rigidity: 100.0,
            gravity: Vec2::ZERO,
            air_damping_rate: 0.0,
            integrator: IntegratorKind::Rosenbrock2,
            ..SimulationConfig::default()
        };
        let mut simulation = Simulation::new(config).unwrap();
        let extension = 0.1;
        simulation.set_payload_target(Some(KinematicTarget::new(
            Vec2::new(0.0, -(config.rope_length + extension)),
            Vec2::ZERO,
        )));
        simulation.release_payload(Vec2::ZERO);

        // Stay in the taut branch; a tension-only rope is not a bilateral
        // harmonic oscillator after the extension crosses zero.
        let duration = 0.1;
        let steps = (duration / dt) as usize;
        for _ in 0..steps {
            simulation.step(dt).unwrap();
        }

        let mass = config.payload_mass + 0.5 * config.rope_mass;
        let frequency = (config.axial_rigidity / config.rope_length / mass).sqrt();
        let exact_extension = extension * (frequency * duration).cos();
        let exact_velocity = -extension * frequency * (frequency * duration).sin();
        let actual_extension = -simulation.payload_position().y - config.rope_length;
        let actual_velocity = -simulation.payload_velocity().y;
        ((actual_extension - exact_extension).powi(2)
            + ((actual_velocity - exact_velocity) / frequency).powi(2))
        .sqrt()
    }

    let coarse = error(0.02);
    let fine = error(0.01);
    assert!(
        coarse > 3.5 * fine,
        "expected second-order convergence, coarse={coarse:e}, fine={fine:e}"
    );
}

#[test]
fn rosenbrock_damps_an_undamped_oscillator_less_than_backward_euler() {
    fn retained_energy(integrator: IntegratorKind) -> f64 {
        let config = SimulationConfig {
            segment_count: 1,
            rope_length: 1.0,
            rope_mass: 0.001,
            payload_mass: 1.0,
            axial_rigidity: 100.0,
            gravity: Vec2::ZERO,
            air_damping_rate: 0.0,
            integrator,
            ..SimulationConfig::default()
        };
        let mut simulation = Simulation::new(config).unwrap();
        simulation.set_payload_target(Some(KinematicTarget::new(
            Vec2::new(0.0, -(simulation.rest_length() + 0.1)),
            Vec2::ZERO,
        )));
        simulation.release_payload(Vec2::ZERO);
        for _ in 0..300 {
            simulation.step(1.0 / 60.0).unwrap();
        }
        simulation.diagnostics().total_mechanical_energy
    }

    let backward_euler = retained_energy(IntegratorKind::BackwardEuler);
    let rosenbrock = retained_energy(IntegratorKind::Rosenbrock2);
    assert!(
        rosenbrock > 1.5 * backward_euler,
        "expected less numerical damping: ROS2={rosenbrock:e}, BE={backward_euler:e}"
    );
}

#[test]
fn backward_euler_remains_stable_for_a_stiffer_rope() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        axial_rigidity: 100_000.0,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.05);
}

#[test]
fn backward_euler_handles_kelvin_voigt_without_air_damping() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::KelvinVoigt,
        axial_viscosity: 2.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
fn backward_euler_handles_sls_without_air_damping() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::StandardLinearSolid,
        axial_rigidity: 30_000.0,
        transient_axial_rigidity: 15_000.0,
        axial_viscosity: 1_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
fn tr_bdf2_handles_sls_without_air_damping() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::StandardLinearSolid,
        axial_rigidity: 30_000.0,
        transient_axial_rigidity: 15_000.0,
        axial_viscosity: 1_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
fn tr_bdf2_handles_kelvin_voigt_without_air_damping() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::KelvinVoigt,
        axial_viscosity: 1_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
#[ignore = "ROS2 is experimental: maximum-piece SLS can produce a singular block factorization"]
fn rosenbrock_handles_sls_at_the_maximum_ui_piece_count() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::StandardLinearSolid,
        axial_rigidity: 30_000.0,
        transient_axial_rigidity: 15_000.0,
        axial_viscosity: 1_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(2 * 240) {
        simulation.step(dt).unwrap();
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
#[ignore = "ROS2 is experimental: maximum-piece Kelvin-Voigt can produce a singular block factorization"]
fn rosenbrock_handles_kelvin_voigt_at_the_maximum_ui_piece_count() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::KelvinVoigt,
        axial_viscosity: 1_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();

    let substeps = simulation.recommended_substeps(dt).unwrap();
    for _ in 0..(2 * 240) {
        for _ in 0..substeps {
            simulation.step(dt / substeps as f64).unwrap();
        }
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
fn backward_euler_handles_the_maximum_ui_piece_count() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..60 {
        simulation.step(dt).unwrap();
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite())
    );
}

#[test]
fn rk4_handles_the_maximum_ui_piece_count_with_recommended_substeps() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        integrator: IntegratorKind::RungeKutta4,
        ..SimulationConfig::default()
    })
    .unwrap();

    let substeps = simulation.recommended_substeps(dt).unwrap();
    assert!(substeps > 1);
    for _ in 0..(2 * 240) {
        for _ in 0..substeps {
            simulation.step(dt / substeps as f64).unwrap();
        }
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.1);
}

#[test]
fn backward_euler_supports_interpolated_payload_motion() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(0.25, 0.0),
        Vec2::ZERO,
    );
    simulation.interpolate_payload_target(target, 0.1).unwrap();

    for _ in 0..240 {
        simulation.step(dt).unwrap();
    }

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.2);
}

#[test]
fn backward_euler_survives_a_sustained_drag_path() {
    sustained_drag_path(64, IntegratorKind::BackwardEuler);
}

#[test]
fn tr_bdf2_survives_a_sustained_drag_path() {
    sustained_drag_path_with_limits(
        SimulationConfig {
            segment_count: 64,
            integrator: IntegratorKind::TrBdf2,
            ..SimulationConfig::default()
        },
        0.35,
    );
}

#[test]
fn tr_bdf2_sls_survives_a_default_sustained_drag_path() {
    sustained_drag_path_with_limits(
        SimulationConfig {
            segment_count: 20,
            rope_model: RopeModelKind::StandardLinearSolid,
            integrator: IntegratorKind::TrBdf2,
            ..SimulationConfig::default()
        },
        0.35,
    );
}

#[test]
fn semi_implicit_euler_survives_a_sustained_drag_path_at_64_pieces() {
    sustained_drag_path(64, IntegratorKind::SemiImplicitEuler);
}

#[test]
fn rk4_survives_a_sustained_drag_path_at_64_pieces() {
    sustained_drag_path(64, IntegratorKind::RungeKutta4);
}

#[test]
#[ignore = "ROS2 is experimental: sustained slack-to-taut interaction remains unstable"]
fn rosenbrock_survives_a_sustained_drag_path_at_64_pieces() {
    sustained_drag_path(64, IntegratorKind::Rosenbrock2);
}

#[test]
#[ignore = "ROS2 is experimental: sustained slack-to-taut interaction remains unstable"]
fn rosenbrock_survives_a_sustained_drag_path_at_20_pieces() {
    sustained_drag_path(20, IntegratorKind::Rosenbrock2);
}

#[test]
#[ignore = "ROS2 is experimental: sustained SLS interaction remains unstable"]
fn rosenbrock_sls_survives_a_sustained_drag_path_at_64_pieces() {
    sustained_drag_path_with_config(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::StandardLinearSolid,
        transient_axial_rigidity: 30_000.0,
        axial_viscosity: 30_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    });
}

#[test]
#[ignore = "ROS2 is experimental: undamped slack-to-taut interaction remains unstable"]
fn rosenbrock_hooke_without_air_survives_a_sustained_drag_path() {
    sustained_drag_path_with_config(SimulationConfig {
        segment_count: 64,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    });
}

#[test]
#[ignore = "ROS2 is experimental: undamped slack-to-taut interaction remains unstable"]
fn rosenbrock_hooke_without_air_survives_drag_at_default_piece_count() {
    sustained_drag_path_with_config(SimulationConfig {
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    });
}

#[test]
fn rosenbrock_survives_small_rapid_mouse_reversals() {
    let physics_dt = 1.0 / 240.0;
    let frame_dt = 1.0 / 60.0;
    let mut simulation = Simulation::new(SimulationConfig {
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let initial = simulation.payload_position();
    let mut previous = initial;

    for frame in 0..300 {
        let offset = if frame % 2 == 0 { 0.05 } else { -0.05 };
        let position = initial + Vec2::new(offset, 0.0);
        let velocity = (position - previous) / frame_dt;
        previous = position;
        simulation
            .interpolate_payload_target(KinematicTarget::new(position, velocity), frame_dt)
            .unwrap();
        for _ in 0..4 {
            let substeps = simulation.recommended_substeps(physics_dt).unwrap();
            for _ in 0..substeps {
                simulation.step(physics_dt / substeps as f64).unwrap();
            }
        }
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.3);
}

#[test]
fn rosenbrock_interaction_substeps_scale_past_the_old_cap() {
    let physics_dt = 1.0 / 240.0;
    let frame_dt = 1.0 / 60.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_length: 2.0,
        rope_mass: 0.2,
        payload_mass: 10.0,
        axial_rigidity: 100_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let initial = simulation.payload_position();
    let mut maximum_substeps = 0;

    for frame in 0..60 {
        let time = frame as f64 * frame_dt;
        let position = initial + Vec2::new(0.05 * (3.0 * time).sin(), 0.0);
        simulation
            .interpolate_payload_target(KinematicTarget::new(position, Vec2::ZERO), frame_dt)
            .unwrap();
        for _ in 0..4 {
            let substeps = simulation.recommended_substeps(physics_dt).unwrap();
            maximum_substeps = maximum_substeps.max(substeps);
            for _ in 0..substeps {
                simulation.step(physics_dt / substeps as f64).unwrap();
            }
        }
    }

    assert!(maximum_substeps > 4);
    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.3);
}

#[test]
fn rosenbrock_substeps_account_for_pending_rapid_payload_motion() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        axial_rigidity: 100_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(5.0, 0.0),
        Vec2::new(500.0, 0.0),
    );
    simulation.interpolate_payload_target(target, dt).unwrap();

    assert!(simulation.recommended_substeps(dt).unwrap() > 32);
}

#[test]
fn backward_euler_substeps_account_for_rapid_kelvin_voigt_boundary_motion() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::KelvinVoigt,
        axial_rigidity: 30_000.0,
        axial_viscosity: 30_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(-0.23, -0.02),
        Vec2::new(-6.0, -3.0),
    );
    simulation.interpolate_payload_target(target, dt).unwrap();

    let substeps = simulation.recommended_substeps(dt).unwrap();
    assert!(substeps > 16);
    for _ in 0..substeps {
        simulation.step(dt / substeps as f64).unwrap();
    }
    assert!(simulation.positions().iter().all(|value| value.is_finite()));
}

#[test]
#[ignore = "ROS2 is experimental: the first rapid prescribed movement remains unstable"]
fn rosenbrock_handles_first_rapid_running_payload_move() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        axial_rigidity: 100_000.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();

    for _ in 0..(3 * 240) {
        let substeps = simulation.recommended_substeps(dt).unwrap();
        for _ in 0..substeps {
            simulation.step(dt / substeps as f64).unwrap();
        }
    }

    let position = simulation.payload_position() + Vec2::new(0.7, 0.0);
    simulation
        .interpolate_payload_target(
            KinematicTarget::new(position, Vec2::new(20.0, 0.0)),
            0.7 / 20.0,
        )
        .unwrap();

    for _ in 0..12 {
        let substeps = simulation.recommended_substeps(dt).unwrap();
        for _ in 0..substeps {
            simulation.step(dt / substeps as f64).unwrap();
        }
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    let strain = simulation.diagnostics().maximum_absolute_strain;
    assert!(strain < 0.3, "maximum strain was {strain}");
}

#[test]
#[ignore = "documents the known ROS2 release instability without a Backward Euler fallback"]
fn rosenbrock_recovers_from_released_payload_displacement() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        axial_rigidity: 100_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let position = simulation.payload_position() + Vec2::new(0.7, 0.0);
    simulation.set_payload_target(Some(KinematicTarget::new(position, Vec2::new(20.0, 0.0))));
    simulation.release_payload(Vec2::new(20.0, 0.0));

    for _ in 0..2 {
        let substeps = simulation.recommended_substeps(dt).unwrap();
        for _ in 0..substeps {
            simulation.step(dt / substeps as f64).unwrap();
        }
    }

    let strain = simulation.diagnostics().maximum_absolute_strain;
    assert!(strain < 1.0, "maximum strain was {strain}");
    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
}

#[test]
#[ignore = "ROS2 is experimental: jittered frontend interaction remains unstable"]
fn rosenbrock_survives_jittered_frontend_mouse_updates() {
    const FIXED_DT: f64 = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::StandardLinearSolid,
        transient_axial_rigidity: 30_000.0,
        axial_viscosity: 30_000.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::Rosenbrock2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let initial = simulation.payload_position();
    let frame_times = [
        1.0 / 300.0,
        1.0 / 144.0,
        1.0 / 60.0,
        1.0 / 45.0,
        1.0 / 200.0,
        1.0 / 30.0,
    ];
    let mut elapsed: f64 = 0.0;
    let mut accumulator = 0.0;
    let mut previous_pointer = initial;
    let mut drag_velocity = Vec2::ZERO;

    for frame in 0..360 {
        let frame_dt = frame_times[frame % frame_times.len()];
        elapsed += frame_dt;
        let smooth = initial + Vec2::new(1.5 * (1.7 * elapsed).sin(), 1.0 * (1.1 * elapsed).sin());
        let pointer = Vec2::new(
            (smooth.x / 0.02).round() * 0.02,
            (smooth.y / 0.02).round() * 0.02,
        );
        let measured_velocity = (pointer - previous_pointer) / frame_dt;
        previous_pointer = pointer;
        drag_velocity = drag_velocity * 0.55 + measured_velocity * 0.45;
        simulation
            .interpolate_payload_target(
                KinematicTarget::new(pointer, drag_velocity),
                frame_dt.max(FIXED_DT),
            )
            .unwrap();

        accumulator += frame_dt;
        while accumulator >= FIXED_DT {
            let substeps = simulation.recommended_substeps(FIXED_DT).unwrap();
            for _ in 0..substeps {
                simulation.step(FIXED_DT / substeps as f64).unwrap();
            }
            accumulator -= FIXED_DT;
        }
    }

    assert!(simulation.positions().iter().all(|value| value.is_finite()));
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|value| value.is_finite())
    );
    assert!(simulation.diagnostics().maximum_absolute_strain < 0.3);
}

fn sustained_drag_path(segment_count: usize, integrator: IntegratorKind) {
    sustained_drag_path_with_limits(
        SimulationConfig {
            segment_count,
            integrator,
            ..SimulationConfig::default()
        },
        0.3,
    );
}

fn sustained_drag_path_with_config(config: SimulationConfig) {
    sustained_drag_path_with_limits(config, 0.3);
}

fn sustained_drag_path_with_limits(config: SimulationConfig, maximum_allowed_strain: f64) {
    let physics_dt = 1.0 / 240.0;
    let frame_dt = 1.0 / 60.0;
    let segment_count = config.segment_count;
    let integrator = config.integrator;
    let mut simulation = Simulation::new(config).unwrap();
    let initial_position = simulation.payload_position();
    let mut previous_target = initial_position;
    let mut maximum_strain: f64 = 0.0;
    let mut maximum_speed: f64 = 0.0;

    for frame in 0..300 {
        let time = frame as f64 * frame_dt;
        let blend = time.min(1.0);
        let orbit = Vec2::new(3.0 * (2.7 * time).sin(), -7.0 + 2.5 * (1.9 * time).sin());
        let target_position = initial_position * (1.0 - blend) + orbit * blend;
        let target_velocity = (target_position - previous_target) / frame_dt;
        previous_target = target_position;
        simulation
            .interpolate_payload_target(
                KinematicTarget::new(target_position, target_velocity),
                frame_dt,
            )
            .unwrap();

        for _ in 0..4 {
            advance_and_measure(
                &mut simulation,
                physics_dt,
                &mut maximum_strain,
                &mut maximum_speed,
            );
        }
    }

    let release_velocity = simulation.payload_velocity();
    simulation.release_payload(release_velocity);
    for _ in 0..(2 * 240) {
        advance_and_measure(
            &mut simulation,
            physics_dt,
            &mut maximum_strain,
            &mut maximum_speed,
        );
    }

    assert!(
        maximum_strain < maximum_allowed_strain,
        "{integrator:?} reached excessive strain {maximum_strain:.3} with {segment_count} pieces"
    );
    assert!(
        maximum_speed < 110.0,
        "{integrator:?} reached excessive speed {maximum_speed:.3} m/s with {segment_count} pieces"
    );

    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite() && position.length() < 30.0)
    );
}

fn advance_and_measure(
    simulation: &mut Simulation,
    outer_dt: f64,
    maximum_strain: &mut f64,
    maximum_speed: &mut f64,
) {
    let substeps = simulation.recommended_substeps(outer_dt).unwrap();
    for _ in 0..substeps {
        simulation.step(outer_dt / substeps as f64).unwrap();
        *maximum_strain = maximum_strain.max(simulation.diagnostics().maximum_tensile_strain);
        *maximum_speed = maximum_speed.max(
            simulation
                .velocities()
                .iter()
                .map(|velocity| velocity.length())
                .fold(0.0, f64::max),
        );
    }
}

#[test]
fn changing_integrator_preserves_the_current_state() {
    let mut simulation = Simulation::new(SimulationConfig::default()).unwrap();
    simulation.step(1.0 / 1000.0).unwrap();
    let positions_before = simulation.positions().to_vec();
    let velocities_before = simulation.velocities().to_vec();

    let outcome = simulation
        .reconfigure(SimulationConfig {
            integrator: IntegratorKind::BackwardEuler,
            ..simulation.config()
        })
        .unwrap();

    assert_eq!(outcome, ReconfigureOutcome::Updated);
    assert_eq!(simulation.positions(), positions_before);
    assert_eq!(simulation.velocities(), velocities_before);
}

#[test]
fn compressed_discrete_element_restores_its_material_length() {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 1,
        gravity: Vec2::ZERO,
        air_damping_rate: 0.0,
        ..SimulationConfig::default()
    })
    .unwrap();
    let compressed_position = Vec2::new(0.0, -0.5 * simulation.rest_length());
    simulation.set_payload_target(Some(KinematicTarget::new(compressed_position, Vec2::ZERO)));
    simulation.release_payload(Vec2::ZERO);

    simulation.step(0.001).unwrap();

    assert!(simulation.payload_position().y < compressed_position.y);
    assert!(simulation.payload_velocity().y < 0.0);
    assert!(simulation.diagnostics().elastic_energy > 0.0);
}

#[test]
fn air_damping_is_mass_independent() {
    let config = SimulationConfig {
        segment_count: 1,
        gravity: Vec2::ZERO,
        axial_rigidity: 1.0,
        air_damping_rate: 2.0,
        ..SimulationConfig::default()
    };
    let mut simulation = Simulation::new(config).unwrap();
    simulation.release_payload(Vec2::new(4.0, 0.0));
    simulation.step(0.1).unwrap();

    assert!((simulation.payload_velocity().x - 3.2).abs() < EPSILON);
}

#[test]
fn kelvin_voigt_damps_axial_motion_without_air_damping() {
    fn velocity_after_step(rope_model: RopeModelKind) -> f64 {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count: 1,
            rope_mass: 1.0,
            payload_mass: 1.0,
            axial_rigidity: 1.0,
            rope_model,
            axial_viscosity: 120.0,
            air_damping_rate: 0.0,
            gravity: Vec2::ZERO,
            ..SimulationConfig::default()
        })
        .unwrap();
        simulation.release_payload(Vec2::new(0.0, 4.0));
        simulation.step(0.001).unwrap();
        simulation.payload_velocity().y
    }

    let spring_velocity = velocity_after_step(RopeModelKind::HookeSpring);
    let kelvin_voigt_velocity = velocity_after_step(RopeModelKind::KelvinVoigt);

    assert!((spring_velocity - 4.0).abs() < EPSILON);
    assert!(kelvin_voigt_velocity < spring_velocity);
    assert!(kelvin_voigt_velocity > 0.0);
}

#[test]
fn standard_linear_solid_relaxes_its_transient_spring_force() {
    let dt = 0.01;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 1,
        rope_model: RopeModelKind::StandardLinearSolid,
        axial_rigidity: 30_000.0,
        transient_axial_rigidity: 15_000.0,
        axial_viscosity: 1_000.0,
        air_damping_rate: 0.0,
        gravity: Vec2::ZERO,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    let stretched_position = simulation.payload_position() + Vec2::new(0.0, -1.0);
    simulation.set_payload_target(Some(KinematicTarget::new(
        stretched_position,
        Vec2::new(0.0, -100.0),
    )));
    simulation.step(dt).unwrap();
    let energized = simulation.diagnostics().elastic_energy;

    simulation.set_payload_target(Some(KinematicTarget::new(stretched_position, Vec2::ZERO)));
    for _ in 0..200 {
        simulation.step(dt).unwrap();
    }
    let relaxed = simulation.diagnostics().elastic_energy;
    let equilibrium_energy = 0.5 * 30_000.0 / simulation.rest_length();

    assert!(
        energized > relaxed + 100.0,
        "expected transient energy: energized={energized}, relaxed={relaxed}"
    );
    assert!((relaxed - equilibrium_energy).abs() < 1.0e-6 * equilibrium_energy);
}

#[test]
fn topology_changes_reset_the_simulation() {
    let mut simulation = Simulation::new(SimulationConfig::default()).unwrap();
    simulation.step(1.0 / 1000.0).unwrap();
    let outcome = simulation
        .reconfigure(SimulationConfig {
            segment_count: 12,
            ..SimulationConfig::default()
        })
        .unwrap();

    assert_eq!(outcome, ReconfigureOutcome::Reset);
    assert_eq!(simulation.positions().len(), 13);
    assert_eq!(simulation.diagnostics().simulation_time, 0.0);
}

#[test]
fn invalid_configuration_is_rejected() {
    let result = Simulation::new(SimulationConfig {
        rope_length: 0.0,
        ..SimulationConfig::default()
    });

    assert!(matches!(
        result,
        Err(ConfigError::InvalidParameter {
            name: "rope length",
            ..
        })
    ));

    let result = Simulation::new(SimulationConfig {
        rope_model: RopeModelKind::StandardLinearSolid,
        axial_viscosity: 0.0,
        ..SimulationConfig::default()
    });
    assert!(matches!(
        result,
        Err(ConfigError::InvalidParameter {
            name: "axial viscosity",
            ..
        })
    ));
}

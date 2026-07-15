use ropesim_physics::{
    ConfigError, IntegratorKind, KinematicTarget, ReconfigureOutcome, RopeModelKind, Simulation,
    SimulationConfig, StepError, Vec2,
};

const EPSILON: f64 = 1.0e-10;

#[test]
fn backward_euler_is_the_default_integrator() {
    assert_eq!(IntegratorKind::default(), IntegratorKind::BackwardEuler);
    assert_eq!(
        SimulationConfig::default().integrator,
        IntegratorKind::BackwardEuler
    );
}

#[test]
fn calibrated_sls_is_the_default_rope_preset() {
    let config = SimulationConfig::default();

    assert_eq!(RopeModelKind::default(), RopeModelKind::StandardLinearSolid);
    assert_eq!(config.rope_model, RopeModelKind::StandardLinearSolid);
    assert_eq!(config.rope_length, 12.0);
    assert_eq!(config.rope_mass, 0.648);
    assert_eq!(config.payload_mass, 80.0);
    assert_eq!(config.axial_rigidity, 10_335.377);
    assert_eq!(config.transient_axial_rigidity, 18_325.2);
    assert_eq!(config.axial_viscosity, 7_288.0);
    assert_eq!(config.air_damping_rate, 0.0);
}

#[test]
fn every_rope_model_loads_its_own_recommended_parameters() {
    let mut config = SimulationConfig::default();

    config.apply_recommended_rope_model(RopeModelKind::HookeSpring);
    assert_eq!(config.axial_rigidity, 25_281.9);
    assert_eq!(config.air_damping_rate, 0.05);

    config.apply_recommended_rope_model(RopeModelKind::KelvinVoigt);
    assert_eq!(config.axial_rigidity, 10_335.377);
    assert_eq!(config.axial_viscosity, 2_045.3);
    assert_eq!(config.air_damping_rate, 0.0);

    config.apply_recommended_rope_model(RopeModelKind::QuadraticKelvinVoigt);
    assert_eq!(config.axial_rigidity, 30_000.0);
    assert_eq!(config.quadratic_axial_rigidity, 100_000.0);
    assert_eq!(config.axial_viscosity, 0.6);
    assert_eq!(config.air_damping_rate, 0.05);

    config.apply_recommended_rope_model(RopeModelKind::StandardLinearSolid);
    assert_eq!(config.axial_rigidity, 10_335.377);
    assert_eq!(config.transient_axial_rigidity, 18_325.2);
    assert_eq!(config.axial_viscosity, 7_288.0);
    assert_eq!(config.air_damping_rate, 0.0);
}

#[test]
fn bending_is_opt_in_and_old_configs_deserialize_to_axial_only() {
    let config = SimulationConfig::default();
    assert_eq!(config.bending_rigidity, 0.0);
    assert_eq!(config.bending_viscosity, 0.0);

    let mut json = serde_json::to_value(config).unwrap();
    let object = json.as_object_mut().unwrap();
    object.remove("bending_rigidity");
    object.remove("bending_viscosity");
    let restored: SimulationConfig = serde_json::from_value(json).unwrap();

    assert_eq!(restored, config);
}

#[test]
fn implicit_integrators_use_the_banded_solver_with_bending() {
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count: 20,
            bending_rigidity: 0.01,
            bending_viscosity: 0.001,
            air_damping_rate: 0.05,
            integrator,
            ..SimulationConfig::default()
        })
        .unwrap();
        simulation
            .interpolate_payload_target(
                KinematicTarget::new(Vec2::new(2.0, -10.0), Vec2::new(4.0, 4.0)),
                0.5,
            )
            .unwrap();
        let mut maximum_bending_energy = 0.0_f64;

        for _ in 0..240 {
            let outer_dt = 1.0 / 240.0;
            let substeps = simulation.recommended_substeps(outer_dt).unwrap();
            for _ in 0..substeps {
                simulation.step(outer_dt / substeps as f64).unwrap();
            }
            maximum_bending_energy =
                maximum_bending_energy.max(simulation.diagnostics().bending_energy);
        }

        let diagnostics = simulation.diagnostics();
        assert!(maximum_bending_energy > 0.0);
        assert!(diagnostics.block_factorizations > 0);
        assert_eq!(diagnostics.sparse_factorizations, 0);
        assert!(
            simulation
                .positions()
                .iter()
                .all(|position| position.is_finite())
        );
    }
}

#[test]
fn xpbd_bending_changes_the_end_shortened_equilibrium() {
    fn held_diagnostics(bending_rigidity: f64) -> ropesim_physics::Diagnostics {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count: 20,
            bending_rigidity,
            bending_viscosity: 0.001,
            ..SimulationConfig::default()
        })
        .unwrap();
        let target = KinematicTarget::new(Vec2::new(0.5, -6.0), Vec2::ZERO);
        simulation.set_manipulation_target(target);
        for _ in 0..480 {
            simulation.step(1.0 / 240.0).unwrap();
        }
        simulation.diagnostics()
    }

    let axial_only = held_diagnostics(0.0);
    let with_bending = held_diagnostics(0.1);
    assert!(
        (with_bending.maximum_curvature - axial_only.maximum_curvature).abs() > 0.1,
        "bending did not materially change the held shape"
    );
    assert!(with_bending.bending_energy > 0.0);
    assert!(with_bending.maximum_curvature.is_finite());
}

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
fn implicit_integrators_resolve_gravity_at_deep_retry_timestep() {
    let dt = (1.0 / 240.0) / 64.0;
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count: 1,
            axial_rigidity: 1.0,
            rope_model: RopeModelKind::HookeSpring,
            air_damping_rate: 0.0,
            integrator,
            ..SimulationConfig::default()
        })
        .unwrap();

        simulation.step(dt).unwrap();

        let expected_velocity = -9.81 * dt;
        let actual_velocity = simulation.velocities()[1].y;
        assert!(
            (actual_velocity - expected_velocity).abs() < 1.0e-9,
            "{} produced {actual_velocity:e} m/s instead of {expected_velocity:e} m/s",
            integrator.display_name()
        );
    }
}

#[test]
fn implicit_convergence_is_independent_of_world_translation() {
    fn payload_state(integrator: IntegratorKind, anchor: Vec2) -> (Vec2, Vec2) {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count: 1,
            axial_rigidity: 1.0,
            rope_model: RopeModelKind::HookeSpring,
            air_damping_rate: 0.0,
            anchor,
            integrator,
            ..SimulationConfig::default()
        })
        .unwrap();
        simulation.step(5.0e-4).unwrap();
        (
            simulation.payload_position() - anchor,
            simulation.payload_velocity(),
        )
    }

    let translation = Vec2::new(10_000.0, -10_000.0);
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        let local = payload_state(integrator, Vec2::ZERO);
        let translated = payload_state(integrator, translation);
        let position_difference = (local.0 - translated.0).length();
        let velocity_difference = (local.1 - translated.1).length();
        assert!(
            position_difference < 1.0e-8,
            "{} changed position by {position_difference:e} m under translation",
            integrator.display_name()
        );
        assert!(
            velocity_difference < 1.0e-7,
            "{} changed velocity by {velocity_difference:e} m/s under translation",
            integrator.display_name()
        );
    }
}

#[test]
fn xpbd_manipulation_pins_the_payload_and_preserves_boundary_velocity() {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 8,
        rope_model: RopeModelKind::KelvinVoigt,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(Vec2::new(1.5, -6.0), Vec2::new(2.0, -0.5));

    simulation.set_manipulation_target(target);
    for _ in 0..120 {
        assert_eq!(simulation.recommended_substeps(1.0 / 240.0).unwrap(), 1);
        simulation.step(1.0 / 240.0).unwrap();
    }

    assert_eq!(simulation.payload_position(), target.position);
    assert_eq!(simulation.payload_velocity(), target.velocity);
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|velocity| velocity.is_finite())
    );
}

#[test]
fn xpbd_manipulation_rate_limits_extreme_cursor_motion_and_velocity_together() {
    const MAXIMUM_SPEED: f64 = 30.0;
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig::default()).unwrap();
    let start = simulation.payload_position();
    simulation.set_manipulation_target(KinematicTarget::new(
        start + Vec2::new(100.0, 0.0),
        Vec2::new(100.0, 0.0),
    ));

    simulation.step(dt).unwrap();

    assert!((simulation.payload_position() - start).length() <= MAXIMUM_SPEED * dt + 1.0e-12);
    assert!((simulation.payload_velocity().length() - MAXIMUM_SPEED).abs() < 1.0e-12);
}

#[test]
fn xpbd_release_immediately_hands_throw_velocity_to_the_selected_integrator() {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 1,
        gravity: Vec2::ZERO,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let throw_velocity = Vec2::new(3.0, 1.5);
    simulation.set_manipulation_target(KinematicTarget::new(
        simulation.payload_position(),
        throw_velocity,
    ));
    simulation.step(1.0 / 240.0).unwrap();
    simulation.release_manipulation(throw_velocity);

    assert!(!simulation.manipulation_active());
    assert_eq!(simulation.payload_velocity(), throw_velocity);
}

#[test]
fn xpbd_release_handoff_preserves_time_and_the_last_applied_velocity() {
    let mut simulation = Simulation::new(SimulationConfig {
        gravity: Vec2::ZERO,
        integrator: IntegratorKind::BackwardEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let held_velocity = Vec2::new(4.0, -2.0);
    simulation.set_manipulation_target(KinematicTarget::new(
        simulation.payload_position(),
        held_velocity,
    ));
    simulation.step(1.0 / 240.0).unwrap();
    let time_before_release = simulation.diagnostics().simulation_time;
    let payload_position_before_release = simulation.payload_position();

    simulation.release_manipulation(Vec2::new(40.0, 40.0));

    assert_eq!(
        simulation.diagnostics().simulation_time,
        time_before_release
    );
    assert_eq!(
        simulation.payload_position(),
        payload_position_before_release
    );
    assert_eq!(simulation.payload_velocity(), held_velocity);
    assert_eq!(simulation.diagnostics().manipulation_release_handoffs, 1);
}

#[test]
fn hybrid_manipulation_correction_runs_at_120_hz_without_double_stepping_time() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(simulation.payload_position(), Vec2::ZERO);
    simulation.set_manipulation_target(target);

    simulation.step(dt).unwrap();
    let first = simulation.diagnostics();
    assert_eq!(first.simulation_time, dt);
    assert_eq!(
        first.manipulation_corrections + first.manipulation_correction_fallbacks,
        0
    );

    simulation.step(dt).unwrap();
    let second = simulation.diagnostics();
    assert_eq!(second.simulation_time, 2.0 * dt);
    assert_eq!(
        second.manipulation_corrections + second.manipulation_correction_fallbacks,
        1
    );
    assert_eq!(second.manipulation_corrections, 1);
}

#[test]
fn hybrid_sls_drag_converges_near_the_full_rope_radius() {
    let dt = 1.0 / 240.0;
    let step_count = 480;
    let mut simulation = Simulation::new(SimulationConfig {
        rope_model: RopeModelKind::StandardLinearSolid,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let start = simulation.payload_position();
    let end = Vec2::new(11.9, 0.0);
    let velocity = (end - start) / (step_count as f64 * dt);

    for step in 1..=step_count {
        let u = step as f64 / step_count as f64;
        simulation
            .set_manipulation_target(KinematicTarget::new(start + (end - start) * u, velocity));
        simulation.step(dt).unwrap();
    }

    let diagnostics = simulation.diagnostics();
    assert_eq!(simulation.payload_position(), end);
    assert!(
        diagnostics.manipulation_corrections > diagnostics.manipulation_correction_fallbacks,
        "corrections={}, fallbacks={}",
        diagnostics.manipulation_corrections,
        diagnostics.manipulation_correction_fallbacks
    );
    assert!(
        diagnostics.maximum_tensile_strain < 0.02,
        "unexpected held SLS strain {}",
        diagnostics.maximum_tensile_strain
    );
}

#[test]
fn hybrid_keeps_the_finite_xpbd_predictor_when_newton_misses_its_budget() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::QuadraticKelvinVoigt,
        quadratic_axial_rigidity: 10_000_000.0,
        axial_viscosity: 20_000.0,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(Vec2::new(20.0, 0.0), Vec2::new(20.0, 0.0));
    simulation.set_manipulation_target(target);

    simulation.step(dt).unwrap();
    simulation.step(dt).unwrap();

    assert!(simulation.diagnostics().manipulation_correction_fallbacks >= 1);
    assert!(
        simulation
            .positions()
            .iter()
            .all(|position| position.is_finite())
    );
    assert!(
        simulation
            .velocities()
            .iter()
            .all(|velocity| velocity.is_finite())
    );
}

#[test]
fn xpbd_manipulation_evolves_sls_transient_stress_for_handoff() {
    let config = SimulationConfig {
        segment_count: 1,
        gravity: Vec2::ZERO,
        rope_model: RopeModelKind::StandardLinearSolid,
        ..SimulationConfig::default()
    };
    let mut simulation = Simulation::new(config).unwrap();
    let extension = 1.0;
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(0.0, -extension),
        Vec2::new(0.0, -1.0),
    );

    simulation.set_manipulation_target(target);
    simulation.step(0.1).unwrap();

    let relaxed_energy = 0.5 * config.axial_rigidity / simulation.rest_length() * extension.powi(2);
    assert!(simulation.diagnostics().elastic_energy > relaxed_energy);
}

#[test]
fn xpbd_throw_remains_dynamic_with_implicit_integrators_and_every_material() {
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        for rope_model in RopeModelKind::ALL {
            let mut simulation = Simulation::new(SimulationConfig {
                rope_model,
                integrator,
                ..SimulationConfig::default()
            })
            .unwrap();
            let throw_velocity = Vec2::new(3.0, 0.5);
            simulation.set_manipulation_target(KinematicTarget::new(
                simulation.payload_position(),
                throw_velocity,
            ));
            simulation.step(1.0 / 240.0).unwrap();
            simulation.release_manipulation(throw_velocity);
            let release_position = simulation.payload_position();

            for _ in 0..120 {
                let outer_dt = 1.0 / 240.0;
                let substeps = simulation.recommended_substeps(outer_dt).unwrap();
                for _ in 0..substeps {
                    simulation.step(outer_dt / substeps as f64).unwrap();
                }
            }

            assert!(
                simulation.payload_position().x > release_position.x + 0.25,
                "{} did not preserve throwing with {}",
                integrator.display_name(),
                rope_model.display_name()
            );
        }
    }
}

#[test]
fn gravity_accumulates_motion_during_xpbd_manipulation() {
    fn internal_node_after_steps(gravity: Vec2) -> Vec2 {
        let mut simulation = Simulation::new(SimulationConfig {
            segment_count: 2,
            rope_model: RopeModelKind::QuadraticKelvinVoigt,
            gravity,
            ..SimulationConfig::default()
        })
        .unwrap();
        let target = KinematicTarget::new(simulation.payload_position(), Vec2::ZERO);
        simulation.set_manipulation_target(target);
        for _ in 0..20 {
            simulation.step(1.0 / 240.0).unwrap();
        }
        simulation.positions()[1]
    }

    let without_gravity = internal_node_after_steps(Vec2::ZERO);
    let with_gravity = internal_node_after_steps(Vec2::new(0.0, -9.81));

    assert!(with_gravity.y < without_gravity.y - 1.0e-4);
}

#[test]
fn xpbd_manipulation_represents_quadratic_kelvin_voigt_slack_with_folds() {
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 2,
        gravity: Vec2::ZERO,
        ..SimulationConfig::default()
            .with_recommended_rope_model(RopeModelKind::QuadraticKelvinVoigt)
    })
    .unwrap();
    let target = KinematicTarget::new(Vec2::new(0.0, -simulation.rest_length()), Vec2::ZERO);

    simulation.set_manipulation_target(target);
    for _ in 0..120 {
        simulation.step(1.0 / 240.0).unwrap();
    }

    assert_eq!(simulation.payload_position(), target.position);
    let minimum_length = simulation.diagnostics().minimum_segment_length;
    let rest_length = simulation.rest_length();
    assert!(
        (minimum_length - rest_length).abs() < 0.02 * rest_length,
        "minimum element length {minimum_length:.6} differed from rest length {rest_length:.6}"
    );
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
        integrator: IntegratorKind::SemiImplicitEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let refined = Simulation::new(SimulationConfig {
        segment_count: 64,
        integrator: IntegratorKind::SemiImplicitEuler,
        ..SimulationConfig::default()
    })
    .unwrap();

    assert!(
        refined.recommended_substeps(outer_dt).unwrap()
            > coarse.recommended_substeps(outer_dt).unwrap()
    );
}

#[test]
fn quadratic_stiffening_reduces_the_explicit_timestep_when_stretched() {
    let outer_dt = 1.0 / 60.0;
    let mut simulation = Simulation::new(SimulationConfig {
        rope_model: RopeModelKind::QuadraticKelvinVoigt,
        quadratic_axial_rigidity: 1_000_000.0,
        axial_viscosity: 0.0,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::SemiImplicitEuler,
        ..SimulationConfig::default()
    })
    .unwrap();
    let unstretched_substeps = simulation.recommended_substeps(outer_dt).unwrap();
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(0.0, -3.0),
        Vec2::ZERO,
    );
    simulation.set_payload_target(Some(target));
    let stretched_substeps = simulation.recommended_substeps(outer_dt).unwrap();

    assert!(stretched_substeps > unstretched_substeps);
}

#[test]
fn quadratic_kelvin_voigt_samples_its_activation_band_during_payload_motion() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        rope_model: RopeModelKind::QuadraticKelvinVoigt,
        integrator: IntegratorKind::TrBdf2,
        ..SimulationConfig::default()
    })
    .unwrap();
    let target = KinematicTarget::new(
        simulation.payload_position() + Vec2::new(0.0, 0.02),
        Vec2::ZERO,
    );
    simulation.interpolate_payload_target(target, dt).unwrap();

    assert!(simulation.recommended_substeps(dt).unwrap() > 1);
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
        rope_model: RopeModelKind::HookeSpring,
        air_damping_rate: 0.0,
        integrator: IntegratorKind::SemiImplicitEuler,
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
    let coarse = single_axial_oscillator_error(IntegratorKind::TrBdf2, 0.02);
    let fine = single_axial_oscillator_error(IntegratorKind::TrBdf2, 0.01);
    assert!(
        coarse > 3.5 * fine,
        "expected second-order convergence, coarse={coarse:e}, fine={fine:e}"
    );
}

#[test]
fn backward_euler_is_first_order_for_a_single_axial_oscillator() {
    let coarse = single_axial_oscillator_error(IntegratorKind::BackwardEuler, 0.02);
    let fine = single_axial_oscillator_error(IntegratorKind::BackwardEuler, 0.01);
    assert!(
        coarse > 1.7 * fine,
        "expected first-order convergence, coarse={coarse:e}, fine={fine:e}"
    );
}

fn single_axial_oscillator_error(integrator: IntegratorKind, dt: f64) -> f64 {
    let config = SimulationConfig {
        segment_count: 1,
        rope_length: 1.0,
        rope_mass: 0.001,
        payload_mass: 1.0,
        axial_rigidity: 100.0,
        rope_model: RopeModelKind::HookeSpring,
        gravity: Vec2::ZERO,
        air_damping_rate: 0.0,
        integrator,
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
            rope_model: RopeModelKind::HookeSpring,
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
fn backward_euler_handles_quadratic_kelvin_voigt_without_air_damping() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::QuadraticKelvinVoigt,
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
    assert!(simulation.diagnostics().maximum_tensile_strain < 0.1);
}

#[test]
fn quadratic_kelvin_voigt_high_viscosity_settling_is_bounded() {
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        for timestep_scale in [1.0, 0.5, 0.25] {
            let dt = timestep_scale / 240.0;
            let mut simulation = Simulation::new(SimulationConfig {
                segment_count: 32,
                rope_model: RopeModelKind::QuadraticKelvinVoigt,
                axial_viscosity: 20_000.0,
                air_damping_rate: 0.0,
                integrator,
                ..SimulationConfig::default()
            })
            .unwrap();
            let mut maximum_payload_speed = 0.0_f64;
            while simulation.diagnostics().simulation_time < 1.34 {
                simulation.step(dt).unwrap();
                maximum_payload_speed =
                    maximum_payload_speed.max(simulation.payload_velocity().length());
            }
            assert!(maximum_payload_speed < 20.0);
        }
    }
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
fn quadratic_kelvin_voigt_survives_mouse_motion_with_implicit_integrators() {
    for integrator in [IntegratorKind::BackwardEuler, IntegratorKind::TrBdf2] {
        sustained_drag_path_with_config(SimulationConfig {
            rope_model: RopeModelKind::QuadraticKelvinVoigt,
            quadratic_axial_rigidity: 1_000_000.0,
            axial_viscosity: 1_000.0,
            air_damping_rate: 0.0,
            integrator,
            ..SimulationConfig::default()
        });
    }
}

#[test]
fn implicit_integrators_distinguish_smooth_and_quantized_boundary_motion() {
    let uninterrupted =
        prescribed_velocity_experiment(IntegratorKind::TrBdf2, PrescribedMotion::Uninterrupted)
            .unwrap();
    let consistent = prescribed_velocity_experiment(
        IntegratorKind::TrBdf2,
        PrescribedMotion::FramewiseConsistent,
    )
    .unwrap();
    let inconsistent = prescribed_velocity_experiment(
        IntegratorKind::TrBdf2,
        PrescribedMotion::FramewiseInconsistent,
    )
    .unwrap();
    let quantized = prescribed_velocity_experiment(
        IntegratorKind::TrBdf2,
        PrescribedMotion::FramewiseQuantized,
    )
    .unwrap();
    let backward_euler_smooth = prescribed_velocity_experiment(
        IntegratorKind::BackwardEuler,
        PrescribedMotion::Uninterrupted,
    )
    .unwrap();
    let backward_euler_quantized = prescribed_velocity_experiment(
        IntegratorKind::BackwardEuler,
        PrescribedMotion::FramewiseQuantized,
    );
    let backward_euler_quantized_speed = backward_euler_quantized
        .as_ref()
        .map_or(f64::INFINITY, |metrics| metrics.maximum_internal_speed);

    println!(
        "TR-BDF2: smooth {:.3} m/s, framewise consistent {:.3} m/s, inconsistent velocity {:.3} m/s, quantized {:.3} m/s; BE: smooth {:.3} m/s, quantized {:.3} m/s",
        uninterrupted.maximum_internal_speed,
        consistent.maximum_internal_speed,
        inconsistent.maximum_internal_speed,
        quantized.maximum_internal_speed,
        backward_euler_smooth.maximum_internal_speed,
        backward_euler_quantized_speed,
    );
    assert!(
        (uninterrupted.maximum_internal_speed - consistent.maximum_internal_speed).abs()
            < 0.02 * uninterrupted.maximum_internal_speed
    );
    assert!(inconsistent.maximum_internal_speed < 1.2 * consistent.maximum_internal_speed);
    assert!(quantized.maximum_internal_speed > 1.3 * consistent.maximum_internal_speed);
    assert!(backward_euler_quantized_speed > 1.3 * backward_euler_smooth.maximum_internal_speed);
}

#[derive(Clone, Copy)]
enum PrescribedMotion {
    Uninterrupted,
    FramewiseConsistent,
    FramewiseInconsistent,
    FramewiseQuantized,
}

struct PrescribedMotionMetrics {
    maximum_internal_speed: f64,
}

fn prescribed_velocity_experiment(
    integrator: IntegratorKind,
    motion: PrescribedMotion,
) -> Result<PrescribedMotionMetrics, StepError> {
    const PHYSICS_DT: f64 = 1.0 / 240.0;
    const SETTLING_STEPS: usize = 240;
    const MOTION_STEPS: usize = 240;

    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 64,
        rope_model: RopeModelKind::QuadraticKelvinVoigt,
        quadratic_axial_rigidity: 1_000.0,
        axial_viscosity: 15_000.0,
        air_damping_rate: 0.5,
        integrator,
        ..SimulationConfig::default()
    })
    .unwrap();
    let mut maximum_internal_speed = 0.0_f64;

    for _ in 0..SETTLING_STEPS {
        advance_prescribed_experiment(&mut simulation, PHYSICS_DT, &mut maximum_internal_speed)?;
    }
    maximum_internal_speed = 0.0;

    let start = simulation.payload_position();
    let path_velocity = Vec2::new(0.0, 2.5);
    let mut previous_endpoint = start;
    if matches!(motion, PrescribedMotion::Uninterrupted) {
        simulation.interpolate_payload_target(
            KinematicTarget::new(
                start + path_velocity * (MOTION_STEPS as f64 * PHYSICS_DT),
                path_velocity,
            ),
            MOTION_STEPS as f64 * PHYSICS_DT,
        )?;
    }

    for step in 0..MOTION_STEPS {
        if !matches!(motion, PrescribedMotion::Uninterrupted) {
            let ideal_offset = path_velocity * ((step + 1) as f64 * PHYSICS_DT);
            let target_position = if matches!(motion, PrescribedMotion::FramewiseQuantized) {
                // Approximately one vertical screen pixel in bug-4-1. At this
                // speed the cursor alternates between stationary samples and
                // two-pixel jumps even though the intended path is smooth.
                const WORLD_PIXEL_SIZE: f64 = 0.020_119_7;
                Vec2::new(
                    start.x,
                    start.y + (ideal_offset.y / WORLD_PIXEL_SIZE).round() * WORLD_PIXEL_SIZE,
                )
            } else {
                start + ideal_offset
            };
            let endpoint_velocity = match motion {
                PrescribedMotion::FramewiseConsistent => path_velocity,
                PrescribedMotion::FramewiseQuantized => {
                    (target_position - previous_endpoint) / PHYSICS_DT
                }
                PrescribedMotion::FramewiseInconsistent => Vec2::ZERO,
                PrescribedMotion::Uninterrupted => unreachable!(),
            };
            simulation.interpolate_payload_target(
                KinematicTarget::new(target_position, endpoint_velocity),
                PHYSICS_DT,
            )?;
            previous_endpoint = target_position;
        }
        advance_prescribed_experiment(&mut simulation, PHYSICS_DT, &mut maximum_internal_speed)?;
    }

    Ok(PrescribedMotionMetrics {
        maximum_internal_speed,
    })
}

fn advance_prescribed_experiment(
    simulation: &mut Simulation,
    outer_dt: f64,
    maximum_internal_speed: &mut f64,
) -> Result<(), StepError> {
    let substeps = simulation.recommended_substeps(outer_dt)?;
    for _ in 0..substeps {
        simulation.step_without_diagnostics(outer_dt / substeps as f64)?;
        *maximum_internal_speed = maximum_internal_speed.max(
            simulation
                .velocities()
                .iter()
                .take(simulation.velocities().len().saturating_sub(1))
                .map(|velocity| velocity.length())
                .fold(0.0, f64::max),
        );
    }
    Ok(())
}

#[test]
fn backward_euler_handles_the_maximum_ui_piece_count() {
    let dt = 1.0 / 240.0;
    let mut simulation = Simulation::new(SimulationConfig {
        segment_count: 512,
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
fn rk4_handles_64_pieces_with_recommended_substeps() {
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

fn sustained_drag_path(segment_count: usize, integrator: IntegratorKind) {
    sustained_drag_path_with_limits(
        SimulationConfig {
            segment_count,
            rope_model: RopeModelKind::HookeSpring,
            integrator,
            ..SimulationConfig::default()
        },
        0.65,
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
        maximum_speed < 140.0,
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
        integrator: IntegratorKind::SemiImplicitEuler,
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

    let result = Simulation::new(SimulationConfig {
        bending_rigidity: -1.0,
        ..SimulationConfig::default()
    });
    assert!(matches!(
        result,
        Err(ConfigError::InvalidParameter {
            name: "bending rigidity",
            ..
        })
    ));
}

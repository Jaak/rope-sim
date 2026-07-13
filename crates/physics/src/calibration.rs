use std::error::Error;
use std::fmt;

use crate::{ConfigError, IntegratorKind, Simulation, SimulationConfig, StepError, Vec2};

/// Published single-rope measurements used as aggregate calibration targets.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DynamicRopeReference {
    pub name: &'static str,
    pub diameter_metres: f64,
    pub linear_density_kg_per_metre: f64,
    pub static_test_mass: f64,
    pub static_elongation: f64,
    pub drop_test_mass: f64,
    pub drop_test_rope_length: f64,
    pub free_fall_height: f64,
    pub maximum_dynamic_elongation: f64,
    pub maximum_impact_force: f64,
}

impl DynamicRopeReference {
    pub fn drop_test_rope_mass(self) -> f64 {
        self.linear_density_kg_per_metre * self.drop_test_rope_length
    }

    pub fn fall_factor(self) -> f64 {
        self.free_fall_height / self.drop_test_rope_length
    }
}

/// Petzl VOLTA GUIDE 9 mm, interpreted as a single rope.
///
/// Product measurements are 54 g/m, 7.6% static elongation, 34% dynamic
/// elongation, and 8.5 kN impact force. The idealized EN 892/UIAA fall fixture
/// uses 2.6 m of active rope and a 4.6 m free fall (fall factor about 1.77).
///
/// Sources:
/// - <https://www.petzl.com/INT/en/Sport/Ropes/VOLTA-GUIDE-9-mm>
/// - <https://www.sigmadewe.com/fileadmin/user_upload/pdf-Dateien/SEILPHYSIK.pdf>
pub const VOLTA_GUIDE_9MM: DynamicRopeReference = DynamicRopeReference {
    name: "Petzl VOLTA GUIDE 9 mm (single)",
    diameter_metres: 0.009,
    linear_density_kg_per_metre: 0.054,
    static_test_mass: 80.0,
    static_elongation: 0.076,
    drop_test_mass: 80.0,
    drop_test_rope_length: 2.6,
    free_fall_height: 4.6,
    maximum_dynamic_elongation: 0.34,
    maximum_impact_force: 8_500.0,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationSettings {
    pub segment_count: usize,
    pub timestep: f64,
    pub static_duration: f64,
    pub dynamic_duration: f64,
}

impl Default for CalibrationSettings {
    fn default() -> Self {
        Self {
            segment_count: 64,
            timestep: 1.0 / 240.0,
            static_duration: 30.0,
            dynamic_duration: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationMeasurements {
    pub static_elongation: f64,
    pub static_endpoint_speed: f64,
    pub maximum_dynamic_elongation: f64,
    pub peak_anchor_tension: f64,
    pub peak_payload_tension: f64,
    pub peak_anchor_tension_time: f64,
    pub peak_payload_tension_time: f64,
}

impl CalibrationMeasurements {
    pub fn static_elongation_error(self, reference: DynamicRopeReference) -> f64 {
        self.static_elongation - reference.static_elongation
    }

    pub fn dynamic_elongation_error(self, reference: DynamicRopeReference) -> f64 {
        self.maximum_dynamic_elongation - reference.maximum_dynamic_elongation
    }

    pub fn impact_force_error(self, reference: DynamicRopeReference) -> f64 {
        self.peak_payload_tension - reference.maximum_impact_force
    }
}

/// Measure one constitutive parameter set in an idealized EN 892/UIAA fixture.
///
/// The material fields and rope model are taken from `material_config`. Test
/// geometry, masses, gravity, zero environmental damping, and backward Euler
/// are imposed here so comparisons between material models use identical
/// production numerics.
pub fn run_dynamic_rope_calibration(
    material_config: SimulationConfig,
    reference: DynamicRopeReference,
    settings: CalibrationSettings,
) -> Result<CalibrationMeasurements, CalibrationError> {
    validate_settings(settings)?;

    let static_config = fixture_config(
        material_config,
        reference,
        settings,
        reference.static_test_mass,
    );
    let mut static_simulation = Simulation::new(static_config)?;
    advance_for(
        &mut static_simulation,
        settings.static_duration,
        settings.timestep,
        |_, _| true,
    )?;
    let static_elongation = end_to_end_elongation(&static_simulation, reference);

    let dynamic_config = fixture_config(
        material_config,
        reference,
        settings,
        reference.drop_test_mass,
    );
    let mut dynamic_simulation = Simulation::new(dynamic_config)?;
    let impact_velocity = (2.0 * -dynamic_config.gravity.y * reference.free_fall_height).sqrt();
    dynamic_simulation.release_payload(Vec2::new(0.0, -impact_velocity));

    let mut maximum_dynamic_elongation = 0.0_f64;
    let mut peak_anchor_tension = 0.0_f64;
    let mut peak_payload_tension = 0.0_f64;
    let mut peak_anchor_tension_time = 0.0;
    let mut peak_payload_tension_time = 0.0;
    advance_for(
        &mut dynamic_simulation,
        settings.dynamic_duration,
        settings.timestep,
        |simulation, time| {
            maximum_dynamic_elongation =
                maximum_dynamic_elongation.max(end_to_end_elongation(simulation, reference));
            let anchor_tension = simulation.segment_tension(0).unwrap_or(0.0).max(0.0);
            if anchor_tension > peak_anchor_tension {
                peak_anchor_tension = anchor_tension;
                peak_anchor_tension_time = time;
            }
            let payload_tension = simulation
                .segment_tension(settings.segment_count - 1)
                .unwrap_or(0.0)
                .max(0.0);
            if payload_tension > peak_payload_tension {
                peak_payload_tension = payload_tension;
                peak_payload_tension_time = time;
            }
            simulation.payload_velocity().y < 0.0
        },
    )?;

    Ok(CalibrationMeasurements {
        static_elongation,
        static_endpoint_speed: static_simulation.payload_velocity().length(),
        maximum_dynamic_elongation,
        peak_anchor_tension,
        peak_payload_tension,
        peak_anchor_tension_time,
        peak_payload_tension_time,
    })
}

fn fixture_config(
    material_config: SimulationConfig,
    reference: DynamicRopeReference,
    settings: CalibrationSettings,
    payload_mass: f64,
) -> SimulationConfig {
    SimulationConfig {
        segment_count: settings.segment_count,
        rope_length: reference.drop_test_rope_length,
        rope_mass: reference.drop_test_rope_mass(),
        payload_mass,
        air_damping_rate: 0.0,
        gravity: Vec2::new(0.0, -9.81),
        anchor: Vec2::ZERO,
        integrator: IntegratorKind::BackwardEuler,
        ..material_config
    }
}

fn end_to_end_elongation(simulation: &Simulation, reference: DynamicRopeReference) -> f64 {
    ((simulation.payload_position() - simulation.config().anchor).length()
        / reference.drop_test_rope_length
        - 1.0)
        .max(0.0)
}

fn advance_for(
    simulation: &mut Simulation,
    duration: f64,
    timestep: f64,
    mut observe: impl FnMut(&Simulation, f64) -> bool,
) -> Result<(), StepError> {
    let mut elapsed = 0.0;
    while elapsed < duration {
        let outer_step = timestep.min(duration - elapsed);
        let substeps = simulation.recommended_substeps(outer_step)?;
        let substep = outer_step / substeps as f64;
        for _ in 0..substeps {
            simulation.step_without_diagnostics(substep)?;
            elapsed += substep;
            if !observe(simulation, elapsed) {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn validate_settings(settings: CalibrationSettings) -> Result<(), CalibrationError> {
    if settings.segment_count == 0 {
        return Err(CalibrationError::InvalidSettings(
            "segment count must be positive",
        ));
    }
    for (name, value) in [
        ("timestep", settings.timestep),
        ("static duration", settings.static_duration),
        ("dynamic duration", settings.dynamic_duration),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(CalibrationError::InvalidSettings(name));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum CalibrationError {
    InvalidSettings(&'static str),
    InvalidConfiguration(ConfigError),
    Integration(StepError),
}

impl fmt::Display for CalibrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSettings(name) => write!(formatter, "invalid calibration {name}"),
            Self::InvalidConfiguration(error) => error.fmt(formatter),
            Self::Integration(error) => error.fmt(formatter),
        }
    }
}

impl Error for CalibrationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidSettings(_) => None,
            Self::InvalidConfiguration(error) => Some(error),
            Self::Integration(error) => Some(error),
        }
    }
}

impl From<ConfigError> for CalibrationError {
    fn from(error: ConfigError) -> Self {
        Self::InvalidConfiguration(error)
    }
}

impl From<StepError> for CalibrationError {
    fn from(error: StepError) -> Self {
        Self::Integration(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_derives_fixture_mass_and_fall_factor() {
        assert!((VOLTA_GUIDE_9MM.drop_test_rope_mass() - 0.1404).abs() < 1.0e-12);
        assert!((VOLTA_GUIDE_9MM.fall_factor() - 4.6 / 2.6).abs() < 1.0e-12);
    }

    #[test]
    fn short_fixture_run_is_finite() {
        let measurements = run_dynamic_rope_calibration(
            SimulationConfig {
                rope_model: crate::RopeModelKind::HookeSpring,
                ..SimulationConfig::default()
            },
            VOLTA_GUIDE_9MM,
            CalibrationSettings {
                segment_count: 4,
                timestep: 1.0 / 240.0,
                static_duration: 0.05,
                dynamic_duration: 0.05,
            },
        )
        .unwrap();

        assert!(measurements.static_elongation.is_finite());
        assert!(measurements.maximum_dynamic_elongation.is_finite());
        assert!(measurements.peak_anchor_tension.is_finite());
        assert!(measurements.peak_payload_tension.is_finite());
    }
}

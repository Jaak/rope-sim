use std::error::Error;
use std::fmt;

use crate::integrators::IntegratorKind;
use crate::math::Vec2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RopeModelKind {
    #[default]
    HookeSpring,
    KelvinVoigt,
    StandardLinearSolid,
}

impl RopeModelKind {
    pub const ALL: [Self; 3] = [
        Self::HookeSpring,
        Self::KelvinVoigt,
        Self::StandardLinearSolid,
    ];

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::HookeSpring => "Hooke spring",
            Self::KelvinVoigt => "Kelvin-Voigt",
            Self::StandardLinearSolid => "Standard linear solid",
        }
    }
}

/// Configuration shared by all frontends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationConfig {
    /// Number of axial elements. The node count is one greater.
    pub segment_count: usize,
    /// Unstretched rope length in metres.
    pub rope_length: f64,
    /// Total distributed rope mass in kilograms.
    pub rope_mass: f64,
    /// Point mass attached to the last node in kilograms.
    pub payload_mass: f64,
    /// Combined axial rigidity EA in newtons.
    pub axial_rigidity: f64,
    /// Additional instantaneous axial rigidity of the Maxwell branch in the
    /// standard linear solid model, in newtons.
    pub transient_axial_rigidity: f64,
    /// Axial constitutive model.
    pub rope_model: RopeModelKind,
    /// Combined axial viscosity eta*A in newton-seconds.
    ///
    /// A discrete Kelvin-Voigt element uses `eta*A / rest_length` in N*s/m,
    /// making this parameter independent of the number of rope pieces. In the
    /// standard linear solid it sets the Maxwell relaxation time together with
    /// `transient_axial_rigidity`.
    pub axial_viscosity: f64,
    /// Mass-proportional damping rate in inverse seconds.
    pub air_damping_rate: f64,
    /// Gravitational acceleration in metres per second squared.
    pub gravity: Vec2,
    /// Fixed endpoint in world coordinates.
    pub anchor: Vec2,
    /// Time integration method.
    pub integrator: IntegratorKind,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            segment_count: 20,
            // Representative 10.5 mm low-stretch kernmantle rope:
            // 75 g/m, with roughly 3.4% static elongation.
            rope_length: 12.0,
            rope_mass: 0.9,
            payload_mass: 80.0,
            axial_rigidity: 30_000.0,
            transient_axial_rigidity: 15_000.0,
            rope_model: RopeModelKind::HookeSpring,
            axial_viscosity: 1_000.0,
            air_damping_rate: 0.5,
            gravity: Vec2::new(0.0, -9.81),
            anchor: Vec2::ZERO,
            integrator: IntegratorKind::SemiImplicitEuler,
        }
    }
}

impl SimulationConfig {
    pub fn validate(self) -> Result<Self, ConfigError> {
        if self.segment_count == 0 {
            return Err(ConfigError::ZeroSegments);
        }

        validate_positive("rope length", self.rope_length)?;
        validate_positive("rope mass", self.rope_mass)?;
        validate_positive("payload mass", self.payload_mass)?;
        validate_positive("axial rigidity", self.axial_rigidity)?;
        validate_positive("transient axial rigidity", self.transient_axial_rigidity)?;
        if self.rope_model == RopeModelKind::StandardLinearSolid {
            validate_positive("axial viscosity", self.axial_viscosity)?;
        } else {
            validate_nonnegative("axial viscosity", self.axial_viscosity)?;
        }

        validate_nonnegative("air damping rate", self.air_damping_rate)?;
        if !self.gravity.is_finite() {
            return Err(ConfigError::InvalidVector("gravity"));
        }
        if !self.anchor.is_finite() {
            return Err(ConfigError::InvalidVector("anchor"));
        }

        Ok(self)
    }
}

fn validate_nonnegative(name: &'static str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value < 0.0 {
        Err(ConfigError::InvalidParameter { name, value })
    } else {
        Ok(())
    }
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), ConfigError> {
    if !value.is_finite() || value <= 0.0 {
        Err(ConfigError::InvalidParameter { name, value })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    ZeroSegments,
    InvalidParameter { name: &'static str, value: f64 },
    InvalidVector(&'static str),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSegments => write!(f, "the rope must have at least one segment"),
            Self::InvalidParameter { name, value } => {
                write!(f, "{name} must be finite and positive (received {value})")
            }
            Self::InvalidVector(name) => write!(f, "{name} must contain finite components"),
        }
    }
}

impl Error for ConfigError {}

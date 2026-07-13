use std::error::Error;
use std::fmt;

use crate::integrators::IntegratorKind;
use crate::math::Vec2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeModelKind {
    HookeSpring,
    KelvinVoigt,
    QuadraticKelvinVoigt,
    #[default]
    StandardLinearSolid,
}

impl RopeModelKind {
    pub const ALL: [Self; 4] = [
        Self::HookeSpring,
        Self::KelvinVoigt,
        Self::QuadraticKelvinVoigt,
        Self::StandardLinearSolid,
    ];

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::HookeSpring => "Hooke spring",
            Self::KelvinVoigt => "Kelvin-Voigt",
            Self::QuadraticKelvinVoigt => "Quadratic Kelvin-Voigt (tension only)",
            Self::StandardLinearSolid => "Standard linear solid",
        }
    }
}

/// Configuration shared by all frontends.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Quadratic tensile rigidity in newtons. For tensile strain `e`, the
    /// quadratic Kelvin-Voigt model contributes `quadratic_axial_rigidity*e^2`.
    #[serde(default = "default_quadratic_axial_rigidity")]
    pub quadratic_axial_rigidity: f64,
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
            // Petzl VOLTA GUIDE 9 mm reference scene: 54 g/m and an 80 kg
            // single-rope test mass. SLS parameters are calibrated separately
            // against its published static and EN 892/UIAA dynamic results.
            rope_length: 12.0,
            rope_mass: 0.648,
            payload_mass: 80.0,
            axial_rigidity: SLS_RELAXED_RIGIDITY,
            quadratic_axial_rigidity: QKV_QUADRATIC_RIGIDITY,
            transient_axial_rigidity: SLS_TRANSIENT_RIGIDITY,
            rope_model: RopeModelKind::StandardLinearSolid,
            axial_viscosity: SLS_VISCOSITY,
            air_damping_rate: 0.0,
            gravity: Vec2::new(0.0, -9.81),
            anchor: Vec2::ZERO,
            integrator: IntegratorKind::BackwardEuler,
        }
    }
}

impl SimulationConfig {
    /// Load the recommended parameters for one constitutive model.
    ///
    /// SLS is calibrated to the VOLTA GUIDE 9 mm aggregate measurements, KV
    /// is the best static-constrained compromise, and Hooke/QKV are deliberately
    /// lively illustrative presets rather than claimed material fits.
    pub fn apply_recommended_rope_model(&mut self, rope_model: RopeModelKind) {
        self.rope_model = rope_model;
        match rope_model {
            RopeModelKind::HookeSpring => {
                self.axial_rigidity = HOOKE_DYNAMIC_RIGIDITY;
                self.air_damping_rate = FUN_AIR_DAMPING_RATE;
            }
            RopeModelKind::KelvinVoigt => {
                self.axial_rigidity = RELAXED_RIGIDITY;
                self.axial_viscosity = KELVIN_VOIGT_VISCOSITY;
                self.air_damping_rate = 0.0;
            }
            RopeModelKind::QuadraticKelvinVoigt => {
                self.axial_rigidity = QKV_LINEAR_RIGIDITY;
                self.quadratic_axial_rigidity = QKV_QUADRATIC_RIGIDITY;
                self.axial_viscosity = QKV_VISCOSITY;
                self.air_damping_rate = FUN_AIR_DAMPING_RATE;
            }
            RopeModelKind::StandardLinearSolid => {
                self.axial_rigidity = SLS_RELAXED_RIGIDITY;
                self.transient_axial_rigidity = SLS_TRANSIENT_RIGIDITY;
                self.axial_viscosity = SLS_VISCOSITY;
                self.air_damping_rate = 0.0;
            }
        }
    }

    pub fn with_recommended_rope_model(mut self, rope_model: RopeModelKind) -> Self {
        self.apply_recommended_rope_model(rope_model);
        self
    }

    pub fn validate(self) -> Result<Self, ConfigError> {
        if self.segment_count == 0 {
            return Err(ConfigError::ZeroSegments);
        }

        validate_positive("rope length", self.rope_length)?;
        validate_positive("rope mass", self.rope_mass)?;
        validate_positive("payload mass", self.payload_mass)?;
        validate_positive("axial rigidity", self.axial_rigidity)?;
        validate_nonnegative("quadratic axial rigidity", self.quadratic_axial_rigidity)?;
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

const fn default_quadratic_axial_rigidity() -> f64 {
    QKV_QUADRATIC_RIGIDITY
}

const RELAXED_RIGIDITY: f64 = 10_335.377;
const HOOKE_DYNAMIC_RIGIDITY: f64 = 25_281.9;
const KELVIN_VOIGT_VISCOSITY: f64 = 2_045.3;
const QKV_LINEAR_RIGIDITY: f64 = 30_000.0;
const QKV_QUADRATIC_RIGIDITY: f64 = 100_000.0;
const QKV_VISCOSITY: f64 = 0.6;
const SLS_RELAXED_RIGIDITY: f64 = RELAXED_RIGIDITY;
const SLS_TRANSIENT_RIGIDITY: f64 = 18_325.2;
const SLS_VISCOSITY: f64 = 7_288.0;
const FUN_AIR_DAMPING_RATE: f64 = 0.05;

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

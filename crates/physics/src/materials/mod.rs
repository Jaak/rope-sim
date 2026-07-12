mod hooke;
mod kelvin_voigt;
mod standard_linear_solid;

use crate::config::{RopeModelKind, SimulationConfig};

use hooke::Hooke;
use kelvin_voigt::KelvinVoigt;
use standard_linear_solid::StandardLinearSolid;

#[derive(Clone, Copy, Debug)]
pub(crate) struct AxialKinematics {
    pub extension: f64,
    pub extension_rate: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AxialResponse {
    pub force: f64,
    pub length_tangent: f64,
    pub rate_tangent: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct MaterialStateTangents {
    pub extension: f64,
    pub extension_rate: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum AxialMaterial {
    Hooke(Hooke),
    KelvinVoigt(KelvinVoigt),
    StandardLinearSolid(StandardLinearSolid),
}

impl AxialMaterial {
    pub fn from_config(config: SimulationConfig) -> Self {
        match config.rope_model {
            RopeModelKind::HookeSpring => Self::Hooke(Hooke::new(config.axial_rigidity)),
            RopeModelKind::KelvinVoigt => Self::KelvinVoigt(KelvinVoigt::new(
                config.axial_rigidity,
                config.axial_viscosity,
            )),
            RopeModelKind::StandardLinearSolid => {
                Self::StandardLinearSolid(StandardLinearSolid::new(
                    config.axial_rigidity,
                    config.transient_axial_rigidity,
                    config.axial_viscosity,
                ))
            }
        }
    }

    pub fn response(
        self,
        kinematics: AxialKinematics,
        material_state: f64,
        rest_length: f64,
    ) -> AxialResponse {
        match self {
            Self::Hooke(material) => material.response(kinematics, rest_length),
            Self::KelvinVoigt(material) => material.response(kinematics, rest_length),
            Self::StandardLinearSolid(material) => {
                material.response(kinematics, material_state, rest_length)
            }
        }
    }

    pub fn backward_euler_response(
        self,
        kinematics: AxialKinematics,
        material_state: f64,
        rest_length: f64,
        dt: f64,
    ) -> AxialResponse {
        match self {
            Self::StandardLinearSolid(material) => {
                material.backward_euler_response(kinematics, material_state, rest_length, dt)
            }
            Self::Hooke(_) | Self::KelvinVoigt(_) => {
                self.response(kinematics, material_state, rest_length)
            }
        }
    }

    pub fn has_internal_state(self) -> bool {
        matches!(self, Self::StandardLinearSolid(_))
    }

    pub fn state_derivative(
        self,
        extension_rate: f64,
        material_state: f64,
        rest_length: f64,
    ) -> f64 {
        match self {
            Self::StandardLinearSolid(material) => {
                material.state_derivative(extension_rate, material_state, rest_length)
            }
            Self::Hooke(_) | Self::KelvinVoigt(_) => 0.0,
        }
    }

    pub fn force_state_tangent(self) -> f64 {
        match self {
            Self::StandardLinearSolid(_) => 1.0,
            Self::Hooke(_) | Self::KelvinVoigt(_) => 0.0,
        }
    }

    pub fn state_tangents(self, rest_length: f64) -> MaterialStateTangents {
        match self {
            Self::StandardLinearSolid(material) => MaterialStateTangents {
                extension: 0.0,
                extension_rate: material.transient_rigidity() / rest_length,
            },
            Self::Hooke(_) | Self::KelvinVoigt(_) => MaterialStateTangents::default(),
        }
    }

    pub fn backward_euler_state(
        self,
        extension_rate: f64,
        initial_material_state: f64,
        rest_length: f64,
        dt: f64,
    ) -> f64 {
        match self {
            Self::StandardLinearSolid(material) => material.backward_euler_state(
                extension_rate,
                initial_material_state,
                rest_length,
                dt,
            ),
            Self::Hooke(_) | Self::KelvinVoigt(_) => initial_material_state,
        }
    }

    pub fn stored_energy(self, extension: f64, material_state: f64, rest_length: f64) -> f64 {
        match self {
            Self::Hooke(material) => material.stored_energy(extension, rest_length),
            Self::KelvinVoigt(material) => material.stored_energy(extension, rest_length),
            Self::StandardLinearSolid(material) => {
                material.stored_energy(extension, material_state, rest_length)
            }
        }
    }

    pub fn instantaneous_rigidity(self) -> f64 {
        match self {
            Self::Hooke(material) => material.rigidity(),
            Self::KelvinVoigt(material) => material.rigidity(),
            Self::StandardLinearSolid(material) => material.instantaneous_rigidity(),
        }
    }

    pub fn explicit_viscosity(self) -> f64 {
        match self {
            Self::KelvinVoigt(material) => material.viscosity(),
            Self::Hooke(_) | Self::StandardLinearSolid(_) => 0.0,
        }
    }

    pub fn relaxation_rate(self) -> f64 {
        match self {
            Self::StandardLinearSolid(material) => material.relaxation_rate(),
            Self::Hooke(_) | Self::KelvinVoigt(_) => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REST_LENGTH: f64 = 2.0;
    const KINEMATICS: AxialKinematics = AxialKinematics {
        extension: 0.2,
        extension_rate: 0.4,
    };

    #[test]
    fn hooke_response_exposes_force_and_tangent() {
        let material = AxialMaterial::Hooke(Hooke::new(30_000.0));
        let response = material.response(KINEMATICS, 0.0, REST_LENGTH);

        assert_eq!(response.force, 3_000.0);
        assert_eq!(response.length_tangent, 15_000.0);
        assert_eq!(response.rate_tangent, 0.0);
    }

    #[test]
    fn kelvin_voigt_response_includes_viscous_force_and_tangent() {
        let material = AxialMaterial::KelvinVoigt(KelvinVoigt::new(30_000.0, 1_000.0));
        let response = material.response(KINEMATICS, 0.0, REST_LENGTH);

        assert_eq!(response.force, 3_200.0);
        assert_eq!(response.length_tangent, 15_000.0);
        assert_eq!(response.rate_tangent, 500.0);
    }

    #[test]
    fn sls_backward_euler_response_uses_eliminated_internal_state() {
        let material = AxialMaterial::StandardLinearSolid(StandardLinearSolid::new(
            30_000.0, 15_000.0, 1_000.0,
        ));
        let dt = 0.01;
        let state = material.backward_euler_state(0.4, 100.0, REST_LENGTH, dt);
        let response = material.backward_euler_response(KINEMATICS, state, REST_LENGTH, dt);

        let denominator = 1.0 + dt * 15_000.0 / 1_000.0;
        let expected_state = (100.0 + dt * 15_000.0 * 0.4 / REST_LENGTH) / denominator;
        assert!((state - expected_state).abs() < 1.0e-12);
        assert!((response.force - (3_000.0 + expected_state)).abs() < 1.0e-12);
        assert!((response.rate_tangent - 75.0 / denominator).abs() < 1.0e-12);
    }
}

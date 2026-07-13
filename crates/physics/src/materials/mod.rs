mod hooke;
mod kelvin_voigt;
mod quadratic_kelvin_voigt;
mod standard_linear_solid;

use crate::config::{RopeModelKind, SimulationConfig};

use hooke::Hooke;
use kelvin_voigt::KelvinVoigt;
use quadratic_kelvin_voigt::QuadraticKelvinVoigt;
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
    QuadraticKelvinVoigt(QuadraticKelvinVoigt),
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
            RopeModelKind::QuadraticKelvinVoigt => {
                Self::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
                    config.axial_rigidity,
                    config.quadratic_axial_rigidity,
                    config.axial_viscosity,
                ))
            }
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
            Self::QuadraticKelvinVoigt(material) => material.response(kinematics, rest_length),
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
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => {
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
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => 0.0,
        }
    }

    pub fn force_state_tangent(self) -> f64 {
        match self {
            Self::StandardLinearSolid(_) => 1.0,
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => 0.0,
        }
    }

    pub fn state_tangents(self, rest_length: f64) -> MaterialStateTangents {
        match self {
            Self::StandardLinearSolid(material) => MaterialStateTangents {
                extension: 0.0,
                extension_rate: material.transient_rigidity() / rest_length,
            },
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => {
                MaterialStateTangents::default()
            }
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
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => {
                initial_material_state
            }
        }
    }

    pub fn stored_energy(self, extension: f64, material_state: f64, rest_length: f64) -> f64 {
        match self {
            Self::Hooke(material) => material.stored_energy(extension, rest_length),
            Self::KelvinVoigt(material) => material.stored_energy(extension, rest_length),
            Self::QuadraticKelvinVoigt(material) => material.stored_energy(extension, rest_length),
            Self::StandardLinearSolid(material) => {
                material.stored_energy(extension, material_state, rest_length)
            }
        }
    }

    pub fn instantaneous_tangent_rigidity(
        self,
        kinematics: AxialKinematics,
        rest_length: f64,
    ) -> f64 {
        match self {
            Self::Hooke(material) => material.rigidity(),
            Self::KelvinVoigt(material) => material.rigidity(),
            Self::QuadraticKelvinVoigt(material) => {
                material.tangent_rigidity(kinematics, rest_length)
            }
            Self::StandardLinearSolid(material) => material.instantaneous_rigidity(),
        }
    }

    pub fn explicit_viscosity(self) -> f64 {
        match self {
            Self::KelvinVoigt(material) => material.viscosity(),
            Self::QuadraticKelvinVoigt(material) => material.explicit_viscosity_bound(),
            Self::Hooke(_) | Self::StandardLinearSolid(_) => 0.0,
        }
    }

    pub fn relaxation_rate(self) -> f64 {
        match self {
            Self::StandardLinearSolid(material) => material.relaxation_rate(),
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => 0.0,
        }
    }

    pub fn maximum_kinematic_travel_fraction(self) -> f64 {
        match self {
            Self::QuadraticKelvinVoigt(material) => 0.5 * material.activation_strain(),
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::StandardLinearSolid(_) => 0.1,
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
    fn quadratic_kelvin_voigt_response_uses_strain_and_exact_tangents() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let response = material.response(KINEMATICS, 0.0, REST_LENGTH);
        let elastic_force: f64 = 4_000.0;
        let damping_scale: f64 = 3.0;
        let damping_activation = elastic_force / (elastic_force + damping_scale);
        let damping_activation_tangent = damping_scale / (elastic_force + damping_scale).powi(2);
        let raw_force = elastic_force + 1_000.0 * damping_activation * 0.2;
        let ratio = raw_force / elastic_force;
        let radius = ratio.hypot(0.05);
        let normalization = 1.0 + 1.0_f64.hypot(0.05);
        let value_ratio = (ratio + radius) / normalization;
        let raw_tangent = (1.0 + ratio / radius) / normalization;
        let elastic_tangent = value_ratio - ratio * raw_tangent;
        let elastic_strain_tangent = 50_000.0;
        let raw_strain_tangent =
            elastic_strain_tangent * (1.0 + 1_000.0 * damping_activation_tangent * 0.2);
        let expected_force = elastic_force * value_ratio;
        let expected_length_tangent = (raw_tangent * raw_strain_tangent
            + elastic_tangent * elastic_strain_tangent)
            / REST_LENGTH;
        let expected_rate_tangent = raw_tangent * 1_000.0 * damping_activation / REST_LENGTH;

        assert!((response.force - expected_force).abs() < 1.0e-12);
        assert!((response.length_tangent - expected_length_tangent).abs() < 1.0e-12);
        assert!((response.rate_tangent - expected_rate_tangent).abs() < 1.0e-12);
    }

    #[test]
    fn quadratic_kelvin_voigt_never_produces_compression() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let slack = material.response(
            AxialKinematics {
                extension: -0.1,
                extension_rate: 100.0,
            },
            0.0,
            REST_LENGTH,
        );
        let rapidly_shortening = material.response(
            AxialKinematics {
                extension: 0.1,
                extension_rate: -100.0,
            },
            0.0,
            REST_LENGTH,
        );

        assert_eq!(slack.force, 0.0);
        assert_eq!(slack.length_tangent, 0.0);
        assert_eq!(slack.rate_tangent, 0.0);
        assert!(rapidly_shortening.force >= 0.0);
        assert!(rapidly_shortening.force < 1.0);
    }

    #[test]
    fn quadratic_kelvin_voigt_energy_is_mesh_independent() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let coarse = material.stored_energy(0.2, 0.0, 2.0);
        let refined = 2.0 * material.stored_energy(0.1, 0.0, 1.0);

        assert!((coarse - refined).abs() < 1.0e-12);
        assert_eq!(material.stored_energy(-0.1, 0.0, 1.0), 0.0);
    }

    #[test]
    fn quadratic_kelvin_voigt_regularized_tangents_match_finite_differences() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let kinematics = AxialKinematics {
            extension: 0.002,
            extension_rate: -0.04,
        };
        let response = material.response(kinematics, 0.0, REST_LENGTH);
        let step = 1.0e-8;

        let position_difference = (material
            .response(
                AxialKinematics {
                    extension: kinematics.extension + step,
                    ..kinematics
                },
                0.0,
                REST_LENGTH,
            )
            .force
            - material
                .response(
                    AxialKinematics {
                        extension: kinematics.extension - step,
                        ..kinematics
                    },
                    0.0,
                    REST_LENGTH,
                )
                .force)
            / (2.0 * step);
        let rate_difference = (material
            .response(
                AxialKinematics {
                    extension_rate: kinematics.extension_rate + step,
                    ..kinematics
                },
                0.0,
                REST_LENGTH,
            )
            .force
            - material
                .response(
                    AxialKinematics {
                        extension_rate: kinematics.extension_rate - step,
                        ..kinematics
                    },
                    0.0,
                    REST_LENGTH,
                )
                .force)
            / (2.0 * step);

        assert!((response.length_tangent - position_difference).abs() < 1.0e-3);
        assert!((response.rate_tangent - rate_difference).abs() < 1.0e-5);
    }

    #[test]
    fn quadratic_kelvin_voigt_stored_energy_matches_regularized_elastic_force() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let extension = 0.002;
        let step = 1.0e-8;
        let energy_difference = (material.stored_energy(extension + step, 0.0, REST_LENGTH)
            - material.stored_energy(extension - step, 0.0, REST_LENGTH))
            / (2.0 * step);
        let force = material
            .response(
                AxialKinematics {
                    extension,
                    extension_rate: 0.0,
                },
                0.0,
                REST_LENGTH,
            )
            .force;

        assert!((force - energy_difference).abs() < 1.0e-5);
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

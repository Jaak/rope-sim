mod hooke;
mod kelvin_voigt;
mod quadratic_kelvin_voigt;
mod standard_linear_solid;

use crate::config::{RopeModelKind, SimulationConfig};

use hooke::Hooke;
use kelvin_voigt::KelvinVoigt;
use quadratic_kelvin_voigt::QuadraticKelvinVoigt;
use standard_linear_solid::StandardLinearSolid;
pub(crate) use standard_linear_solid::{
    StandardLinearSolidBackwardEulerTrial, StandardLinearSolidState,
    StandardLinearSolidStateDerivative,
};

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

/// Result of eliminating an element's internal state for one backward-Euler
/// trial. The caller may inspect this freely; the returned state is committed
/// only if the enclosing integration stage succeeds.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BackwardEulerMaterialTrial {
    pub sls_state: Option<StandardLinearSolidState>,
    pub response: AxialResponse,
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
        sls_state: Option<StandardLinearSolidState>,
        rest_length: f64,
    ) -> AxialResponse {
        match self {
            Self::Hooke(material) => material.response(kinematics, rest_length),
            Self::KelvinVoigt(material) => material.response(kinematics, rest_length),
            Self::QuadraticKelvinVoigt(material) => material.response(kinematics, rest_length),
            Self::StandardLinearSolid(material) => material.response(
                kinematics,
                sls_state.expect("SLS requires per-element state"),
                rest_length,
            ),
        }
    }

    pub fn initial_sls_state(self, element_count: usize) -> Option<Vec<StandardLinearSolidState>> {
        match self {
            Self::StandardLinearSolid(_) => {
                Some(vec![StandardLinearSolidState::default(); element_count])
            }
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => None,
        }
    }

    pub fn backward_euler_trial(
        self,
        kinematics: AxialKinematics,
        committed_state: Option<StandardLinearSolidState>,
        rest_length: f64,
        dt: f64,
    ) -> BackwardEulerMaterialTrial {
        match self {
            Self::StandardLinearSolid(material) => {
                let trial = material.backward_euler_trial(
                    kinematics,
                    committed_state.expect("SLS requires committed per-element state"),
                    rest_length,
                    dt,
                );
                BackwardEulerMaterialTrial {
                    sls_state: Some(trial.state),
                    response: trial.response,
                }
            }
            Self::Hooke(_) | Self::KelvinVoigt(_) | Self::QuadraticKelvinVoigt(_) => {
                BackwardEulerMaterialTrial {
                    sls_state: None,
                    response: self.response(kinematics, None, rest_length),
                }
            }
        }
    }

    pub fn sls_backward_euler_trial(
        self,
        kinematics: AxialKinematics,
        committed_state: StandardLinearSolidState,
        rest_length: f64,
        dt: f64,
    ) -> StandardLinearSolidBackwardEulerTrial {
        let Self::StandardLinearSolid(material) = self else {
            panic!("SLS backward-Euler trial requires the standard-linear-solid material");
        };
        material.backward_euler_trial(kinematics, committed_state, rest_length, dt)
    }

    pub fn sls_state_derivative(
        self,
        extension_rate: f64,
        sls_state: StandardLinearSolidState,
        rest_length: f64,
    ) -> StandardLinearSolidStateDerivative {
        let Self::StandardLinearSolid(material) = self else {
            panic!("SLS state derivative requires the standard-linear-solid material");
        };
        material.state_derivative(extension_rate, sls_state, rest_length)
    }

    pub fn stored_energy(
        self,
        extension: f64,
        sls_state: Option<StandardLinearSolidState>,
        rest_length: f64,
    ) -> f64 {
        match self {
            Self::Hooke(material) => material.stored_energy(extension, rest_length),
            Self::KelvinVoigt(material) => material.stored_energy(extension, rest_length),
            Self::QuadraticKelvinVoigt(material) => material.stored_energy(extension, rest_length),
            Self::StandardLinearSolid(material) => material.stored_energy(
                extension,
                sls_state.expect("SLS requires per-element state"),
                rest_length,
            ),
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

    fn response(
        material: AxialMaterial,
        kinematics: AxialKinematics,
        rest_length: f64,
    ) -> AxialResponse {
        let state = material.initial_sls_state(1);
        material.response(
            kinematics,
            state.as_ref().map(|states| states[0]),
            rest_length,
        )
    }

    fn stored_energy(material: AxialMaterial, extension: f64, rest_length: f64) -> f64 {
        let state = material.initial_sls_state(1);
        material.stored_energy(
            extension,
            state.as_ref().map(|states| states[0]),
            rest_length,
        )
    }

    #[test]
    fn hooke_response_exposes_force_and_tangent() {
        let material = AxialMaterial::Hooke(Hooke::new(30_000.0));
        let response = response(material, KINEMATICS, REST_LENGTH);

        assert_eq!(response.force, 3_000.0);
        assert_eq!(response.length_tangent, 15_000.0);
        assert_eq!(response.rate_tangent, 0.0);
    }

    #[test]
    fn kelvin_voigt_response_includes_viscous_force_and_tangent() {
        let material = AxialMaterial::KelvinVoigt(KelvinVoigt::new(30_000.0, 1_000.0));
        let response = response(material, KINEMATICS, REST_LENGTH);

        assert_eq!(response.force, 3_200.0);
        assert_eq!(response.length_tangent, 15_000.0);
        assert_eq!(response.rate_tangent, 500.0);
    }

    #[test]
    fn quadratic_kelvin_voigt_response_uses_strain_and_exact_tangents() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let response = response(material, KINEMATICS, REST_LENGTH);
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
        let slack = response(
            material,
            AxialKinematics {
                extension: -0.1,
                extension_rate: 100.0,
            },
            REST_LENGTH,
        );
        let rapidly_shortening = response(
            material,
            AxialKinematics {
                extension: 0.1,
                extension_rate: -100.0,
            },
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
        let coarse = stored_energy(material, 0.2, 2.0);
        let refined = 2.0 * stored_energy(material, 0.1, 1.0);

        assert!((coarse - refined).abs() < 1.0e-12);
        assert_eq!(stored_energy(material, -0.1, 1.0), 0.0);
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
        let evaluated = response(material, kinematics, REST_LENGTH);
        let step = 1.0e-8;

        let position_difference = (response(
            material,
            AxialKinematics {
                extension: kinematics.extension + step,
                ..kinematics
            },
            REST_LENGTH,
        )
        .force
            - response(
                material,
                AxialKinematics {
                    extension: kinematics.extension - step,
                    ..kinematics
                },
                REST_LENGTH,
            )
            .force)
            / (2.0 * step);
        let rate_difference = (response(
            material,
            AxialKinematics {
                extension_rate: kinematics.extension_rate + step,
                ..kinematics
            },
            REST_LENGTH,
        )
        .force
            - response(
                material,
                AxialKinematics {
                    extension_rate: kinematics.extension_rate - step,
                    ..kinematics
                },
                REST_LENGTH,
            )
            .force)
            / (2.0 * step);

        assert!((evaluated.length_tangent - position_difference).abs() < 1.0e-3);
        assert!((evaluated.rate_tangent - rate_difference).abs() < 1.0e-5);
    }

    #[test]
    fn quadratic_kelvin_voigt_stored_energy_matches_regularized_elastic_force() {
        let material = AxialMaterial::QuadraticKelvinVoigt(QuadraticKelvinVoigt::new(
            30_000.0, 100_000.0, 1_000.0,
        ));
        let extension = 0.002;
        let step = 1.0e-8;
        let energy_difference = (stored_energy(material, extension + step, REST_LENGTH)
            - stored_energy(material, extension - step, REST_LENGTH))
            / (2.0 * step);
        let force = response(
            material,
            AxialKinematics {
                extension,
                extension_rate: 0.0,
            },
            REST_LENGTH,
        )
        .force;

        assert!((force - energy_difference).abs() < 1.0e-5);
    }

    #[test]
    fn sls_backward_euler_trial_returns_state_and_eliminated_response() {
        let material = AxialMaterial::StandardLinearSolid(StandardLinearSolid::new(
            30_000.0, 15_000.0, 1_000.0,
        ));
        let dt = 0.01;
        let committed = StandardLinearSolidState::new(100.0);
        let trial = material.backward_euler_trial(KINEMATICS, Some(committed), REST_LENGTH, dt);

        let denominator = 1.0 + dt * 15_000.0 / 1_000.0;
        let expected_state = (100.0 + dt * 15_000.0 * 0.4 / REST_LENGTH) / denominator;
        let trial_state = trial.sls_state.expect("SLS returned a stateless trial");
        assert!((trial_state.transient_force() - expected_state).abs() < 1.0e-12);
        assert!((trial.response.force - (3_000.0 + expected_state)).abs() < 1.0e-12);
        assert!((trial.response.rate_tangent - 75.0 / denominator).abs() < 1.0e-12);
    }

    #[test]
    fn stateless_backward_euler_trial_has_no_internal_state() {
        let material = AxialMaterial::Hooke(Hooke::new(30_000.0));
        let trial = material.backward_euler_trial(KINEMATICS, None, REST_LENGTH, 0.01);

        assert!(trial.sls_state.is_none());
        assert_eq!(trial.response.force, 3_000.0);
    }
}

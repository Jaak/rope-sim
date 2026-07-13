use super::{AxialKinematics, AxialResponse};

const INACTIVE_RESPONSE: AxialResponse = AxialResponse {
    force: 0.0,
    length_tangent: 0.0,
    rate_tangent: 0.0,
};
// A narrow C1 transition prevents Newton active-set chatter without permitting
// compressive force. Outside this 1% strain band the requested elastic law is
// recovered exactly; shortening-rate saturation is regularized separately.
const ACTIVATION_STRAIN: f64 = 1.0e-2;
const DAMPING_ACTIVATION_FORCE_FRACTION: f64 = 1.0e-4;
const UNILATERAL_SMOOTHNESS: f64 = 5.0e-2;

/// Tension-only Kelvin-Voigt material with quadratic elastic stiffening.
///
/// Parameters are defined against strain rather than discrete extension, so a
/// rope retains the same constitutive response when its segment count changes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QuadraticKelvinVoigt {
    linear_rigidity: f64,
    quadratic_rigidity: f64,
    viscosity: f64,
}

impl QuadraticKelvinVoigt {
    pub(super) fn new(linear_rigidity: f64, quadratic_rigidity: f64, viscosity: f64) -> Self {
        Self {
            linear_rigidity,
            quadratic_rigidity,
            viscosity,
        }
    }

    pub(super) fn response(self, kinematics: AxialKinematics, rest_length: f64) -> AxialResponse {
        let strain = kinematics.extension / rest_length;
        let activation = tensile_activation(strain);
        let strain_rate = kinematics.extension_rate / rest_length;
        let elastic_force = self.quadratic_rigidity * activation.strain * activation.strain
            + self.linear_rigidity * activation.strain;
        let damping_force_scale = DAMPING_ACTIVATION_FORCE_FRACTION * self.linear_rigidity;
        let damping_activation = elastic_force / (elastic_force + damping_force_scale);
        let damping_activation_tangent =
            damping_force_scale / (elastic_force + damping_force_scale).powi(2);
        let raw_force = elastic_force + self.viscosity * damping_activation * strain_rate;
        let force = smooth_unilateral_force(raw_force, elastic_force);
        if force.value <= 0.0 {
            return INACTIVE_RESPONSE;
        }

        let elastic_strain_tangent = (2.0 * self.quadratic_rigidity * activation.strain
            + self.linear_rigidity)
            * activation.strain_tangent;
        let raw_strain_tangent = elastic_strain_tangent
            * (1.0 + self.viscosity * damping_activation_tangent * strain_rate);
        AxialResponse {
            force: force.value,
            length_tangent: (force.raw_tangent * raw_strain_tangent
                + force.elastic_tangent * elastic_strain_tangent)
                / rest_length,
            rate_tangent: force.raw_tangent * self.viscosity * damping_activation / rest_length,
        }
    }

    pub(super) fn stored_energy(self, extension: f64, rest_length: f64) -> f64 {
        let strain = extension / rest_length;
        if strain <= 0.0 {
            return 0.0;
        }

        let band_energy = |ratio: f64| {
            let linear_integral = 2.0 * ratio.powi(3) / 3.0 - ratio.powi(4) / 4.0;
            let quadratic_integral =
                4.0 * ratio.powi(5) / 5.0 - 2.0 * ratio.powi(6) / 3.0 + ratio.powi(7) / 7.0;
            rest_length
                * (self.linear_rigidity * ACTIVATION_STRAIN.powi(2) * linear_integral
                    + self.quadratic_rigidity * ACTIVATION_STRAIN.powi(3) * quadratic_integral)
        };

        if strain < ACTIVATION_STRAIN {
            band_energy(strain / ACTIVATION_STRAIN)
        } else {
            band_energy(1.0)
                + rest_length
                    * (0.5 * self.linear_rigidity * (strain * strain - ACTIVATION_STRAIN.powi(2))
                        + self.quadratic_rigidity * (strain.powi(3) - ACTIVATION_STRAIN.powi(3))
                            / 3.0)
        }
    }

    /// Conservative elastic tangent in strain coordinates. The linear term is
    /// retained while slack so explicit stepping remains prepared for a
    /// slack-to-taut transition within the next step.
    pub(super) fn tangent_rigidity(self, kinematics: AxialKinematics, rest_length: f64) -> f64 {
        let strain = kinematics.extension / rest_length;
        let transition_bound = if strain < ACTIVATION_STRAIN {
            (4.0 / 3.0) * (self.linear_rigidity + 2.0 * self.quadratic_rigidity * ACTIVATION_STRAIN)
        } else {
            0.0
        };
        let current_tangent = self.response(kinematics, rest_length).length_tangent * rest_length;
        transition_bound.max(current_tangent).max(0.0)
    }

    pub(super) fn explicit_viscosity_bound(self) -> f64 {
        // The C1 unilateral-force ramp has a maximum slope of 4/3.
        (4.0 / 3.0) * self.viscosity
    }

    pub(super) const fn activation_strain(self) -> f64 {
        ACTIVATION_STRAIN
    }
}

#[derive(Clone, Copy)]
struct TensileActivation {
    strain: f64,
    strain_tangent: f64,
}

fn tensile_activation(strain: f64) -> TensileActivation {
    if strain <= 0.0 {
        return TensileActivation {
            strain: 0.0,
            strain_tangent: 0.0,
        };
    }
    if strain >= ACTIVATION_STRAIN {
        return TensileActivation {
            strain,
            strain_tangent: 1.0,
        };
    }

    let ratio = strain / ACTIVATION_STRAIN;
    TensileActivation {
        strain: ACTIVATION_STRAIN * (2.0 * ratio * ratio - ratio * ratio * ratio),
        strain_tangent: 4.0 * ratio - 3.0 * ratio * ratio,
    }
}

#[derive(Clone, Copy)]
struct SmoothUnilateralForce {
    value: f64,
    raw_tangent: f64,
    elastic_tangent: f64,
}

fn smooth_unilateral_force(raw_force: f64, elastic_force: f64) -> SmoothUnilateralForce {
    if elastic_force <= 0.0 {
        return SmoothUnilateralForce {
            value: 0.0,
            raw_tangent: 0.0,
            elastic_tangent: 0.0,
        };
    }

    // A normalized smooth positive part avoids a second derivative kink when
    // the dashpot would otherwise put the element into compression. It is
    // exactly the elastic force at zero strain rate and approaches zero under
    // rapid shortening without ever producing compression.
    let ratio = raw_force / elastic_force;
    let radius = ratio.hypot(UNILATERAL_SMOOTHNESS);
    let normalization = 1.0 + 1.0_f64.hypot(UNILATERAL_SMOOTHNESS);
    let value_ratio = (ratio + radius) / normalization;
    let ratio_tangent = (1.0 + ratio / radius) / normalization;
    SmoothUnilateralForce {
        value: elastic_force * value_ratio,
        raw_tangent: ratio_tangent,
        elastic_tangent: value_ratio - ratio * ratio_tangent,
    }
}

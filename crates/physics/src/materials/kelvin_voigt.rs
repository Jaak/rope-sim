use super::{AxialKinematics, AxialResponse};

#[derive(Clone, Copy, Debug)]
pub(crate) struct KelvinVoigt {
    rigidity: f64,
    viscosity: f64,
}

impl KelvinVoigt {
    pub(super) fn new(rigidity: f64, viscosity: f64) -> Self {
        Self {
            rigidity,
            viscosity,
        }
    }

    pub(super) fn response(self, kinematics: AxialKinematics, rest_length: f64) -> AxialResponse {
        let stiffness = self.rigidity / rest_length;
        let damping = self.viscosity / rest_length;
        AxialResponse {
            force: stiffness * kinematics.extension + damping * kinematics.extension_rate,
            length_tangent: stiffness,
            rate_tangent: damping,
        }
    }

    pub(super) fn stored_energy(self, extension: f64, rest_length: f64) -> f64 {
        0.5 * self.rigidity / rest_length * extension * extension
    }

    pub(super) fn rigidity(self) -> f64 {
        self.rigidity
    }

    pub(super) fn viscosity(self) -> f64 {
        self.viscosity
    }
}

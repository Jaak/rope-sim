use super::{AxialKinematics, AxialResponse};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Hooke {
    rigidity: f64,
}

impl Hooke {
    pub(super) fn new(rigidity: f64) -> Self {
        Self { rigidity }
    }

    pub(super) fn response(self, kinematics: AxialKinematics, rest_length: f64) -> AxialResponse {
        let stiffness = self.rigidity / rest_length;
        AxialResponse {
            force: stiffness * kinematics.extension,
            length_tangent: stiffness,
            rate_tangent: 0.0,
        }
    }

    pub(super) fn stored_energy(self, extension: f64, rest_length: f64) -> f64 {
        0.5 * self.rigidity / rest_length * extension * extension
    }

    pub(super) fn rigidity(self) -> f64 {
        self.rigidity
    }
}

use super::{AxialKinematics, AxialResponse};

#[derive(Clone, Copy, Debug)]
pub(crate) struct StandardLinearSolid {
    relaxed_rigidity: f64,
    transient_rigidity: f64,
    viscosity: f64,
}

impl StandardLinearSolid {
    pub(super) fn new(relaxed_rigidity: f64, transient_rigidity: f64, viscosity: f64) -> Self {
        Self {
            relaxed_rigidity,
            transient_rigidity,
            viscosity,
        }
    }

    pub(super) fn response(
        self,
        kinematics: AxialKinematics,
        material_state: f64,
        rest_length: f64,
    ) -> AxialResponse {
        let stiffness = self.relaxed_rigidity / rest_length;
        AxialResponse {
            force: stiffness * kinematics.extension + material_state,
            length_tangent: stiffness,
            rate_tangent: 0.0,
        }
    }

    pub(super) fn backward_euler_response(
        self,
        kinematics: AxialKinematics,
        material_state: f64,
        rest_length: f64,
        dt: f64,
    ) -> AxialResponse {
        let mut response = self.response(kinematics, material_state, rest_length);
        response.rate_tangent =
            dt * self.transient_rigidity / (rest_length * self.backward_euler_denominator(dt));
        response
    }

    pub(super) fn state_derivative(
        self,
        extension_rate: f64,
        material_state: f64,
        rest_length: f64,
    ) -> f64 {
        self.transient_rigidity * extension_rate / rest_length
            - self.relaxation_rate() * material_state
    }

    pub(super) fn backward_euler_state(
        self,
        extension_rate: f64,
        initial_material_state: f64,
        rest_length: f64,
        dt: f64,
    ) -> f64 {
        (initial_material_state + dt * self.transient_rigidity * extension_rate / rest_length)
            / self.backward_euler_denominator(dt)
    }

    pub(super) fn stored_energy(
        self,
        extension: f64,
        material_state: f64,
        rest_length: f64,
    ) -> f64 {
        0.5 * self.relaxed_rigidity / rest_length * extension * extension
            + 0.5 * material_state * material_state * rest_length / self.transient_rigidity
    }

    pub(super) fn instantaneous_rigidity(self) -> f64 {
        self.relaxed_rigidity + self.transient_rigidity
    }

    pub(super) fn transient_rigidity(self) -> f64 {
        self.transient_rigidity
    }

    pub(super) fn relaxation_rate(self) -> f64 {
        self.transient_rigidity / self.viscosity
    }

    fn backward_euler_denominator(self, dt: f64) -> f64 {
        1.0 + dt * self.relaxation_rate()
    }
}

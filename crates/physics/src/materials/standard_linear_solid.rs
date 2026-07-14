use super::{AxialKinematics, AxialResponse};

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StandardLinearSolidState {
    transient_force: f64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StandardLinearSolidStateDerivative {
    transient_force_rate: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct StandardLinearSolidBackwardEulerTrial {
    pub state: StandardLinearSolidState,
    pub response: AxialResponse,
}

impl StandardLinearSolidState {
    pub(crate) fn new(transient_force: f64) -> Self {
        Self { transient_force }
    }

    pub(crate) fn transient_force(self) -> f64 {
        self.transient_force
    }

    pub(crate) fn is_finite(self) -> bool {
        self.transient_force.is_finite()
    }

    pub(crate) fn add_scaled(
        &mut self,
        derivative: StandardLinearSolidStateDerivative,
        scale: f64,
    ) {
        self.transient_force += scale * derivative.transient_force_rate;
    }
}

impl StandardLinearSolidStateDerivative {
    pub(super) fn new(transient_force_rate: f64) -> Self {
        Self {
            transient_force_rate,
        }
    }
}

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
        state: StandardLinearSolidState,
        rest_length: f64,
    ) -> AxialResponse {
        let stiffness = self.relaxed_rigidity / rest_length;
        AxialResponse {
            force: stiffness * kinematics.extension + state.transient_force,
            length_tangent: stiffness,
            rate_tangent: 0.0,
        }
    }

    pub(super) fn backward_euler_trial(
        self,
        kinematics: AxialKinematics,
        committed_state: StandardLinearSolidState,
        rest_length: f64,
        dt: f64,
    ) -> StandardLinearSolidBackwardEulerTrial {
        let state = StandardLinearSolidState::new(
            (committed_state.transient_force
                + dt * self.transient_rigidity * kinematics.extension_rate / rest_length)
                / self.backward_euler_denominator(dt),
        );
        let mut response = self.response(kinematics, state, rest_length);
        response.rate_tangent =
            dt * self.transient_rigidity / (rest_length * self.backward_euler_denominator(dt));
        StandardLinearSolidBackwardEulerTrial { state, response }
    }

    pub(super) fn state_derivative(
        self,
        extension_rate: f64,
        state: StandardLinearSolidState,
        rest_length: f64,
    ) -> StandardLinearSolidStateDerivative {
        StandardLinearSolidStateDerivative::new(
            self.transient_rigidity * extension_rate / rest_length
                - self.relaxation_rate() * state.transient_force,
        )
    }

    pub(super) fn stored_energy(
        self,
        extension: f64,
        state: StandardLinearSolidState,
        rest_length: f64,
    ) -> f64 {
        0.5 * self.relaxed_rigidity / rest_length * extension * extension
            + 0.5 * state.transient_force * state.transient_force * rest_length
                / self.transient_rigidity
    }

    pub(super) fn instantaneous_rigidity(self) -> f64 {
        self.relaxed_rigidity + self.transient_rigidity
    }

    pub(super) fn relaxation_rate(self) -> f64 {
        self.transient_rigidity / self.viscosity
    }

    fn backward_euler_denominator(self, dt: f64) -> f64 {
        1.0 + dt * self.relaxation_rate()
    }
}

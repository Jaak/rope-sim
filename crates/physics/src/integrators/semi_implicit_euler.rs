use crate::materials::StandardLinearSolidStateDerivative;
use crate::math::Vec2;
use crate::state::State;

use super::{DynamicalSystem, StepError, TimeIntegrator, validate_timestep};

pub(super) struct SemiImplicitEuler {
    accelerations: Vec<Vec2>,
    sls_derivatives: Vec<StandardLinearSolidStateDerivative>,
}

impl SemiImplicitEuler {
    pub fn new(node_count: usize) -> Self {
        Self {
            accelerations: vec![Vec2::ZERO; node_count],
            sls_derivatives: Vec::with_capacity(node_count.saturating_sub(1)),
        }
    }
}

impl TimeIntegrator for SemiImplicitEuler {
    fn step(
        &mut self,
        system: &dyn DynamicalSystem,
        state: &mut State,
        dt: f64,
    ) -> Result<(), StepError> {
        validate_timestep(dt)?;

        system.enforce_kinematics(state, 0.0);
        self.accelerations.resize(state.node_count(), Vec2::ZERO);
        self.accelerations.fill(Vec2::ZERO);
        system.accelerations(state, &mut self.accelerations);
        system.sls_state_derivatives(state, &mut self.sls_derivatives);
        if let Some(states) = &mut state.sls_state {
            for (state, derivative) in states.iter_mut().zip(&self.sls_derivatives) {
                state.add_scaled(*derivative, dt);
            }
        }

        for index in 0..state.node_count() {
            if !system.is_dynamic_node(index) {
                continue;
            }

            state.velocities[index] += self.accelerations[index] * dt;
            state.positions[index] += state.velocities[index] * dt;
        }

        system.enforce_kinematics(state, dt);
        if !state.is_finite() {
            return Err(StepError::NonFiniteState);
        }

        Ok(())
    }

    fn recommended_substeps(
        &self,
        system: &dyn DynamicalSystem,
        state: &State,
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        let maximum_dt = system.explicit_stable_timestep(state);
        Ok((outer_dt / maximum_dt).ceil().max(1.0) as usize)
    }
}

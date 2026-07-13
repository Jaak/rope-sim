use crate::math::Vec2;
use crate::state::State;

use super::{DynamicalSystem, StepError, TimeIntegrator, validate_timestep};

pub(super) struct SemiImplicitEuler {
    accelerations: Vec<Vec2>,
    material_derivatives: Vec<f64>,
}

impl SemiImplicitEuler {
    pub fn new(node_count: usize) -> Self {
        Self {
            accelerations: vec![Vec2::ZERO; node_count],
            material_derivatives: vec![0.0; node_count.saturating_sub(1)],
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
        self.material_derivatives
            .resize(state.material_state.len(), 0.0);
        system.material_state_derivatives(state, &mut self.material_derivatives);

        for (value, derivative) in state
            .material_state
            .iter_mut()
            .zip(&self.material_derivatives)
        {
            *value += derivative * dt;
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
        outer_dt: f64,
    ) -> Result<usize, StepError> {
        validate_timestep(outer_dt)?;
        let maximum_dt = system.explicit_stable_timestep();
        Ok((outer_dt / maximum_dt).ceil().max(1.0) as usize)
    }
}

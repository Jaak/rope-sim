use crate::materials::StandardLinearSolidState;
use crate::math::Vec2;

#[derive(Clone)]
pub(crate) struct State {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    /// One transient-force state per element for SLS, absent for stateless
    /// constitutive models.
    pub sls_state: Option<Vec<StandardLinearSolidState>>,
}

impl State {
    pub fn new(positions: Vec<Vec2>) -> Self {
        let node_count = positions.len();
        Self {
            positions,
            velocities: vec![Vec2::ZERO; node_count],
            sls_state: None,
        }
    }

    pub fn node_count(&self) -> usize {
        self.positions.len()
    }

    pub fn is_finite(&self) -> bool {
        if let Some(states) = &self.sls_state {
            for state in states {
                if !state.is_finite() {
                    return false;
                }
            }
        }

        for position in &self.positions {
            if !position.is_finite() {
                return false;
            }
        }

        for velocity in &self.velocities {
            if !velocity.is_finite() {
                return false;
            }
        }

        true
    }
}

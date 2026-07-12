use crate::math::Vec2;

#[derive(Clone)]
pub(crate) struct State {
    pub positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    /// Per-element constitutive state. Currently this stores the Maxwell-branch
    /// axial force for the standard linear solid model.
    pub material_state: Vec<f64>,
}

impl State {
    pub fn new(positions: Vec<Vec2>) -> Self {
        let node_count = positions.len();
        Self {
            positions,
            velocities: vec![Vec2::ZERO; node_count],
            material_state: vec![0.0; node_count.saturating_sub(1)],
        }
    }

    pub fn node_count(&self) -> usize {
        self.positions.len()
    }

    pub fn is_finite(&self) -> bool {
        self.positions
            .iter()
            .chain(&self.velocities)
            .all(|value| value.is_finite())
            && self.material_state.iter().all(|value| value.is_finite())
    }
}

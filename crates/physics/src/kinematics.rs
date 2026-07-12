use crate::math::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KinematicTarget {
    pub position: Vec2,
    pub velocity: Vec2,
}

impl KinematicTarget {
    pub const fn new(position: Vec2, velocity: Vec2) -> Self {
        Self { position, velocity }
    }
}

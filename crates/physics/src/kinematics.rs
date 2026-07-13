use crate::math::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct KinematicTarget {
    pub position: Vec2,
    pub velocity: Vec2,
}

impl KinematicTarget {
    pub const fn new(position: Vec2, velocity: Vec2) -> Self {
        Self { position, velocity }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct KinematicMotion {
    start: KinematicTarget,
    end: KinematicTarget,
    duration: f64,
    elapsed: f64,
}

impl KinematicMotion {
    pub(crate) fn new(start: KinematicTarget, end: KinematicTarget, duration: f64) -> Self {
        Self {
            start,
            end,
            duration,
            elapsed: 0.0,
        }
    }

    pub(crate) fn target_after(self, time: f64) -> KinematicTarget {
        let elapsed = (self.elapsed + time).clamp(0.0, self.duration);
        let u = elapsed / self.duration;
        let displacement = self.end.position - self.start.position;
        let position = self.start.position + displacement * u;
        let finished = elapsed >= self.duration;
        let velocity = if finished {
            self.end.velocity
        } else {
            displacement / self.duration
        };
        KinematicTarget::new(position, velocity)
    }

    pub(crate) fn advance(&mut self, dt: f64) -> (KinematicTarget, bool) {
        self.elapsed = (self.elapsed + dt).min(self.duration);
        (self.target_after(0.0), self.elapsed >= self.duration)
    }

    pub(crate) fn maximum_speed(self) -> f64 {
        ((self.end.position - self.start.position) / self.duration)
            .length()
            .max(self.end.velocity.length())
    }
}

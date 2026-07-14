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
    left_continuous_final_stage: bool,
}

impl KinematicMotion {
    pub(crate) fn new(start: KinematicTarget, end: KinematicTarget, duration: f64) -> Self {
        Self {
            start,
            end,
            duration,
            elapsed: 0.0,
            left_continuous_final_stage: false,
        }
    }

    pub(crate) fn new_left_continuous(
        start: KinematicTarget,
        end: KinematicTarget,
        duration: f64,
    ) -> Self {
        Self {
            left_continuous_final_stage: true,
            ..Self::new(start, end, duration)
        }
    }

    pub(crate) fn target_after(self, time: f64) -> KinematicTarget {
        let elapsed = (self.elapsed + time).clamp(0.0, self.duration);
        let u = elapsed / self.duration;
        let displacement = self.end.position - self.start.position;
        let position = self.start.position + displacement * u;
        // While the trajectory is active, its final stage samples the
        // left-hand velocity of this interval. `advance` commits the endpoint
        // velocity for the following interval after the step succeeds.
        let velocity = if self.elapsed >= self.duration
            || (!self.left_continuous_final_stage && elapsed >= self.duration)
        {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_motion_uses_left_hand_velocity_at_its_final_stage() {
        let start = KinematicTarget::new(Vec2::ZERO, Vec2::ZERO);
        let end = KinematicTarget::new(Vec2::new(2.0, 0.0), Vec2::new(7.0, 0.0));
        let mut motion = KinematicMotion::new_left_continuous(start, end, 1.0);

        assert_eq!(motion.target_after(1.0).velocity, Vec2::new(2.0, 0.0));
        let (committed, finished) = motion.advance(1.0);
        assert!(finished);
        assert_eq!(committed.velocity, end.velocity);
    }
}

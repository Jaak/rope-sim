use crate::config::SimulationConfig;
use crate::kinematics::KinematicTarget;
use crate::materials::AxialMaterial;
use crate::math::Vec2;
use crate::state::State;

const CONSTRAINT_SWEEPS: usize = 12;
const MINIMUM_MANIPULATION_DAMPING_RATE: f64 = 0.5;

/// Position-based rope dynamics used while the payload is manipulated.
///
/// This is an interaction aid, not a selectable constitutive time integrator.
pub(crate) struct XpbdRopeRelaxer {
    multipliers: Vec<f64>,
}

impl XpbdRopeRelaxer {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            multipliers: vec![0.0; node_count.saturating_sub(1)],
        }
    }

    pub(crate) fn step_held(
        &mut self,
        config: &SimulationConfig,
        state: &mut State,
        masses: &[f64],
        rest_length: f64,
        target: KinematicTarget,
        dt: f64,
    ) {
        self.resize(state.node_count());
        let payload = state.node_count() - 1;
        let constraint_sweeps = config.segment_count.clamp(CONSTRAINT_SWEEPS, 64);
        let damping_rate = config
            .air_damping_rate
            .max(MINIMUM_MANIPULATION_DAMPING_RATE);
        let damping = (-damping_rate * dt).exp();
        for node in 1..payload {
            state.velocities[node] = (state.velocities[node] + config.gravity * dt) * damping;
            state.positions[node] += state.velocities[node] * dt;
        }
        state.positions[0] = config.anchor;
        state.positions[payload] = target.position;

        self.multipliers.fill(0.0);
        self.project_constraints(
            config,
            state,
            masses,
            rest_length,
            dt,
            Some(target.position),
            constraint_sweeps,
        );

        // Position corrections include cursor-driven reshaping and must not be
        // interpreted as momentum. Project the independently integrated free
        // velocities instead, so the held constraints can support gravity.
        state.velocities[0] = Vec2::ZERO;
        state.velocities[payload] = target.velocity;
        project_velocities(state, masses, target.velocity, constraint_sweeps);
        advance_material_state(config, state, rest_length, dt);
    }

    #[allow(clippy::too_many_arguments)]
    fn project_constraints(
        &mut self,
        config: &SimulationConfig,
        state: &mut State,
        masses: &[f64],
        rest_length: f64,
        dt: f64,
        held_payload: Option<Vec2>,
        constraint_sweeps: usize,
    ) {
        let compliance = rest_length / config.axial_rigidity;
        let scaled_compliance = compliance / (dt * dt);
        // Even a tension-only material retains its geometric material length
        // while held; slack is represented by folds rather than zero-length
        // elements. Its physical constitutive response remains unilateral.
        for sweep in 0..constraint_sweeps {
            if sweep % 2 == 0 {
                for element in 0..config.segment_count {
                    project_element(
                        state,
                        masses,
                        &mut self.multipliers,
                        element,
                        rest_length,
                        scaled_compliance,
                        held_payload.is_some(),
                    );
                }
            } else {
                for element in (0..config.segment_count).rev() {
                    project_element(
                        state,
                        masses,
                        &mut self.multipliers,
                        element,
                        rest_length,
                        scaled_compliance,
                        held_payload.is_some(),
                    );
                }
            }
            state.positions[0] = config.anchor;
            if let Some(payload_position) = held_payload {
                let payload = state.node_count() - 1;
                state.positions[payload] = payload_position;
            }
        }
    }

    fn resize(&mut self, node_count: usize) {
        self.multipliers.resize(node_count.saturating_sub(1), 0.0);
    }
}

fn advance_material_state(config: &SimulationConfig, state: &mut State, rest_length: f64, dt: f64) {
    let material = AxialMaterial::from_config(*config);
    for element in 0..config.segment_count {
        let delta = state.positions[element + 1] - state.positions[element];
        let length = delta.length();
        let extension_rate = if length > f64::EPSILON {
            (delta / length).dot(state.velocities[element + 1] - state.velocities[element])
        } else {
            0.0
        };
        state.material_state[element] = material.backward_euler_state(
            extension_rate,
            state.material_state[element],
            rest_length,
            dt,
        );
    }
}

fn project_velocities(
    state: &mut State,
    masses: &[f64],
    payload_velocity: Vec2,
    constraint_sweeps: usize,
) {
    let payload = state.node_count() - 1;
    for sweep in 0..constraint_sweeps {
        if sweep.is_multiple_of(2) {
            for element in 0..payload {
                project_element_velocity(state, masses, element, payload);
            }
        } else {
            for element in (0..payload).rev() {
                project_element_velocity(state, masses, element, payload);
            }
        }
        state.velocities[0] = Vec2::ZERO;
        state.velocities[payload] = payload_velocity;
    }
}

fn project_element_velocity(state: &mut State, masses: &[f64], element: usize, payload: usize) {
    let left = element;
    let right = element + 1;
    let delta = state.positions[right] - state.positions[left];
    let length = delta.length();
    if length <= f64::EPSILON {
        return;
    }
    let direction = delta / length;
    let left_inverse_mass = inverse_mass(masses, left, payload, true);
    let right_inverse_mass = inverse_mass(masses, right, payload, true);
    let denominator = left_inverse_mass + right_inverse_mass;
    if denominator <= 0.0 {
        return;
    }
    let relative_speed = direction.dot(state.velocities[right] - state.velocities[left]);
    let impulse = -relative_speed / denominator;
    state.velocities[left] -= direction * (left_inverse_mass * impulse);
    state.velocities[right] += direction * (right_inverse_mass * impulse);
}

#[allow(clippy::too_many_arguments)]
fn project_element(
    state: &mut State,
    masses: &[f64],
    multipliers: &mut [f64],
    element: usize,
    rest_length: f64,
    scaled_compliance: f64,
    payload_is_held: bool,
) {
    let left = element;
    let right = element + 1;
    let delta = state.positions[right] - state.positions[left];
    let length = delta.length();
    let extension = length - rest_length;
    let direction = if length > f64::EPSILON {
        delta / length
    } else {
        Vec2::new(if element.is_multiple_of(2) { 1.0 } else { -1.0 }, 0.0)
    };
    let payload = state.node_count() - 1;
    let left_inverse_mass = inverse_mass(masses, left, payload, payload_is_held);
    let right_inverse_mass = inverse_mass(masses, right, payload, payload_is_held);
    let denominator = left_inverse_mass + right_inverse_mass + scaled_compliance;
    if denominator <= 0.0 {
        return;
    }

    let old_multiplier = multipliers[element];
    let multiplier_change = (-extension - scaled_compliance * old_multiplier) / denominator;
    let new_multiplier = old_multiplier + multiplier_change;
    let applied_change = new_multiplier - old_multiplier;
    multipliers[element] = new_multiplier;

    state.positions[left] -= direction * (left_inverse_mass * applied_change);
    state.positions[right] += direction * (right_inverse_mass * applied_change);
}

fn inverse_mass(masses: &[f64], node: usize, payload: usize, payload_is_held: bool) -> f64 {
    if node == 0 || (payload_is_held && node == payload) {
        0.0
    } else {
        1.0 / masses[node]
    }
}

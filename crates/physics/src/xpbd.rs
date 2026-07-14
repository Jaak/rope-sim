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
    bending_multipliers: Vec<f64>,
    velocity_projection: VelocityProjection,
}

impl XpbdRopeRelaxer {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            multipliers: vec![0.0; node_count.saturating_sub(1)],
            bending_multipliers: vec![0.0; node_count.saturating_sub(2)],
            velocity_projection: VelocityProjection::new(node_count.saturating_sub(1)),
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
        self.bending_multipliers.fill(0.0);
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
        self.velocity_projection
            .project(state, masses, target.velocity);
        damp_bending_velocities(config, state, masses, rest_length, target.velocity, dt);
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
                if config.bending_rigidity > 0.0 {
                    for center in 1..state.node_count().saturating_sub(1) {
                        project_bending_vertex(
                            state,
                            masses,
                            &mut self.bending_multipliers,
                            center,
                            rest_length,
                            config.bending_rigidity,
                            dt,
                            held_payload.is_some(),
                        );
                    }
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
                if config.bending_rigidity > 0.0 {
                    for center in (1..state.node_count().saturating_sub(1)).rev() {
                        project_bending_vertex(
                            state,
                            masses,
                            &mut self.bending_multipliers,
                            center,
                            rest_length,
                            config.bending_rigidity,
                            dt,
                            held_payload.is_some(),
                        );
                    }
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
        self.bending_multipliers
            .resize(node_count.saturating_sub(2), 0.0);
        self.velocity_projection
            .resize(node_count.saturating_sub(1));
    }
}

/// Mass-weighted projection onto the axial velocity constraints.
///
/// The constraint-space matrix is scalar tridiagonal because neighboring rope
/// elements share exactly one node. A tiny diagonal regularization selects a
/// stable impulse when a perfectly straight rope makes the endpoint-held
/// constraint system rank deficient; it has negligible effect on velocities.
struct VelocityProjection {
    directions: Vec<Vec2>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
    right_hand_side: Vec<f64>,
}

impl VelocityProjection {
    fn new(element_count: usize) -> Self {
        Self {
            directions: vec![Vec2::ZERO; element_count],
            diagonal: vec![0.0; element_count],
            upper: vec![0.0; element_count.saturating_sub(1)],
            right_hand_side: vec![0.0; element_count],
        }
    }

    fn resize(&mut self, element_count: usize) {
        self.directions.resize(element_count, Vec2::ZERO);
        self.diagonal.resize(element_count, 0.0);
        self.upper.resize(element_count.saturating_sub(1), 0.0);
        self.right_hand_side.resize(element_count, 0.0);
    }

    fn project(&mut self, state: &mut State, masses: &[f64], payload_velocity: Vec2) {
        let element_count = state.node_count().saturating_sub(1);
        if element_count == 0 {
            return;
        }
        self.resize(element_count);
        let payload = state.node_count() - 1;

        for element in 0..element_count {
            let delta = state.positions[element + 1] - state.positions[element];
            let length = delta.length();
            self.directions[element] = if length > f64::EPSILON {
                delta / length
            } else {
                Vec2::ZERO
            };
        }
        let mut maximum_diagonal = 0.0_f64;
        for element in 0..element_count {
            let left_inverse_mass = inverse_mass(masses, element, payload, true);
            let right_inverse_mass = inverse_mass(masses, element + 1, payload, true);
            let direction = self.directions[element];
            self.diagonal[element] =
                (left_inverse_mass + right_inverse_mass) * direction.length_squared();
            maximum_diagonal = maximum_diagonal.max(self.diagonal[element]);
            self.right_hand_side[element] =
                -direction.dot(state.velocities[element + 1] - state.velocities[element]);
            if element + 1 < element_count {
                let shared_inverse_mass = right_inverse_mass;
                self.upper[element] =
                    -shared_inverse_mass * direction.dot(self.directions[element + 1]);
            }
        }

        let regularization = maximum_diagonal.max(1.0) * 1.0e-12;
        for diagonal in &mut self.diagonal {
            *diagonal += regularization;
        }
        for index in 1..element_count {
            let factor = self.upper[index - 1] / self.diagonal[index - 1];
            self.diagonal[index] -= factor * self.upper[index - 1];
            self.right_hand_side[index] -= factor * self.right_hand_side[index - 1];
        }
        self.right_hand_side[element_count - 1] /= self.diagonal[element_count - 1];
        for index in (0..element_count - 1).rev() {
            self.right_hand_side[index] = (self.right_hand_side[index]
                - self.upper[index] * self.right_hand_side[index + 1])
                / self.diagonal[index];
        }

        for element in 0..element_count {
            let impulse = self.directions[element] * self.right_hand_side[element];
            let left_inverse_mass = inverse_mass(masses, element, payload, true);
            let right_inverse_mass = inverse_mass(masses, element + 1, payload, true);
            state.velocities[element] -= impulse * left_inverse_mass;
            state.velocities[element + 1] += impulse * right_inverse_mass;
        }
        state.velocities[0] = Vec2::ZERO;
        state.velocities[payload] = payload_velocity;
    }
}

#[allow(clippy::too_many_arguments)]
fn project_bending_vertex(
    state: &mut State,
    masses: &[f64],
    multipliers: &mut [f64],
    center: usize,
    rest_length: f64,
    bending_rigidity: f64,
    dt: f64,
    payload_is_held: bool,
) {
    let Some(geometry) = crate::dynamics::bending::geometry(state, center) else {
        return;
    };
    let nodes = [center - 1, center, center + 1];
    let payload = state.node_count() - 1;
    let inverse_masses = nodes.map(|node| inverse_mass(masses, node, payload, payload_is_held));
    let scaled_compliance = rest_length / (bending_rigidity * dt * dt);
    let denominator = scaled_compliance
        + inverse_masses
            .iter()
            .zip(geometry.gradients)
            .map(|(inverse_mass, gradient)| inverse_mass * gradient.length_squared())
            .sum::<f64>();
    if denominator <= 0.0 || !denominator.is_finite() {
        return;
    }

    let multiplier = &mut multipliers[center - 1];
    let change = (-geometry.angle - scaled_compliance * *multiplier) / denominator;
    *multiplier += change;
    for local in 0..3 {
        state.positions[nodes[local]] +=
            geometry.gradients[local] * (inverse_masses[local] * change);
    }
}

fn damp_bending_velocities(
    config: &SimulationConfig,
    state: &mut State,
    masses: &[f64],
    rest_length: f64,
    payload_velocity: Vec2,
    dt: f64,
) {
    if config.bending_viscosity <= 0.0 {
        return;
    }
    let payload = state.node_count() - 1;
    for reverse in [false, true] {
        if reverse {
            for center in (1..state.node_count().saturating_sub(1)).rev() {
                damp_bending_vertex(
                    state,
                    masses,
                    center,
                    rest_length,
                    config.bending_viscosity,
                    0.5 * dt,
                    payload,
                );
            }
        } else {
            for center in 1..state.node_count().saturating_sub(1) {
                damp_bending_vertex(
                    state,
                    masses,
                    center,
                    rest_length,
                    config.bending_viscosity,
                    0.5 * dt,
                    payload,
                );
            }
        }
        state.velocities[0] = Vec2::ZERO;
        state.velocities[payload] = payload_velocity;
    }
}

fn damp_bending_vertex(
    state: &mut State,
    masses: &[f64],
    center: usize,
    rest_length: f64,
    bending_viscosity: f64,
    dt: f64,
    payload: usize,
) {
    let Some(geometry) = crate::dynamics::bending::geometry(state, center) else {
        return;
    };
    let nodes = [center - 1, center, center + 1];
    let inverse_masses = nodes.map(|node| inverse_mass(masses, node, payload, true));
    let effective_inverse_mass = inverse_masses
        .iter()
        .zip(geometry.gradients)
        .map(|(inverse_mass, gradient)| inverse_mass * gradient.length_squared())
        .sum::<f64>();
    if effective_inverse_mass <= 0.0 {
        return;
    }
    let damping_rate = bending_viscosity * effective_inverse_mass / rest_length;
    let removed_fraction = 1.0 - (-damping_rate * dt).exp();
    let impulse = -removed_fraction * geometry.angle_rate / effective_inverse_mass;
    for local in 0..3 {
        state.velocities[nodes[local]] +=
            geometry.gradients[local] * (inverse_masses[local] * impulse);
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
        let trial = material.backward_euler_trial(
            crate::materials::AxialKinematics {
                extension: length - rest_length,
                extension_rate,
            },
            state.sls_state.as_ref().map(|states| states[element]),
            rest_length,
            dt,
        );
        if let (Some(states), Some(next_state)) = (&mut state.sls_state, trial.sls_state) {
            states[element] = next_state;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_velocity_projection_enforces_axial_constraints() {
        for positions in [
            vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(0.2, -1.0),
                Vec2::new(-0.3, -2.0),
                Vec2::new(0.4, -3.0),
                Vec2::new(0.0, -4.0),
            ],
            (0..5).map(|node| Vec2::new(0.0, -node as f64)).collect(),
        ] {
            let mut state = State::new(positions);
            state.velocities = vec![
                Vec2::ZERO,
                Vec2::new(3.0, -2.0),
                Vec2::new(-1.0, 4.0),
                Vec2::new(2.0, 1.0),
                Vec2::ZERO,
            ];
            let masses = vec![1.0; state.node_count()];
            let mut projection = VelocityProjection::new(state.node_count() - 1);

            projection.project(&mut state, &masses, Vec2::ZERO);

            for element in 0..state.node_count() - 1 {
                let delta = state.positions[element + 1] - state.positions[element];
                let axial_speed = (delta / delta.length())
                    .dot(state.velocities[element + 1] - state.velocities[element]);
                assert!(
                    axial_speed.abs() < 1.0e-8,
                    "element {element}: {axial_speed:e}"
                );
            }
        }
    }
}

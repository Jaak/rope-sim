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
    global_projection: GlobalProjection,
}

impl XpbdRopeRelaxer {
    pub(crate) fn new(node_count: usize) -> Self {
        Self {
            global_projection: GlobalProjection::new(node_count.saturating_sub(1)),
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
        seed_compressive_buckling(state, rest_length);

        self.project_constraints(
            config,
            state,
            masses,
            rest_length,
            dt,
            Some(target.position),
        );

        // Position corrections include cursor-driven reshaping and must not be
        // interpreted as momentum. Project the independently integrated free
        // velocities instead, so the held constraints can support gravity.
        state.velocities[0] = Vec2::ZERO;
        state.velocities[payload] = target.velocity;
        let scaled_axial_compliance = rest_length / (config.axial_rigidity * dt * dt);
        self.global_projection.project_velocities(
            state,
            masses,
            target.velocity,
            scaled_axial_compliance,
        );
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
    ) {
        let compliance = rest_length / config.axial_rigidity;
        let scaled_compliance = compliance / (dt * dt);
        let bending_weight = config.bending_rigidity * dt * dt / rest_length.powi(3);
        // Even a tension-only material retains its geometric material length
        // while held; slack is represented by folds rather than zero-length
        // elements. Its physical constitutive response remains unilateral.
        self.global_projection.begin_position_step(state);
        for _ in 0..CONSTRAINT_SWEEPS {
            self.global_projection.project_positions(
                state,
                masses,
                rest_length,
                scaled_compliance,
                bending_weight,
                held_payload.is_some(),
            );
            state.positions[0] = config.anchor;
            if let Some(payload_position) = held_payload {
                let payload = state.node_count() - 1;
                state.positions[payload] = payload_position;
            }
        }
    }

    fn resize(&mut self, node_count: usize) {
        self.global_projection.resize(node_count.saturating_sub(1));
    }
}

/// Break the exact straight-chain symmetry when held endpoints move closer
/// than the rope's material length.
///
/// A compressed axial chain with no lateral displacement has a valid but
/// unstable discrete equilibrium. Depending on roundoff to choose a buckling
/// direction made interaction behavior resolution- and solver-dependent. This
/// tiny first-mode imperfection is visually negligible and gives the global
/// projection a deterministic direction in which to represent slack.
fn seed_compressive_buckling(state: &mut State, rest_length: f64) {
    let payload = state.node_count().saturating_sub(1);
    if payload < 2 {
        return;
    }
    let chord = state.positions[payload] - state.positions[0];
    let chord_length = chord.length();
    let material_length = rest_length * payload as f64;
    if chord_length >= material_length || chord_length <= f64::EPSILON {
        return;
    }

    let direction = chord / chord_length;
    let perpendicular = Vec2::new(-direction.y, direction.x);
    let maximum_deviation = state.positions[1..payload]
        .iter()
        .enumerate()
        .map(|(local, position)| {
            let fraction = (local + 1) as f64 / payload as f64;
            perpendicular.dot(*position - (state.positions[0] + chord * fraction))
        })
        .fold(0.0_f64, |maximum, deviation| maximum.max(deviation.abs()));
    let seed_amplitude = 1.0e-3 * material_length;
    if maximum_deviation >= seed_amplitude {
        return;
    }

    let middle = payload / 2;
    let middle_fraction = middle as f64 / payload as f64;
    let middle_deviation =
        perpendicular.dot(state.positions[middle] - (state.positions[0] + chord * middle_fraction));
    let sign = if middle_deviation < 0.0 { -1.0 } else { 1.0 };
    let added_amplitude = seed_amplitude - maximum_deviation;
    for node in 1..payload {
        let fraction = node as f64 / payload as f64;
        state.positions[node] +=
            perpendicular * (sign * added_amplitude * (std::f64::consts::PI * fraction).sin());
    }
}

/// Global mass-weighted projection onto the rope's interaction response.
///
/// Axial position projection is tridiagonal; adding the quadratic bending
/// regularizer makes it pentadiagonal. The axial velocity constraint remains
/// tridiagonal. All are solved directly so a prescribed endpoint correction
/// reaches the whole rope in linear time instead of relying on local
/// Gauss-Seidel propagation.
struct GlobalProjection {
    directions: Vec<Vec2>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
    second_upper: Vec<f64>,
    first_lower: Vec<f64>,
    second_lower: Vec<f64>,
    velocity_right_hand_side: Vec<f64>,
    position_right_hand_side: Vec<Vec2>,
    position_targets: Vec<Vec2>,
}

impl GlobalProjection {
    fn new(element_count: usize) -> Self {
        Self {
            directions: vec![Vec2::ZERO; element_count],
            diagonal: vec![0.0; element_count],
            upper: vec![0.0; element_count.saturating_sub(1)],
            second_upper: vec![0.0; element_count.saturating_sub(2)],
            first_lower: vec![0.0; element_count],
            second_lower: vec![0.0; element_count],
            velocity_right_hand_side: vec![0.0; element_count],
            position_right_hand_side: vec![Vec2::ZERO; element_count],
            position_targets: vec![Vec2::ZERO; element_count + 1],
        }
    }

    fn resize(&mut self, element_count: usize) {
        self.directions.resize(element_count, Vec2::ZERO);
        self.diagonal.resize(element_count, 0.0);
        self.upper.resize(element_count.saturating_sub(1), 0.0);
        self.second_upper
            .resize(element_count.saturating_sub(2), 0.0);
        self.first_lower.resize(element_count, 0.0);
        self.second_lower.resize(element_count, 0.0);
        self.velocity_right_hand_side.resize(element_count, 0.0);
        self.position_right_hand_side
            .resize(element_count, Vec2::ZERO);
        self.position_targets.resize(element_count + 1, Vec2::ZERO);
    }

    fn begin_position_step(&mut self, state: &State) {
        self.resize(state.node_count().saturating_sub(1));
        self.position_targets.clone_from(&state.positions);
    }

    fn project_positions(
        &mut self,
        state: &mut State,
        masses: &[f64],
        rest_length: f64,
        scaled_compliance: f64,
        bending_weight: f64,
        payload_is_held: bool,
    ) {
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
                Vec2::new(if element.is_multiple_of(2) { 1.0 } else { -1.0 }, 0.0)
            };
        }

        // This local/global step minimizes the same compliant spring-plus-
        // inertia objective as the axial XPBD pass. Multiplying its normal
        // equations by dt^2 gives k*dt^2 = 1/scaled_compliance.
        let axial_weight = 1.0 / scaled_compliance;
        let dynamic_end = if payload_is_held {
            payload
        } else {
            payload + 1
        };
        let unknown_count = dynamic_end.saturating_sub(1);
        if unknown_count == 0 {
            return;
        }
        self.upper[..unknown_count.saturating_sub(1)].fill(0.0);
        self.second_upper[..unknown_count.saturating_sub(2)].fill(0.0);
        for local in 0..unknown_count {
            let node = local + 1;
            let has_right_element = node < payload;
            let attached_elements = if has_right_element { 2.0 } else { 1.0 };
            self.diagonal[local] = masses[node] + attached_elements * axial_weight;
            self.position_right_hand_side[local] = self.position_targets[node] * masses[node]
                + self.directions[node - 1] * (axial_weight * rest_length);
            if has_right_element {
                self.position_right_hand_side[local] -=
                    self.directions[node] * (axial_weight * rest_length);
            }
            if local + 1 < unknown_count {
                self.upper[local] = -axial_weight;
            }
        }

        self.position_right_hand_side[0] += state.positions[0] * axial_weight;
        if payload_is_held {
            self.position_right_hand_side[unknown_count - 1] +=
                state.positions[payload] * axial_weight;
        }

        if bending_weight > 0.0 {
            let coefficients = [1.0, -2.0, 1.0];
            for center in 1..payload {
                let nodes = [center - 1, center, center + 1];
                for a in 0..3 {
                    let row_node = nodes[a];
                    if !(1..dynamic_end).contains(&row_node) {
                        continue;
                    }
                    let row = row_node - 1;
                    self.diagonal[row] += bending_weight * coefficients[a] * coefficients[a];

                    for b in a + 1..3 {
                        let column_node = nodes[b];
                        if (1..dynamic_end).contains(&column_node) {
                            let column = column_node - 1;
                            let value = bending_weight * coefficients[a] * coefficients[b];
                            match column - row {
                                1 => self.upper[row] += value,
                                2 => self.second_upper[row] += value,
                                _ => unreachable!("a bending stencil has bandwidth two"),
                            }
                        }
                    }

                    for b in 0..3 {
                        let fixed_node = nodes[b];
                        if !(1..dynamic_end).contains(&fixed_node) {
                            self.position_right_hand_side[row] -= state.positions[fixed_node]
                                * (bending_weight * coefficients[a] * coefficients[b]);
                        }
                    }
                }
            }
        }

        if !solve_symmetric_pentadiagonal_vec2(
            &mut self.diagonal[..unknown_count],
            &self.upper[..unknown_count.saturating_sub(1)],
            &self.second_upper[..unknown_count.saturating_sub(2)],
            &mut self.position_right_hand_side[..unknown_count],
            &mut self.first_lower[..unknown_count],
            &mut self.second_lower[..unknown_count],
        ) {
            return;
        }

        for local in 0..unknown_count {
            state.positions[local + 1] = self.position_right_hand_side[local];
        }
    }

    /// Project free velocities onto the axial velocity constraints.
    ///
    /// The material's finite compliance prevents the interaction aid from
    /// treating a stretchable rope as exactly inextensible. It also regularizes
    /// nearly straight endpoint-held configurations, where an exact constraint
    /// solve can amplify a fast boundary velocity enormously. A tiny numerical
    /// floor remains for a hypothetical zero-compliance caller.
    fn project_velocities(
        &mut self,
        state: &mut State,
        masses: &[f64],
        payload_velocity: Vec2,
        scaled_compliance: f64,
    ) {
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
            self.velocity_right_hand_side[element] =
                -direction.dot(state.velocities[element + 1] - state.velocities[element]);
            if element + 1 < element_count {
                self.upper[element] =
                    -right_inverse_mass * direction.dot(self.directions[element + 1]);
            }
        }

        let regularization = scaled_compliance + maximum_diagonal.max(1.0) * 1.0e-12;
        for diagonal in &mut self.diagonal {
            *diagonal += regularization;
        }
        if !solve_symmetric_tridiagonal(
            &mut self.diagonal,
            &self.upper,
            &mut self.velocity_right_hand_side,
        ) {
            return;
        }

        for element in 0..element_count {
            let impulse = self.directions[element] * self.velocity_right_hand_side[element];
            let left_inverse_mass = inverse_mass(masses, element, payload, true);
            let right_inverse_mass = inverse_mass(masses, element + 1, payload, true);
            state.velocities[element] -= impulse * left_inverse_mass;
            state.velocities[element + 1] += impulse * right_inverse_mass;
        }
        state.velocities[0] = Vec2::ZERO;
        state.velocities[payload] = payload_velocity;
    }
}

fn solve_symmetric_pentadiagonal_vec2(
    diagonal: &mut [f64],
    first_upper: &[f64],
    second_upper: &[f64],
    right_hand_side: &mut [Vec2],
    first_lower: &mut [f64],
    second_lower: &mut [f64],
) -> bool {
    debug_assert_eq!(diagonal.len(), right_hand_side.len());
    debug_assert_eq!(first_upper.len(), diagonal.len().saturating_sub(1));
    debug_assert_eq!(second_upper.len(), diagonal.len().saturating_sub(2));
    debug_assert_eq!(first_lower.len(), diagonal.len());
    debug_assert_eq!(second_lower.len(), diagonal.len());
    if diagonal.is_empty() {
        return true;
    }

    first_lower.fill(0.0);
    second_lower.fill(0.0);
    for index in 0..diagonal.len() {
        if index >= 2 {
            let second_pivot = diagonal[index - 2];
            if !second_pivot.is_finite() || second_pivot <= 0.0 {
                return false;
            }
            second_lower[index] = second_upper[index - 2] / second_pivot;
        }
        if index >= 1 {
            let first_pivot = diagonal[index - 1];
            if !first_pivot.is_finite() || first_pivot <= 0.0 {
                return false;
            }
            let coupling = if index >= 2 {
                second_lower[index] * diagonal[index - 2] * first_lower[index - 1]
            } else {
                0.0
            };
            first_lower[index] = (first_upper[index - 1] - coupling) / first_pivot;
        }
        let first_term = if index >= 1 {
            first_lower[index] * first_lower[index] * diagonal[index - 1]
        } else {
            0.0
        };
        let second_term = if index >= 2 {
            second_lower[index] * second_lower[index] * diagonal[index - 2]
        } else {
            0.0
        };
        diagonal[index] -= first_term + second_term;
        if !diagonal[index].is_finite() || diagonal[index] <= 0.0 {
            return false;
        }
    }

    for index in 0..diagonal.len() {
        if index >= 1 {
            right_hand_side[index] -= right_hand_side[index - 1] * first_lower[index];
        }
        if index >= 2 {
            right_hand_side[index] -= right_hand_side[index - 2] * second_lower[index];
        }
    }
    for (value, &pivot) in right_hand_side.iter_mut().zip(diagonal.iter()) {
        *value = *value / pivot;
    }
    for index in (0..diagonal.len()).rev() {
        if index + 1 < diagonal.len() {
            right_hand_side[index] -= right_hand_side[index + 1] * first_lower[index + 1];
        }
        if index + 2 < diagonal.len() {
            right_hand_side[index] -= right_hand_side[index + 2] * second_lower[index + 2];
        }
    }
    right_hand_side.iter().all(|value| value.is_finite())
}

fn solve_symmetric_tridiagonal(
    diagonal: &mut [f64],
    upper: &[f64],
    right_hand_side: &mut [f64],
) -> bool {
    debug_assert_eq!(diagonal.len(), right_hand_side.len());
    debug_assert_eq!(upper.len(), diagonal.len().saturating_sub(1));
    if diagonal.is_empty() {
        return true;
    }

    for index in 1..diagonal.len() {
        let pivot = diagonal[index - 1];
        if !pivot.is_finite() || pivot <= 0.0 {
            return false;
        }
        let factor = upper[index - 1] / pivot;
        diagonal[index] -= factor * upper[index - 1];
        right_hand_side[index] -= factor * right_hand_side[index - 1];
    }

    let last = diagonal.len() - 1;
    if !diagonal[last].is_finite() || diagonal[last] <= 0.0 {
        return false;
    }
    right_hand_side[last] /= diagonal[last];
    for index in (0..last).rev() {
        if !diagonal[index].is_finite() || diagonal[index] <= 0.0 {
            return false;
        }
        right_hand_side[index] =
            (right_hand_side[index] - upper[index] * right_hand_side[index + 1]) / diagonal[index];
    }
    right_hand_side.iter().all(|value| value.is_finite())
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
    fn pentadiagonal_solver_recovers_a_known_vector_solution() {
        let expected = [
            Vec2::new(1.0, -2.0),
            Vec2::new(-0.5, 3.0),
            Vec2::new(2.5, 0.25),
            Vec2::new(-1.5, -0.75),
            Vec2::new(0.5, 1.25),
        ];
        let original_diagonal = [5.0, 6.0, 6.0, 6.0, 5.0];
        let first_upper = [-2.0; 4];
        let second_upper = [0.5; 3];
        let mut right_hand_side = [Vec2::ZERO; 5];
        for row in 0..expected.len() {
            right_hand_side[row] += expected[row] * original_diagonal[row];
            if row >= 1 {
                right_hand_side[row] += expected[row - 1] * first_upper[row - 1];
            }
            if row + 1 < expected.len() {
                right_hand_side[row] += expected[row + 1] * first_upper[row];
            }
            if row >= 2 {
                right_hand_side[row] += expected[row - 2] * second_upper[row - 2];
            }
            if row + 2 < expected.len() {
                right_hand_side[row] += expected[row + 2] * second_upper[row];
            }
        }
        let mut diagonal = original_diagonal;
        let mut first_lower = [0.0; 5];
        let mut second_lower = [0.0; 5];

        assert!(solve_symmetric_pentadiagonal_vec2(
            &mut diagonal,
            &first_upper,
            &second_upper,
            &mut right_hand_side,
            &mut first_lower,
            &mut second_lower,
        ));

        for (actual, expected) in right_hand_side.into_iter().zip(expected) {
            assert!((actual - expected).length() < 1.0e-12);
        }
    }

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
            let mut projection = GlobalProjection::new(state.node_count() - 1);

            projection.project_velocities(&mut state, &masses, Vec2::ZERO, 0.0);

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

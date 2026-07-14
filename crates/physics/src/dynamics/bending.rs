use crate::math::Vec2;
use crate::state::State;

pub(crate) type Matrix2 = [[f64; 2]; 2];

const ZERO_MATRIX: Matrix2 = [[0.0; 2]; 2];
const MINIMUM_EDGE_LENGTH_SQUARED: f64 = 1.0e-18;

#[derive(Clone, Copy, Debug)]
pub(crate) struct BendingGeometry {
    pub angle: f64,
    pub angle_rate: f64,
    pub(crate) gradients: [Vec2; 3],
    incoming: Vec2,
    outgoing: Vec2,
    incoming_length_squared: f64,
    outgoing_length_squared: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BendingResponse {
    pub position_jacobian: [[Matrix2; 3]; 3],
    pub velocity_jacobian: [[Matrix2; 3]; 3],
}

/// Turning-angle geometry for the vertex at `center`.
///
/// A straight rope has zero angle. The signed angle is in `[-pi, pi]` and is
/// measured from the incoming edge to the outgoing edge.
pub(crate) fn geometry(state: &State, center: usize) -> Option<BendingGeometry> {
    if center == 0 || center + 1 >= state.node_count() {
        return None;
    }

    let incoming = state.positions[center] - state.positions[center - 1];
    let outgoing = state.positions[center + 1] - state.positions[center];
    let incoming_length_squared = incoming.length_squared();
    let outgoing_length_squared = outgoing.length_squared();
    if incoming_length_squared <= MINIMUM_EDGE_LENGTH_SQUARED
        || outgoing_length_squared <= MINIMUM_EDGE_LENGTH_SQUARED
    {
        return None;
    }

    let angle = cross(incoming, outgoing).atan2(incoming.dot(outgoing));
    let incoming_angle_gradient = angle_gradient(incoming, incoming_length_squared);
    let outgoing_angle_gradient = angle_gradient(outgoing, outgoing_length_squared);
    let gradients = [
        incoming_angle_gradient,
        -(incoming_angle_gradient + outgoing_angle_gradient),
        outgoing_angle_gradient,
    ];

    let velocities = [
        state.velocities[center - 1],
        state.velocities[center],
        state.velocities[center + 1],
    ];
    let angle_rate = gradients
        .iter()
        .zip(velocities)
        .map(|(gradient, velocity)| gradient.dot(velocity))
        .sum();

    Some(BendingGeometry {
        angle,
        angle_rate,
        gradients,
        incoming,
        outgoing,
        incoming_length_squared,
        outgoing_length_squared,
    })
}

/// Elastic and viscous bending forces without constructing their tangents.
///
/// Explicit integration and nonlinear residual evaluation only need forces.
/// Keeping this path separate avoids constructing angle Hessians and nine
/// local Jacobian blocks that those callers would immediately discard.
pub(crate) fn forces(
    state: &State,
    center: usize,
    rest_length: f64,
    rigidity: f64,
    viscosity: f64,
) -> Option<[Vec2; 3]> {
    let geometry = geometry(state, center)?;
    Some(forces_from_geometry(
        geometry,
        rest_length,
        rigidity,
        viscosity,
    ))
}

/// Elastic and viscous bending forces and their exact local tangents.
///
/// The elastic energy is `0.5 * B * angle^2 / rest_length`; the Rayleigh
/// dissipation potential is `0.5 * C_B * angle_rate^2 / rest_length`.
pub(crate) fn response(
    state: &State,
    center: usize,
    rest_length: f64,
    rigidity: f64,
    viscosity: f64,
) -> Option<BendingResponse> {
    let geometry = geometry(state, center)?;
    let elastic_scale = rigidity / rest_length;
    let viscous_scale = viscosity / rest_length;
    let incoming_angle_hessian = angle_hessian(geometry.incoming, geometry.incoming_length_squared);
    let outgoing_angle_hessian = angle_hessian(geometry.outgoing, geometry.outgoing_length_squared);
    let mut angle_hessian = [[ZERO_MATRIX; 3]; 3];
    add_edge_hessian(&mut angle_hessian, 0, -1.0, incoming_angle_hessian);
    add_edge_hessian(&mut angle_hessian, 1, 1.0, outgoing_angle_hessian);

    let velocities = [
        state.velocities[center - 1],
        state.velocities[center],
        state.velocities[center + 1],
    ];
    let hessian_velocity: [Vec2; 3] = std::array::from_fn(|row| {
        let mut value = Vec2::ZERO;
        for (column, velocity) in velocities.iter().enumerate() {
            value += multiply_matrix_vector(angle_hessian[row][column], *velocity);
        }
        value
    });

    let mut position_jacobian = [[ZERO_MATRIX; 3]; 3];
    let mut velocity_jacobian = [[ZERO_MATRIX; 3]; 3];
    for row in 0..3 {
        for column in 0..3 {
            let elastic = add_matrix(
                outer_product(geometry.gradients[row], geometry.gradients[column]),
                scale_matrix(angle_hessian[row][column], geometry.angle),
            );
            let viscous = add_matrix(
                outer_product(geometry.gradients[row], hessian_velocity[column]),
                scale_matrix(angle_hessian[row][column], geometry.angle_rate),
            );
            position_jacobian[row][column] = add_matrix(
                scale_matrix(elastic, -elastic_scale),
                scale_matrix(viscous, -viscous_scale),
            );
            velocity_jacobian[row][column] = scale_matrix(
                outer_product(geometry.gradients[row], geometry.gradients[column]),
                -viscous_scale,
            );
        }
    }

    Some(BendingResponse {
        position_jacobian,
        velocity_jacobian,
    })
}

fn forces_from_geometry(
    geometry: BendingGeometry,
    rest_length: f64,
    rigidity: f64,
    viscosity: f64,
) -> [Vec2; 3] {
    let generalized_moment =
        rigidity * geometry.angle / rest_length + viscosity * geometry.angle_rate / rest_length;
    geometry
        .gradients
        .map(|gradient| gradient * -generalized_moment)
}

fn cross(left: Vec2, right: Vec2) -> f64 {
    left.x * right.y - left.y * right.x
}

fn angle_gradient(edge: Vec2, length_squared: f64) -> Vec2 {
    Vec2::new(-edge.y / length_squared, edge.x / length_squared)
}

fn angle_hessian(edge: Vec2, length_squared: f64) -> Matrix2 {
    let inverse_fourth_power = 1.0 / (length_squared * length_squared);
    [
        [
            2.0 * edge.x * edge.y * inverse_fourth_power,
            (edge.y * edge.y - edge.x * edge.x) * inverse_fourth_power,
        ],
        [
            (edge.y * edge.y - edge.x * edge.x) * inverse_fourth_power,
            -2.0 * edge.x * edge.y * inverse_fourth_power,
        ],
    ]
}

fn add_edge_hessian(
    output: &mut [[Matrix2; 3]; 3],
    first_node: usize,
    angle_sign: f64,
    edge_hessian: Matrix2,
) {
    let position_signs = [-1.0, 1.0];
    for local_row in 0..2 {
        for local_column in 0..2 {
            let scale = angle_sign * position_signs[local_row] * position_signs[local_column];
            add_scaled_matrix(
                &mut output[first_node + local_row][first_node + local_column],
                edge_hessian,
                scale,
            );
        }
    }
}

fn outer_product(left: Vec2, right: Vec2) -> Matrix2 {
    [
        [left.x * right.x, left.x * right.y],
        [left.y * right.x, left.y * right.y],
    ]
}

fn multiply_matrix_vector(matrix: Matrix2, vector: Vec2) -> Vec2 {
    Vec2::new(
        matrix[0][0] * vector.x + matrix[0][1] * vector.y,
        matrix[1][0] * vector.x + matrix[1][1] * vector.y,
    )
}

fn scale_matrix(matrix: Matrix2, scale: f64) -> Matrix2 {
    matrix.map(|row| row.map(|value| value * scale))
}

fn add_matrix(left: Matrix2, right: Matrix2) -> Matrix2 {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}

fn add_scaled_matrix(output: &mut Matrix2, input: Matrix2, scale: f64) {
    for row in 0..2 {
        for column in 0..2 {
            output[row][column] += scale * input[row][column];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REST_LENGTH: f64 = 0.8;
    const RIGIDITY: f64 = 0.17;
    const VISCOSITY: f64 = 0.031;

    fn bent_state() -> State {
        let mut state = State::new(vec![
            Vec2::new(-0.7, -0.1),
            Vec2::new(0.0, 0.2),
            Vec2::new(0.8, -0.25),
        ]);
        state.velocities = vec![
            Vec2::new(0.2, -0.1),
            Vec2::new(-0.3, 0.4),
            Vec2::new(0.1, 0.25),
        ];
        state
    }

    #[test]
    fn straight_stationary_vertex_has_no_bending_force() {
        let state = State::new(vec![Vec2::new(-1.0, 0.0), Vec2::ZERO, Vec2::new(1.0, 0.0)]);
        let forces = forces(&state, 1, 1.0, RIGIDITY, VISCOSITY).unwrap();

        assert_eq!(forces, [Vec2::ZERO; 3]);
    }

    #[test]
    fn elastic_force_is_the_negative_bending_energy_gradient() {
        let state = bent_state();
        let forces = forces(&state, 1, REST_LENGTH, RIGIDITY, 0.0).unwrap();
        let step = 1.0e-6;

        for (node, force) in forces.iter().copied().enumerate() {
            for component in 0..2 {
                let mut plus = state.clone();
                let mut minus = state.clone();
                *component_mut(&mut plus.positions[node], component) += step;
                *component_mut(&mut minus.positions[node], component) -= step;
                let plus_geometry = geometry(&plus, 1).unwrap();
                let minus_geometry = geometry(&minus, 1).unwrap();
                let plus_energy = 0.5 * RIGIDITY * plus_geometry.angle.powi(2) / REST_LENGTH;
                let minus_energy = 0.5 * RIGIDITY * minus_geometry.angle.powi(2) / REST_LENGTH;
                let energy_gradient = (plus_energy - minus_energy) / (2.0 * step);
                let force = component_value(force, component);
                assert!((force + energy_gradient).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn bending_tangents_match_central_differences() {
        let state = bent_state();
        let analytic_response = response(&state, 1, REST_LENGTH, RIGIDITY, VISCOSITY).unwrap();
        let step = 1.0e-6;

        for position_derivative in [true, false] {
            for column_node in 0..3 {
                for column_component in 0..2 {
                    let mut plus = state.clone();
                    let mut minus = state.clone();
                    let plus_value = if position_derivative {
                        &mut plus.positions[column_node]
                    } else {
                        &mut plus.velocities[column_node]
                    };
                    let minus_value = if position_derivative {
                        &mut minus.positions[column_node]
                    } else {
                        &mut minus.velocities[column_node]
                    };
                    *component_mut(plus_value, column_component) += step;
                    *component_mut(minus_value, column_component) -= step;
                    let plus_force = forces(&plus, 1, REST_LENGTH, RIGIDITY, VISCOSITY).unwrap();
                    let minus_force = forces(&minus, 1, REST_LENGTH, RIGIDITY, VISCOSITY).unwrap();

                    for row_node in 0..3 {
                        for row_component in 0..2 {
                            let finite_difference =
                                (component_value(plus_force[row_node], row_component)
                                    - component_value(minus_force[row_node], row_component))
                                    / (2.0 * step);
                            let matrix = if position_derivative {
                                analytic_response.position_jacobian[row_node][column_node]
                            } else {
                                analytic_response.velocity_jacobian[row_node][column_node]
                            };
                            assert!(
                                (finite_difference - matrix[row_component][column_component]).abs()
                                    < 2.0e-7,
                                "row={row_node}:{row_component}, column={column_node}:{column_component}, position={position_derivative}, finite={finite_difference}, analytic={}",
                                matrix[row_component][column_component]
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn circular_arc_energy_converges_under_refinement() {
        fn arc_energy(segment_count: usize) -> f64 {
            let radius = 2.0;
            let total_angle = 1.0;
            let rest_length = radius * total_angle / segment_count as f64;
            let positions = (0..=segment_count)
                .map(|index| {
                    let angle = index as f64 * total_angle / segment_count as f64;
                    Vec2::new(radius * angle.sin(), -radius * angle.cos())
                })
                .collect();
            let state = State::new(positions);
            (1..segment_count)
                .map(|center| {
                    let angle = geometry(&state, center).unwrap().angle;
                    0.5 * RIGIDITY * angle * angle / rest_length
                })
                .sum()
        }

        let continuum_energy = 0.5 * RIGIDITY / 2.0;
        let coarse_error = (arc_energy(16) - continuum_energy).abs();
        let fine_error = (arc_energy(128) - continuum_energy).abs();

        assert!(fine_error < coarse_error);
        assert!(fine_error < 0.01 * continuum_energy);
    }

    #[test]
    fn bending_viscosity_never_adds_mechanical_power() {
        let state = bent_state();
        let forces = forces(&state, 1, REST_LENGTH, 0.0, VISCOSITY).unwrap();
        let power: f64 = forces
            .iter()
            .zip(&state.velocities)
            .map(|(force, velocity)| force.dot(*velocity))
            .sum();

        assert!(power <= 0.0);
    }

    fn component_mut(vector: &mut Vec2, component: usize) -> &mut f64 {
        if component == 0 {
            &mut vector.x
        } else {
            &mut vector.y
        }
    }

    fn component_value(vector: Vec2, component: usize) -> f64 {
        if component == 0 { vector.x } else { vector.y }
    }
}

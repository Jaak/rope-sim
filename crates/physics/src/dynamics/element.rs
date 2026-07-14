use crate::materials::{AxialKinematics, AxialResponse};
use crate::math::{Matrix2, Vec2, matrix2_vector_product};
use crate::state::State;

#[derive(Clone, Copy)]
pub(super) struct ElementKinematics {
    length: f64,
    pub axial: AxialKinematics,
    pub direction: Vec2,
    relative_velocity: Vec2,
}

pub(super) fn kinematics(
    state: &State,
    left: usize,
    rest_length: f64,
) -> Option<ElementKinematics> {
    let right = left + 1;
    let delta = state.positions[right] - state.positions[left];
    let length = delta.length();
    if length <= f64::EPSILON {
        return None;
    }

    let direction = delta / length;
    let relative_velocity = state.velocities[right] - state.velocities[left];
    Some(ElementKinematics {
        length,
        direction,
        relative_velocity,
        axial: AxialKinematics {
            extension: length - rest_length,
            extension_rate: direction.dot(relative_velocity),
        },
    })
}

pub(super) fn extension_rate(state: &State, left: usize, rest_length: f64) -> f64 {
    if let Some(k) = kinematics(state, left, rest_length) {
        k.axial.extension_rate
    } else {
        0.0
    }
}

pub(super) fn force_jacobians(
    kinematics: ElementKinematics,
    response: AxialResponse,
) -> (Matrix2, Matrix2) {
    let direction = kinematics.direction;
    let inverse_length = 1.0 / kinematics.length;
    let projection = [
        [
            (1.0 - direction.x * direction.x) * inverse_length,
            -direction.x * direction.y * inverse_length,
        ],
        [
            -direction.x * direction.y * inverse_length,
            (1.0 - direction.y * direction.y) * inverse_length,
        ],
    ];
    let projected_velocity = matrix2_vector_product(projection, kinematics.relative_velocity);
    let direction_components = [direction.x, direction.y];
    let projected_velocity_components = [projected_velocity.x, projected_velocity.y];
    let mut position_jacobian = [[0.0; 2]; 2];
    let mut velocity_jacobian = [[0.0; 2]; 2];

    for row in 0..2 {
        for column in 0..2 {
            let direction_outer = direction_components[row] * direction_components[column];
            position_jacobian[row][column] = response.force * projection[row][column]
                + response.length_tangent * direction_outer
                + response.rate_tangent
                    * direction_components[row]
                    * projected_velocity_components[column];
            velocity_jacobian[row][column] = response.rate_tangent * direction_outer;
        }
    }

    (position_jacobian, velocity_jacobian)
}

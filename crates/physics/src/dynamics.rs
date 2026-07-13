mod element;

use crate::config::SimulationConfig;
use crate::integrators::block_tridiagonal::BlockTridiagonalMatrix;
use crate::integrators::{AccelerationJacobianBlock, DynamicalSystem};
use crate::kinematics::{KinematicMotion, KinematicTarget};
use crate::materials::{AxialKinematics, AxialMaterial, AxialResponse};
use crate::math::Vec2;
use crate::state::State;

use element::{extension_rate, force_jacobians, kinematics, scalar_jacobians};

const POSITION_OFFSET: usize = 0;
const VELOCITY_OFFSET: usize = 2;
const MATERIAL_COMPONENT: usize = 4;
const MAX_KINEMATIC_TRAVEL_FRACTION: f64 = 0.1;

pub(crate) struct RopeDynamics<'a> {
    config: &'a SimulationConfig,
    masses: &'a [f64],
    rest_length: f64,
    payload_target: Option<KinematicTarget>,
    payload_motion: Option<KinematicMotion>,
    kinematic_speed: f64,
    material: AxialMaterial,
}

impl<'a> RopeDynamics<'a> {
    pub(crate) fn new(
        config: &'a SimulationConfig,
        masses: &'a [f64],
        rest_length: f64,
        payload_target: Option<KinematicTarget>,
        payload_motion: Option<KinematicMotion>,
        kinematic_speed: f64,
    ) -> Self {
        Self {
            config,
            masses,
            rest_length,
            payload_target,
            payload_motion,
            kinematic_speed,
            material: AxialMaterial::from_config(*config),
        }
    }

    fn payload_index(&self) -> usize {
        self.masses.len() - 1
    }

    fn payload_target_at(&self, stage_time: f64) -> Option<KinematicTarget> {
        self.payload_motion
            .map(|motion| motion.target_after(stage_time))
            .or(self.payload_target)
    }

    fn assemble_acceleration_jacobian(
        &self,
        state: &State,
        output: &mut Vec<AccelerationJacobianBlock>,
        mut response_for: impl FnMut(usize, AxialKinematics) -> AxialResponse,
    ) {
        output.clear();
        let zero = [[0.0; 2]; 2];
        let air_damping = self.config.air_damping_rate;

        for node in 0..state.node_count() {
            if self.is_dynamic_node(node) {
                output.push(AccelerationJacobianBlock {
                    row_node: node,
                    column_node: node,
                    position: zero,
                    velocity: [[-air_damping, 0.0], [0.0, -air_damping]],
                });
            }
        }

        for left in 0..self.config.segment_count {
            let right = left + 1;
            let (force_position_jacobian, force_velocity_jacobian) =
                if let Some(kinematics) = kinematics(state, left, self.rest_length) {
                    let response = response_for(left, kinematics.axial);
                    force_jacobians(kinematics, response)
                } else {
                    (zero, zero)
                };

            if self.is_dynamic_node(left) {
                push_element_jacobian_row(
                    output,
                    left,
                    left,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    -1.0 / self.masses[left],
                );
                push_element_jacobian_row(
                    output,
                    left,
                    right,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    1.0 / self.masses[left],
                );
            }
            if self.is_dynamic_node(right) {
                push_element_jacobian_row(
                    output,
                    right,
                    left,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    1.0 / self.masses[right],
                );
                push_element_jacobian_row(
                    output,
                    right,
                    right,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    -1.0 / self.masses[right],
                );
            }
        }
    }

    #[cfg(test)]
    fn acceleration_jacobian(&self, state: &State, output: &mut Vec<AccelerationJacobianBlock>) {
        self.assemble_acceleration_jacobian(state, output, |left, kinematics| {
            self.material
                .response(kinematics, state.material_state[left], self.rest_length)
        });
    }
}

impl DynamicalSystem for RopeDynamics<'_> {
    fn accelerations(&self, state: &State, output: &mut [Vec2]) {
        for (index, acceleration) in output.iter_mut().enumerate() {
            if self.is_dynamic_node(index) {
                *acceleration =
                    self.config.gravity - state.velocities[index] * self.config.air_damping_rate;
            }
        }

        for left in 0..self.config.segment_count {
            let Some(kinematics) = kinematics(state, left, self.rest_length) else {
                continue;
            };
            let response = self.material.response(
                kinematics.axial,
                state.material_state[left],
                self.rest_length,
            );
            let force = kinematics.direction * response.force;
            let right = left + 1;

            if self.is_dynamic_node(left) {
                output[left] += force / self.masses[left];
            }
            if self.is_dynamic_node(right) {
                output[right] -= force / self.masses[right];
            }
        }
    }

    fn material_state_derivatives(&self, state: &State, output: &mut [f64]) {
        output.fill(0.0);
        if !self.material.has_internal_state() {
            return;
        }

        for (left, derivative) in output
            .iter_mut()
            .enumerate()
            .take(self.config.segment_count)
        {
            *derivative = self.material.state_derivative(
                extension_rate(state, left),
                state.material_state[left],
                self.rest_length,
            );
        }
    }

    fn first_order_jacobian(&self, state: &State, output: &mut BlockTridiagonalMatrix) {
        output.resize_and_clear(state.node_count());

        for node in 0..state.node_count() {
            if self.is_dynamic_node(node) {
                output.add_value(node, node, 0, 2, 1.0);
                output.add_value(node, node, 1, 3, 1.0);
                output.add_value(node, node, 2, 2, -self.config.air_damping_rate);
                output.add_value(node, node, 3, 3, -self.config.air_damping_rate);
            }
        }

        let state_tangents = self.material.state_tangents(self.rest_length);
        let force_state_tangent = self.material.force_state_tangent();
        for left in 0..self.config.segment_count {
            let right = left + 1;
            output.add_value(
                left,
                left,
                MATERIAL_COMPONENT,
                MATERIAL_COMPONENT,
                -self.material.relaxation_rate(),
            );

            let Some(kinematics) = kinematics(state, left, self.rest_length) else {
                continue;
            };
            let response = self.material.response(
                kinematics.axial,
                state.material_state[left],
                self.rest_length,
            );
            let (force_position_jacobian, force_velocity_jacobian) =
                force_jacobians(kinematics, response);

            if self.is_dynamic_node(left) {
                add_force_jacobian_row(
                    output,
                    left,
                    left,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    -1.0 / self.masses[left],
                );
                add_force_jacobian_row(
                    output,
                    left,
                    right,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    1.0 / self.masses[left],
                );
                output.add_value(
                    left,
                    left,
                    VELOCITY_OFFSET,
                    MATERIAL_COMPONENT,
                    kinematics.direction.x * force_state_tangent / self.masses[left],
                );
                output.add_value(
                    left,
                    left,
                    VELOCITY_OFFSET + 1,
                    MATERIAL_COMPONENT,
                    kinematics.direction.y * force_state_tangent / self.masses[left],
                );
            }
            if self.is_dynamic_node(right) {
                add_force_jacobian_row(
                    output,
                    right,
                    left,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    1.0 / self.masses[right],
                );
                add_force_jacobian_row(
                    output,
                    right,
                    right,
                    force_position_jacobian,
                    force_velocity_jacobian,
                    -1.0 / self.masses[right],
                );
                output.add_value(
                    right,
                    left,
                    VELOCITY_OFFSET,
                    MATERIAL_COMPONENT,
                    -kinematics.direction.x * force_state_tangent / self.masses[right],
                );
                output.add_value(
                    right,
                    left,
                    VELOCITY_OFFSET + 1,
                    MATERIAL_COMPONENT,
                    -kinematics.direction.y * force_state_tangent / self.masses[right],
                );
            }

            let (state_position_jacobian, state_velocity_jacobian) = scalar_jacobians(
                kinematics,
                state_tangents.extension,
                state_tangents.extension_rate,
            );
            for component in 0..2 {
                output.add_value(
                    left,
                    left,
                    MATERIAL_COMPONENT,
                    POSITION_OFFSET + component,
                    -state_position_jacobian[component],
                );
                output.add_value(
                    left,
                    right,
                    MATERIAL_COMPONENT,
                    POSITION_OFFSET + component,
                    state_position_jacobian[component],
                );
                output.add_value(
                    left,
                    left,
                    MATERIAL_COMPONENT,
                    VELOCITY_OFFSET + component,
                    -state_velocity_jacobian[component],
                );
                output.add_value(
                    left,
                    right,
                    MATERIAL_COMPONENT,
                    VELOCITY_OFFSET + component,
                    state_velocity_jacobian[component],
                );
            }
        }
    }

    fn prepare_implicit_state(&self, initial: &State, trial: &mut State, dt: f64) {
        if !self.material.has_internal_state() {
            return;
        }

        for left in 0..self.config.segment_count {
            trial.material_state[left] = self.material.backward_euler_state(
                extension_rate(trial, left),
                initial.material_state[left],
                self.rest_length,
                dt,
            );
        }
    }

    fn implicit_acceleration_jacobian(
        &self,
        _initial: &State,
        state: &State,
        dt: f64,
        output: &mut Vec<AccelerationJacobianBlock>,
    ) {
        self.assemble_acceleration_jacobian(state, output, |left, kinematics| {
            self.material.backward_euler_response(
                kinematics,
                state.material_state[left],
                self.rest_length,
                dt,
            )
        });
    }

    fn is_dynamic_node(&self, index: usize) -> bool {
        index != 0 && !(index == self.payload_index() && self.payload_target.is_some())
    }

    fn enforce_kinematics(&self, state: &mut State, stage_time: f64) {
        state.positions[0] = self.config.anchor;
        state.velocities[0] = Vec2::ZERO;

        if let Some(target) = self.payload_target_at(stage_time) {
            let index = self.payload_index();
            state.positions[index] = target.position;
            state.velocities[index] = target.velocity;
        }
    }

    fn explicit_stable_timestep(&self) -> f64 {
        let minimum_dynamic_mass = self.masses[1..]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let spring_limited_dt = self.elastic_stable_timestep();

        let element_damping = self.material.explicit_viscosity() / self.rest_length;
        let maximum_damping_rate =
            self.config.air_damping_rate + 4.0 * element_damping / minimum_dynamic_mass;
        let damping_limited_dt = if maximum_damping_rate > 0.0 {
            0.5 / maximum_damping_rate
        } else {
            f64::INFINITY
        };

        let relaxation_rate = self.material.relaxation_rate();
        let relaxation_limited_dt = if relaxation_rate > 0.0 {
            0.5 / relaxation_rate
        } else {
            f64::INFINITY
        };

        spring_limited_dt
            .min(damping_limited_dt)
            .min(relaxation_limited_dt)
    }

    fn elastic_stable_timestep(&self) -> f64 {
        let minimum_dynamic_mass = self.masses[1..]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let element_stiffness = self.material.instantaneous_rigidity() / self.rest_length;
        0.8 * (minimum_dynamic_mass / element_stiffness).sqrt()
    }

    fn has_kinematic_target(&self) -> bool {
        self.payload_target.is_some()
    }

    fn kinematic_timestep_limit(&self) -> f64 {
        if self.payload_target.is_some() && self.kinematic_speed > 0.0 {
            MAX_KINEMATIC_TRAVEL_FRACTION * self.rest_length / self.kinematic_speed
        } else {
            f64::INFINITY
        }
    }
}

fn push_element_jacobian_row(
    output: &mut Vec<AccelerationJacobianBlock>,
    row_node: usize,
    column_node: usize,
    force_position_jacobian: [[f64; 2]; 2],
    force_velocity_jacobian: [[f64; 2]; 2],
    scale: f64,
) {
    output.push(AccelerationJacobianBlock {
        row_node,
        column_node,
        position: force_position_jacobian.map(|row| row.map(|value| scale * value)),
        velocity: force_velocity_jacobian.map(|row| row.map(|value| scale * value)),
    });
}

fn add_force_jacobian_row(
    output: &mut BlockTridiagonalMatrix,
    row_node: usize,
    column_node: usize,
    force_position_jacobian: [[f64; 2]; 2],
    force_velocity_jacobian: [[f64; 2]; 2],
    scale: f64,
) {
    for row in 0..2 {
        for column in 0..2 {
            output.add_value(
                row_node,
                column_node,
                VELOCITY_OFFSET + row,
                POSITION_OFFSET + column,
                scale * force_position_jacobian[row][column],
            );
            output.add_value(
                row_node,
                column_node,
                VELOCITY_OFFSET + row,
                VELOCITY_OFFSET + column,
                scale * force_velocity_jacobian[row][column],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RopeModelKind;

    #[test]
    fn analytic_acceleration_jacobian_matches_central_differences() {
        let config = SimulationConfig {
            segment_count: 3,
            rope_model: RopeModelKind::KelvinVoigt,
            axial_viscosity: 7.0,
            ..SimulationConfig::default()
        };
        let rest_length = config.rope_length / config.segment_count as f64;
        let masses = test_masses(config);
        let mut state = test_state();
        state.velocities = vec![
            Vec2::ZERO,
            Vec2::new(0.3, -0.2),
            Vec2::new(-0.4, 0.1),
            Vec2::new(0.2, 0.5),
        ];

        let system = RopeDynamics::new(&config, &masses, rest_length, None, None, 0.0);
        let mut blocks = Vec::new();
        system.acceleration_jacobian(&state, &mut blocks);

        for column_node in 0..state.node_count() {
            for position_derivative in [true, false] {
                for column_component in 0..2 {
                    let mut analytic = vec![Vec2::ZERO; state.node_count()];
                    for block in blocks
                        .iter()
                        .filter(|block| block.column_node == column_node)
                    {
                        let matrix = if position_derivative {
                            block.position
                        } else {
                            block.velocity
                        };
                        analytic[block.row_node].x += matrix[0][column_component];
                        analytic[block.row_node].y += matrix[1][column_component];
                    }

                    let step = 1.0e-6;
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
                    add_component(plus_value, column_component, step);
                    add_component(minus_value, column_component, -step);

                    let mut plus_acceleration = vec![Vec2::ZERO; state.node_count()];
                    let mut minus_acceleration = vec![Vec2::ZERO; state.node_count()];
                    system.accelerations(&plus, &mut plus_acceleration);
                    system.accelerations(&minus, &mut minus_acceleration);

                    for row_node in 1..state.node_count() {
                        let finite_difference = (plus_acceleration[row_node]
                            - minus_acceleration[row_node])
                            / (2.0 * step);
                        assert_component_close(analytic[row_node].x, finite_difference.x);
                        assert_component_close(analytic[row_node].y, finite_difference.y);
                    }
                }
            }
        }
    }

    #[test]
    fn implicit_sls_jacobian_matches_eliminated_state_central_differences() {
        let config = SimulationConfig {
            segment_count: 3,
            rope_model: RopeModelKind::StandardLinearSolid,
            axial_viscosity: 900.0,
            transient_axial_rigidity: 18_000.0,
            ..SimulationConfig::default()
        };
        let rest_length = config.rope_length / config.segment_count as f64;
        let masses = test_masses(config);
        let mut initial = test_state();
        initial.material_state = vec![120.0, -35.0, 80.0];

        let system = RopeDynamics::new(&config, &masses, rest_length, None, None, 0.0);
        let dt = 0.01;
        let trial_velocities = [
            Vec2::ZERO,
            Vec2::new(0.3, -0.2),
            Vec2::new(-0.4, 0.1),
            Vec2::new(0.2, 0.5),
        ];
        let mut trial = initial.clone();
        let node_count = trial.node_count();
        for (position, velocity) in trial.positions[1..node_count]
            .iter_mut()
            .zip(&trial_velocities[1..node_count])
        {
            *position += *velocity * dt;
        }
        trial.velocities[1..node_count].copy_from_slice(&trial_velocities[1..node_count]);
        system.prepare_implicit_state(&initial, &mut trial, dt);

        let mut blocks = Vec::new();
        system.implicit_acceleration_jacobian(&initial, &trial, dt, &mut blocks);

        for column_node in 1..trial.node_count() {
            for column_component in 0..2 {
                let mut analytic = vec![Vec2::ZERO; trial.node_count()];
                for block in blocks
                    .iter()
                    .filter(|block| block.column_node == column_node)
                {
                    analytic[block.row_node].x += block.position[0][column_component]
                        + block.velocity[0][column_component] / dt;
                    analytic[block.row_node].y += block.position[1][column_component]
                        + block.velocity[1][column_component] / dt;
                }

                let step = 1.0e-6;
                let mut plus = trial.clone();
                let mut minus = trial.clone();
                add_component(&mut plus.positions[column_node], column_component, step);
                add_component(&mut minus.positions[column_node], column_component, -step);
                add_component(
                    &mut plus.velocities[column_node],
                    column_component,
                    step / dt,
                );
                add_component(
                    &mut minus.velocities[column_node],
                    column_component,
                    -step / dt,
                );
                system.prepare_implicit_state(&initial, &mut plus, dt);
                system.prepare_implicit_state(&initial, &mut minus, dt);

                let mut plus_acceleration = vec![Vec2::ZERO; trial.node_count()];
                let mut minus_acceleration = vec![Vec2::ZERO; trial.node_count()];
                system.accelerations(&plus, &mut plus_acceleration);
                system.accelerations(&minus, &mut minus_acceleration);

                for row_node in 1..trial.node_count() {
                    let finite_difference =
                        (plus_acceleration[row_node] - minus_acceleration[row_node]) / (2.0 * step);
                    assert_component_close(analytic[row_node].x, finite_difference.x);
                    assert_component_close(analytic[row_node].y, finite_difference.y);
                }
            }
        }
    }

    #[test]
    fn first_order_sls_jacobian_matches_central_differences() {
        let config = SimulationConfig {
            segment_count: 3,
            rope_model: RopeModelKind::StandardLinearSolid,
            axial_viscosity: 900.0,
            transient_axial_rigidity: 18_000.0,
            ..SimulationConfig::default()
        };
        let rest_length = config.rope_length / config.segment_count as f64;
        let masses = test_masses(config);
        let mut state = test_state();
        state.velocities = vec![
            Vec2::ZERO,
            Vec2::new(0.3, -0.2),
            Vec2::new(-0.4, 0.1),
            Vec2::new(0.2, 0.5),
        ];
        state.material_state = vec![120.0, -35.0, 80.0];

        let system = RopeDynamics::new(&config, &masses, rest_length, None, None, 0.0);
        let mut jacobian = BlockTridiagonalMatrix::new(state.node_count());
        system.first_order_jacobian(&state, &mut jacobian);

        for column_node in 0..state.node_count() {
            let component_count = if column_node < state.material_state.len() {
                5
            } else {
                4
            };
            for column_component in 0..component_count {
                let step = 1.0e-6;
                let mut plus = state.clone();
                let mut minus = state.clone();
                add_state_component(&mut plus, column_node, column_component, step);
                add_state_component(&mut minus, column_node, column_component, -step);
                let plus_derivative = first_order_derivative(&system, &plus);
                let minus_derivative = first_order_derivative(&system, &minus);

                for row_node in 0..state.node_count() {
                    for row_component in 0..5 {
                        let finite_difference = (plus_derivative[row_node][row_component]
                            - minus_derivative[row_node][row_component])
                            / (2.0 * step);
                        assert_component_close(
                            block_value(
                                &jacobian,
                                row_node,
                                column_node,
                                row_component,
                                column_component,
                            ),
                            finite_difference,
                        );
                    }
                }
            }
        }
    }

    fn test_state() -> State {
        State::new(vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(0.2, -3.8),
            Vec2::new(-0.1, -8.1),
            Vec2::new(0.4, -12.3),
        ])
    }

    fn test_masses(config: SimulationConfig) -> Vec<f64> {
        let node_count = config.segment_count + 1;
        let element_mass = config.rope_mass / config.segment_count as f64;
        let mut masses = vec![element_mass; node_count];
        masses[0] = 0.5 * element_mass;
        masses[node_count - 1] = 0.5 * element_mass + config.payload_mass;
        masses
    }

    fn add_component(vector: &mut Vec2, component: usize, amount: f64) {
        if component == 0 {
            vector.x += amount;
        } else {
            vector.y += amount;
        }
    }

    fn add_state_component(state: &mut State, node: usize, component: usize, amount: f64) {
        match component {
            0 | 1 => add_component(&mut state.positions[node], component, amount),
            2 | 3 => add_component(&mut state.velocities[node], component - 2, amount),
            4 => state.material_state[node] += amount,
            _ => unreachable!(),
        }
    }

    fn first_order_derivative(system: &RopeDynamics<'_>, state: &State) -> Vec<[f64; 5]> {
        let mut accelerations = vec![Vec2::ZERO; state.node_count()];
        let mut material_derivatives = vec![0.0; state.material_state.len()];
        system.accelerations(state, &mut accelerations);
        system.material_state_derivatives(state, &mut material_derivatives);
        (0..state.node_count())
            .map(|node| {
                let mut derivative = [0.0; 5];
                if system.is_dynamic_node(node) {
                    derivative[0] = state.velocities[node].x;
                    derivative[1] = state.velocities[node].y;
                    derivative[2] = accelerations[node].x;
                    derivative[3] = accelerations[node].y;
                }
                if node < material_derivatives.len() {
                    derivative[4] = material_derivatives[node];
                }
                derivative
            })
            .collect()
    }

    fn block_value(
        matrix: &BlockTridiagonalMatrix,
        row_block: usize,
        column_block: usize,
        row: usize,
        column: usize,
    ) -> f64 {
        if row_block == column_block {
            matrix.diagonal[row_block][row][column]
        } else if row_block == column_block + 1 {
            matrix.lower[column_block][row][column]
        } else if column_block == row_block + 1 {
            matrix.upper[row_block][row][column]
        } else {
            0.0
        }
    }

    fn assert_component_close(analytic: f64, finite_difference: f64) {
        let scale = 1.0_f64.max(analytic.abs()).max(finite_difference.abs());
        assert!(
            (analytic - finite_difference).abs() <= 2.0e-6 * scale,
            "analytic {analytic:e} differs from finite difference {finite_difference:e}"
        );
    }
}

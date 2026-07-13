use crate::math::Vec2;

use super::{AccelerationJacobianBlock, StepError};

const NOT_DYNAMIC: usize = usize::MAX;
type Matrix2 = [[f64; 2]; 2];
const ZERO_MATRIX2: Matrix2 = [[0.0; 2]; 2];

/// Solves the position-only Newton systems produced by the implicit integrators.
///
/// Each dynamic rope node contributes a 2x2 block. Axial elements only couple
/// neighboring nodes, so the resulting matrix is block tridiagonal even when
/// either endpoint is prescribed.
pub(super) struct NewtonBlockTridiagonalSolver {
    lower: Vec<Matrix2>,
    diagonal: Vec<Matrix2>,
    reduced_upper: Vec<Matrix2>,
    right_hand_side: Vec<Vec2>,
}

impl NewtonBlockTridiagonalSolver {
    pub fn new(block_count: usize) -> Self {
        Self {
            lower: vec![ZERO_MATRIX2; block_count.saturating_sub(1)],
            diagonal: vec![ZERO_MATRIX2; block_count],
            reduced_upper: vec![ZERO_MATRIX2; block_count.saturating_sub(1)],
            right_hand_side: vec![Vec2::ZERO; block_count],
        }
    }

    pub fn solve(
        &mut self,
        acceleration_blocks: &[AccelerationJacobianBlock],
        node_to_unknown: &[usize],
        residual: &[f64],
        dt: f64,
        output: &mut [f64],
    ) -> Result<(), StepError> {
        let block_count = residual.len() / 2;
        self.lower
            .resize(block_count.saturating_sub(1), ZERO_MATRIX2);
        self.diagonal.resize(block_count, ZERO_MATRIX2);
        self.reduced_upper
            .resize(block_count.saturating_sub(1), ZERO_MATRIX2);
        self.right_hand_side.resize(block_count, Vec2::ZERO);
        self.lower.fill(ZERO_MATRIX2);
        self.diagonal.fill(ZERO_MATRIX2);
        self.reduced_upper.fill(ZERO_MATRIX2);

        for (block, diagonal) in self.diagonal.iter_mut().enumerate() {
            diagonal[0][0] = 1.0;
            diagonal[1][1] = 1.0;
            self.right_hand_side[block] = Vec2::new(-residual[2 * block], -residual[2 * block + 1]);
        }

        let dt_squared = dt * dt;
        for block in acceleration_blocks {
            let row_offset = node_to_unknown[block.row_node];
            let column_offset = node_to_unknown[block.column_node];
            if row_offset == NOT_DYNAMIC || column_offset == NOT_DYNAMIC {
                continue;
            }

            let row_block = row_offset / 2;
            let column_block = column_offset / 2;
            let target = if row_block == column_block {
                &mut self.diagonal[row_block]
            } else if row_block == column_block + 1 {
                &mut self.lower[column_block]
            } else if column_block == row_block + 1 {
                &mut self.reduced_upper[row_block]
            } else {
                return Err(StepError::SingularJacobian);
            };

            for (row, target_row) in target.iter_mut().enumerate() {
                for (column, target_value) in target_row.iter_mut().enumerate() {
                    *target_value += -dt_squared * block.position[row][column]
                        - dt * block.velocity[row][column];
                }
            }
        }

        // reduced_upper initially stores the original upper blocks. It is
        // overwritten in place with D_i^-1 U_i during elimination.
        for index in 0..block_count {
            if index > 0 {
                self.diagonal[index] = subtract_matrix_product(
                    self.diagonal[index],
                    self.lower[index - 1],
                    self.reduced_upper[index - 1],
                );
                let previous = self.right_hand_side[index - 1];
                self.right_hand_side[index] -=
                    multiply_matrix_vector(self.lower[index - 1], previous);
            }
            self.right_hand_side[index] =
                solve_matrix2(self.diagonal[index], self.right_hand_side[index])?;
            if index + 1 < block_count {
                self.reduced_upper[index] =
                    solve_matrix2_columns(self.diagonal[index], self.reduced_upper[index])?;
            }
        }

        for index in (0..block_count.saturating_sub(1)).rev() {
            let next = self.right_hand_side[index + 1];
            self.right_hand_side[index] -= multiply_matrix_vector(self.reduced_upper[index], next);
        }
        for (block, solution) in self.right_hand_side.iter().enumerate() {
            output[2 * block] = solution.x;
            output[2 * block + 1] = solution.y;
        }

        if output.iter().all(|value| value.is_finite()) {
            Ok(())
        } else {
            Err(StepError::SingularJacobian)
        }
    }
}

fn solve_matrix2(matrix: Matrix2, rhs: Vec2) -> Result<Vec2, StepError> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let scale = matrix
        .iter()
        .flatten()
        .fold(1.0_f64, |maximum, value| maximum.max(value.abs()));
    if !determinant.is_finite() || determinant.abs() <= 64.0 * f64::EPSILON * scale * scale {
        return Err(StepError::SingularJacobian);
    }
    Ok(Vec2::new(
        (matrix[1][1] * rhs.x - matrix[0][1] * rhs.y) / determinant,
        (-matrix[1][0] * rhs.x + matrix[0][0] * rhs.y) / determinant,
    ))
}

fn solve_matrix2_columns(matrix: Matrix2, rhs: Matrix2) -> Result<Matrix2, StepError> {
    let first = solve_matrix2(matrix, Vec2::new(rhs[0][0], rhs[1][0]))?;
    let second = solve_matrix2(matrix, Vec2::new(rhs[0][1], rhs[1][1]))?;
    Ok([[first.x, second.x], [first.y, second.y]])
}

fn multiply_matrix_vector(matrix: Matrix2, vector: Vec2) -> Vec2 {
    Vec2::new(
        matrix[0][0] * vector.x + matrix[0][1] * vector.y,
        matrix[1][0] * vector.x + matrix[1][1] * vector.y,
    )
}

fn subtract_matrix_product(base: Matrix2, left: Matrix2, right: Matrix2) -> Matrix2 {
    let mut output = base;
    for row in 0..2 {
        for column in 0..2 {
            output[row][column] -=
                left[row][0] * right[0][column] + left[row][1] * right[1][column];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_recovers_a_known_solution() {
        let diagonal = [[4.0, 0.2], [-0.1, 3.0]];
        let lower = [[-0.4, 0.1], [0.05, -0.3]];
        let upper = [[-0.2, -0.05], [0.1, -0.25]];
        let expected = [
            Vec2::new(0.5, -0.25),
            Vec2::new(1.0, 0.75),
            Vec2::new(-0.4, 1.2),
        ];
        let mut residual = vec![0.0; 2 * expected.len()];
        for row in 0..expected.len() {
            let mut value = multiply_matrix_vector(diagonal, expected[row]);
            if row > 0 {
                value += multiply_matrix_vector(lower, expected[row - 1]);
            }
            if row + 1 < expected.len() {
                value += multiply_matrix_vector(upper, expected[row + 1]);
            }
            residual[2 * row] = -value.x;
            residual[2 * row + 1] = -value.y;
        }

        let identity = [[1.0, 0.0], [0.0, 1.0]];
        let mut acceleration_blocks = Vec::new();
        for node in 0..expected.len() {
            acceleration_blocks.push(AccelerationJacobianBlock {
                row_node: node,
                column_node: node,
                position: subtract_matrix(identity, diagonal),
                velocity: ZERO_MATRIX2,
            });
            if node + 1 < expected.len() {
                acceleration_blocks.push(AccelerationJacobianBlock {
                    row_node: node,
                    column_node: node + 1,
                    position: negate_matrix(upper),
                    velocity: ZERO_MATRIX2,
                });
                acceleration_blocks.push(AccelerationJacobianBlock {
                    row_node: node + 1,
                    column_node: node,
                    position: negate_matrix(lower),
                    velocity: ZERO_MATRIX2,
                });
            }
        }

        let node_to_unknown = [0, 2, 4];
        let mut actual = vec![0.0; 2 * expected.len()];
        NewtonBlockTridiagonalSolver::new(expected.len())
            .solve(
                &acceleration_blocks,
                &node_to_unknown,
                &residual,
                1.0,
                &mut actual,
            )
            .unwrap();

        for (block, expected) in expected.iter().enumerate() {
            assert!((actual[2 * block] - expected.x).abs() < 1.0e-12);
            assert!((actual[2 * block + 1] - expected.y).abs() < 1.0e-12);
        }
    }

    fn subtract_matrix(left: Matrix2, right: Matrix2) -> Matrix2 {
        let mut output = ZERO_MATRIX2;
        for row in 0..2 {
            for column in 0..2 {
                output[row][column] = left[row][column] - right[row][column];
            }
        }
        output
    }

    fn negate_matrix(matrix: Matrix2) -> Matrix2 {
        let mut output = ZERO_MATRIX2;
        for row in 0..2 {
            for column in 0..2 {
                output[row][column] = -matrix[row][column];
            }
        }
        output
    }
}

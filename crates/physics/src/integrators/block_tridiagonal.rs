use super::StepError;

pub(crate) const BLOCK_SIZE: usize = 5;
pub(crate) type Block = [[f64; BLOCK_SIZE]; BLOCK_SIZE];
pub(crate) type BlockVector = [f64; BLOCK_SIZE];

const ZERO_BLOCK: Block = [[0.0; BLOCK_SIZE]; BLOCK_SIZE];

#[derive(Clone, Debug)]
pub(crate) struct BlockTridiagonalMatrix {
    pub lower: Vec<Block>,
    pub diagonal: Vec<Block>,
    pub upper: Vec<Block>,
}

impl BlockTridiagonalMatrix {
    pub fn new(block_count: usize) -> Self {
        Self {
            lower: vec![ZERO_BLOCK; block_count.saturating_sub(1)],
            diagonal: vec![ZERO_BLOCK; block_count],
            upper: vec![ZERO_BLOCK; block_count.saturating_sub(1)],
        }
    }

    pub fn resize_and_clear(&mut self, block_count: usize) {
        self.lower.resize(block_count.saturating_sub(1), ZERO_BLOCK);
        self.diagonal.resize(block_count, ZERO_BLOCK);
        self.upper.resize(block_count.saturating_sub(1), ZERO_BLOCK);
        self.lower.fill(ZERO_BLOCK);
        self.diagonal.fill(ZERO_BLOCK);
        self.upper.fill(ZERO_BLOCK);
    }

    pub fn add_value(
        &mut self,
        row_block: usize,
        column_block: usize,
        row: usize,
        column: usize,
        value: f64,
    ) {
        let target = if row_block == column_block {
            &mut self.diagonal[row_block]
        } else if row_block == column_block + 1 {
            &mut self.lower[column_block]
        } else if column_block == row_block + 1 {
            &mut self.upper[row_block]
        } else {
            panic!("block entry lies outside the tridiagonal band");
        };
        target[row][column] += value;
    }

    pub fn shift_and_scale(&mut self, scale: f64) {
        for block in self
            .lower
            .iter_mut()
            .chain(&mut self.diagonal)
            .chain(&mut self.upper)
        {
            for row in block {
                for value in row {
                    *value *= scale;
                }
            }
        }
        for block in &mut self.diagonal {
            for (index, row) in block.iter_mut().enumerate() {
                row[index] += 1.0;
            }
        }
    }
}

pub(crate) struct BlockThomasSolver {
    lower: Vec<Block>,
    diagonal: Vec<SmallLu>,
    reduced_upper: Vec<Block>,
}

impl BlockThomasSolver {
    pub fn new(block_count: usize) -> Self {
        Self {
            lower: vec![ZERO_BLOCK; block_count.saturating_sub(1)],
            diagonal: vec![SmallLu::identity(); block_count],
            reduced_upper: vec![ZERO_BLOCK; block_count.saturating_sub(1)],
        }
    }

    pub fn factorize(&mut self, matrix: &BlockTridiagonalMatrix) -> Result<(), StepError> {
        let block_count = matrix.diagonal.len();
        self.lower.clone_from(&matrix.lower);
        self.diagonal.resize(block_count, SmallLu::identity());
        self.reduced_upper
            .resize(block_count.saturating_sub(1), ZERO_BLOCK);

        for index in 0..block_count {
            let mut diagonal = matrix.diagonal[index];
            if index > 0 {
                subtract_product(
                    &mut diagonal,
                    &self.lower[index - 1],
                    &self.reduced_upper[index - 1],
                );
            }

            self.diagonal[index] = SmallLu::factor(diagonal)?;
            if index + 1 < block_count {
                self.reduced_upper[index] = self.diagonal[index].solve_block(matrix.upper[index]);
            }
        }
        Ok(())
    }

    pub fn solve_in_place(&self, right_hand_side: &mut [BlockVector]) {
        for index in 0..right_hand_side.len() {
            if index > 0 {
                let correction =
                    multiply_vector(&self.lower[index - 1], right_hand_side[index - 1]);
                for component in 0..BLOCK_SIZE {
                    right_hand_side[index][component] -= correction[component];
                }
            }
            right_hand_side[index] = self.diagonal[index].solve_vector(right_hand_side[index]);
        }

        for index in (0..right_hand_side.len().saturating_sub(1)).rev() {
            let correction =
                multiply_vector(&self.reduced_upper[index], right_hand_side[index + 1]);
            for component in 0..BLOCK_SIZE {
                right_hand_side[index][component] -= correction[component];
            }
        }
    }
}

#[derive(Clone, Copy)]
struct SmallLu {
    factors: Block,
    pivots: [usize; BLOCK_SIZE],
}

impl SmallLu {
    fn identity() -> Self {
        let mut factors = ZERO_BLOCK;
        for (index, row) in factors.iter_mut().enumerate() {
            row[index] = 1.0;
        }
        Self {
            factors,
            pivots: [0, 1, 2, 3, 4],
        }
    }

    fn factor(mut factors: Block) -> Result<Self, StepError> {
        let scale = factors
            .iter()
            .flatten()
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        let threshold = 64.0 * f64::EPSILON * scale.max(1.0);
        let mut pivots = [0; BLOCK_SIZE];

        for column in 0..BLOCK_SIZE {
            let mut pivot = column;
            let mut pivot_magnitude = factors[column][column].abs();
            for (row, values) in factors.iter().enumerate().skip(column + 1) {
                if values[column].abs() > pivot_magnitude {
                    pivot = row;
                    pivot_magnitude = values[column].abs();
                }
            }
            if !pivot_magnitude.is_finite() || pivot_magnitude <= threshold {
                return Err(StepError::SingularJacobian);
            }
            pivots[column] = pivot;
            factors.swap(column, pivot);

            for row in (column + 1)..BLOCK_SIZE {
                factors[row][column] /= factors[column][column];
                for trailing in (column + 1)..BLOCK_SIZE {
                    factors[row][trailing] -= factors[row][column] * factors[column][trailing];
                }
            }
        }

        Ok(Self { factors, pivots })
    }

    fn solve_vector(self, mut rhs: BlockVector) -> BlockVector {
        for column in 0..BLOCK_SIZE {
            rhs.swap(column, self.pivots[column]);
        }
        for row in 0..BLOCK_SIZE {
            for column in 0..row {
                rhs[row] -= self.factors[row][column] * rhs[column];
            }
        }
        for row in (0..BLOCK_SIZE).rev() {
            for column in (row + 1)..BLOCK_SIZE {
                rhs[row] -= self.factors[row][column] * rhs[column];
            }
            rhs[row] /= self.factors[row][row];
        }
        rhs
    }

    fn solve_block(self, rhs: Block) -> Block {
        let mut solution = ZERO_BLOCK;
        for column in 0..BLOCK_SIZE {
            let mut rhs_column = [0.0; BLOCK_SIZE];
            for row in 0..BLOCK_SIZE {
                rhs_column[row] = rhs[row][column];
            }
            let solved = self.solve_vector(rhs_column);
            for row in 0..BLOCK_SIZE {
                solution[row][column] = solved[row];
            }
        }
        solution
    }
}

fn subtract_product(output: &mut Block, left: &Block, right: &Block) {
    for row in 0..BLOCK_SIZE {
        for column in 0..BLOCK_SIZE {
            let mut value = 0.0;
            for inner in 0..BLOCK_SIZE {
                value += left[row][inner] * right[inner][column];
            }
            output[row][column] -= value;
        }
    }
}

fn multiply_vector(matrix: &Block, vector: BlockVector) -> BlockVector {
    let mut output = [0.0; BLOCK_SIZE];
    for row in 0..BLOCK_SIZE {
        for column in 0..BLOCK_SIZE {
            output[row] += matrix[row][column] * vector[column];
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_thomas_recovers_known_solution() {
        let block_count = 7;
        let mut matrix = BlockTridiagonalMatrix::new(block_count);
        for block in 0..block_count {
            for component in 0..BLOCK_SIZE {
                matrix.diagonal[block][component][component] = 4.0 + component as f64;
                if component + 1 < BLOCK_SIZE {
                    matrix.diagonal[block][component][component + 1] = 0.1;
                    matrix.diagonal[block][component + 1][component] = -0.15;
                }
            }
            if block + 1 < block_count {
                for component in 0..BLOCK_SIZE {
                    matrix.upper[block][component][component] = 0.2;
                    matrix.lower[block][component][component] = -0.25;
                }
            }
        }

        let expected: Vec<BlockVector> = (0..block_count)
            .map(|block| {
                std::array::from_fn(|component| 0.25 * (1 + block * BLOCK_SIZE + component) as f64)
            })
            .collect();
        let mut rhs = multiply_matrix(&matrix, &expected);
        let mut solver = BlockThomasSolver::new(block_count);
        solver.factorize(&matrix).unwrap();
        solver.solve_in_place(&mut rhs);

        for (actual, expected) in rhs.iter().zip(expected) {
            for component in 0..BLOCK_SIZE {
                assert!((actual[component] - expected[component]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn singular_diagonal_block_is_reported() {
        let matrix = BlockTridiagonalMatrix::new(2);
        let mut solver = BlockThomasSolver::new(2);
        assert_eq!(solver.factorize(&matrix), Err(StepError::SingularJacobian));
    }

    #[test]
    fn small_block_factorization_uses_row_pivoting() {
        let mut matrix = BlockTridiagonalMatrix::new(1);
        for component in 0..BLOCK_SIZE {
            matrix.diagonal[0][component][component] = 1.0;
        }
        matrix.diagonal[0][0][0] = 0.0;
        matrix.diagonal[0][0][1] = 1.0;
        matrix.diagonal[0][1][0] = 1.0;
        matrix.diagonal[0][1][1] = 1.0;
        let expected = [[2.0, -1.0, 0.5, 3.0, -2.0]];
        let mut rhs = multiply_matrix(&matrix, &expected);

        let mut solver = BlockThomasSolver::new(1);
        solver.factorize(&matrix).unwrap();
        solver.solve_in_place(&mut rhs);

        for component in 0..BLOCK_SIZE {
            assert!((rhs[0][component] - expected[0][component]).abs() < 1.0e-12);
        }
    }

    fn multiply_matrix(
        matrix: &BlockTridiagonalMatrix,
        vector: &[BlockVector],
    ) -> Vec<BlockVector> {
        let mut output = vec![[0.0; BLOCK_SIZE]; vector.len()];
        for index in 0..vector.len() {
            output[index] = multiply_vector(&matrix.diagonal[index], vector[index]);
            if index > 0 {
                let value = multiply_vector(&matrix.lower[index - 1], vector[index - 1]);
                for component in 0..BLOCK_SIZE {
                    output[index][component] += value[component];
                }
            }
            if index + 1 < vector.len() {
                let value = multiply_vector(&matrix.upper[index], vector[index + 1]);
                for component in 0..BLOCK_SIZE {
                    output[index][component] += value[component];
                }
            }
        }
        output
    }
}

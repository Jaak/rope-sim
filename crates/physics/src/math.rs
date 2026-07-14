use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Row-major 2x2 matrix used by planar force tangents and block solvers.
pub(crate) type Matrix2 = [[f64; 2]; 2];

pub(crate) const ZERO_MATRIX2: Matrix2 = [[0.0; 2]; 2];

/// A double-precision vector in simulation (SI) coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub const ZERO: Self = Self::new(0.0, 0.0);

    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
    }

    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

impl Add for Vec2 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2> for f64 {
    type Output = Vec2;

    #[inline]
    fn mul(self, rhs: Vec2) -> Self::Output {
        rhs * self
    }
}

impl MulAssign<f64> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl Div<f64> for Vec2 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl Neg for Vec2 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y)
    }
}

/// Scalar z-component of the 3D cross product of two planar vectors.
#[inline]
pub(crate) fn cross(left: Vec2, right: Vec2) -> f64 {
    left.x * right.y - left.y * right.x
}

/// Gradient of the polar angle `atan2(edge.y, edge.x)` with respect to `edge`.
///
/// `length_squared` must be the nonzero squared length of `edge`. Accepting it
/// separately lets geometry callers reuse a value they already validated.
#[inline]
pub(crate) fn angle_gradient(edge: Vec2, length_squared: f64) -> Vec2 {
    Vec2::new(-edge.y / length_squared, edge.x / length_squared)
}

/// Hessian of the polar angle `atan2(edge.y, edge.x)` with respect to `edge`.
///
/// `length_squared` must be the nonzero squared length of `edge`. The returned
/// matrix uses the row-major convention shared by the rest of this module.
#[inline]
pub(crate) fn angle_hessian(edge: Vec2, length_squared: f64) -> Matrix2 {
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

#[inline]
pub(crate) fn matrix2_outer_product(left: Vec2, right: Vec2) -> Matrix2 {
    [
        [left.x * right.x, left.x * right.y],
        [left.y * right.x, left.y * right.y],
    ]
}

#[inline]
pub(crate) fn matrix2_vector_product(matrix: Matrix2, vector: Vec2) -> Vec2 {
    Vec2::new(
        matrix[0][0] * vector.x + matrix[0][1] * vector.y,
        matrix[1][0] * vector.x + matrix[1][1] * vector.y,
    )
}

#[inline]
pub(crate) fn matrix2_scale(matrix: Matrix2, scale: f64) -> Matrix2 {
    matrix.map(|row| row.map(|value| value * scale))
}

#[inline]
pub(crate) fn matrix2_add(left: Matrix2, right: Matrix2) -> Matrix2 {
    std::array::from_fn(|row| std::array::from_fn(|column| left[row][column] + right[row][column]))
}

#[inline]
pub(crate) fn matrix2_add_scaled(output: &mut Matrix2, input: Matrix2, scale: f64) {
    for row in 0..2 {
        for column in 0..2 {
            output[row][column] += scale * input[row][column];
        }
    }
}

#[inline]
pub(crate) fn matrix2_subtract_product(base: Matrix2, left: Matrix2, right: Matrix2) -> Matrix2 {
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
    fn matrix2_helpers_follow_row_major_convention() {
        let matrix = [[2.0, -1.0], [3.0, 4.0]];
        assert_eq!(
            matrix2_vector_product(matrix, Vec2::new(5.0, 2.0)),
            Vec2::new(8.0, 23.0)
        );
        assert_eq!(
            matrix2_outer_product(Vec2::new(2.0, 3.0), Vec2::new(4.0, -1.0)),
            [[8.0, -2.0], [12.0, -3.0]]
        );
    }
}

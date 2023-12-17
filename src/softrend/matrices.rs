use std::fmt;
use std::ops::{Index, IndexMut, Mul};

pub struct Vec4([f32; 4]);
// A 4x4 matrix, stored in row-major
pub struct M4x4([f32; 16]);
// A 4xn matrix, stored in column-major
pub struct M4xn(Vec<[f32; 4]>);

impl Vec4 {
    pub fn new() -> Self {
        Self([0.0; 4])
    }
}
impl M4x4 {
    pub fn new() -> Self {
        Self([0.0; 16])
    }
    pub fn from_slice(data: [f32; 16]) -> Self {
        Self(data)
    }
    pub fn mul_packed(&self, by: &mut M4xn) {
        for in_col in by.0.iter_mut() {
            let mut temp = [0.0; 4];
            for (i, coeff_row) in self.0.chunks_exact(4).enumerate() {
                temp[i] =
                    in_col.iter().zip(coeff_row.iter()).map(|(&a, &b)| a * b).sum();
            }
            in_col.copy_from_slice(&temp);
        }
    }
}
impl M4xn {
    pub fn new(n: usize) -> Self {
        Self(vec![[0.0; 4]; n])
    }
    pub fn from_vec(input: Vec<[f32; 4]>) -> Self {
        Self(input)
    }
    pub fn n(&self) -> usize {
        self.0.len()
    }
    pub fn w_divide(&mut self) {
        for point in self.0.iter_mut() {
            let w = point[3];
            point.iter_mut().for_each(|x| *x /= w);
        }
    }
}

pub mod RenderMats {
    use super::*;

    #[rustfmt::skip]
    pub fn rot_x(theta: f32) -> M4x4 {
        M4x4::from_slice([
            1.0,  0.0,         0.0,          0.0,
            0.0,  theta.cos(), -theta.sin(), 0.0,
            0.0,  theta.sin(), theta.cos(),  0.0,
            0.0,  0.0,         0.0,          1.0
        ])
    }
    #[rustfmt::skip]
    pub fn rot_y(theta: f32) -> M4x4 {
        M4x4::from_slice([
            theta.cos(),  0.0,  theta.sin(), 0.0,
            0.0,          1.0,  0.0,         0.0,
            -theta.sin(), 0.0,  theta.cos(), 0.0,
            0.0,          0.0,  0.0,         1.0
        ])
    }
    #[rustfmt::skip]
    pub fn rot_z(theta: f32) -> M4x4 {
        M4x4::from_slice([
            theta.cos(), -theta.sin(), 0.0,  0.0,
            theta.sin(), theta.cos(),  0.0,  0.0,
            0.0,         0.0,          1.0,  0.0,
            0.0,         0.0,          0.0,  1.0
        ])
    }
    #[rustfmt::skip]
    pub fn translate(dx: f32, dy: f32, dz: f32) -> M4x4 {
        M4x4::from_slice([
            1.0, 0.0, 0.0, dx,
            0.0, 1.0, 0.0, dy,
            0.0, 0.0, 1.0, dz,
            0.0, 0.0, 0.0, 1.0
        ])
    }
    #[rustfmt::skip]
    pub fn scale(sx: f32, sy: f32, sz: f32) -> M4x4 {
        M4x4::from_slice([
            sx,  0.0, 0.0, 0.0,
            0.0, sy,  0.0, 0.0,
            0.0, 0.0, sz,  0.0,
            0.0, 0.0, 0.0, 1.0
        ])
    }
    #[rustfmt::skip]
    pub fn proj(near: f32, far: f32, fov: f32) -> M4x4 {
        let half_near_width = 0.5 * near as f32 * (fov * 0.5).tan();
        let inv_hnw = half_near_width.recip();

        M4x4::from_slice([
            inv_hnw, 0.0,     0.0,              0.0,
            0.0,     inv_hnw, 0.0,              0.0,
            0.0,     0.0,     inv_hnw,          0.0,
            0.0,     0.0,     near.recip(),     1.0
        ])
    }
    #[rustfmt::skip]
    pub fn identity() -> M4x4 {
        M4x4::from_slice([
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        ])
    }
}

impl Mul<M4x4> for M4x4 {
    type Output = M4x4;
    fn mul(self, by: M4x4) -> M4x4 {
        let mut acc = M4x4::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    acc[(i, j)] += self[(i, k)] * by[(k, j)];
                }
            }
        }
        acc
    }
}
impl Mul<&Vec4> for &M4x4 {
    type Output = Vec4;
    fn mul(self, by: &Vec4) -> Vec4 {
        let mut acc = Vec4::new();
        for i in 0..4 {
            for j in 0..4 {
                acc[i] += self[(i, j)] * by[j];
            }
        }
        acc
    }
}

impl Index<usize> for Vec4 {
    type Output = f32;
    fn index(&self, index: usize) -> &f32 {
        &self.0[index]
    }
}
impl IndexMut<usize> for Vec4 {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.0[index]
    }
}

impl Index<(usize, usize)> for M4x4 {
    type Output = f32;
    fn index(&self, index: (usize, usize)) -> &f32 {
        unsafe { self.0.get_unchecked(index.0 * 4 + index.1) }
    }
}
impl IndexMut<(usize, usize)> for M4x4 {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut f32 {
        unsafe { self.0.get_unchecked_mut(index.0 * 4 + index.1) }
    }
}
impl Index<usize> for M4xn {
    type Output = [f32; 4];
    fn index(&self, index: usize) -> &[f32; 4] {
        unsafe { self.0.get_unchecked(index) }
    }
}
impl IndexMut<usize> for M4xn {
    fn index_mut(&mut self, index: usize) -> &mut [f32; 4] {
        unsafe { self.0.get_unchecked_mut(index) }
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:?}>", self.0)
    }
}
impl fmt::Display for M4x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for row in self.0.chunks_exact(4) {
            write!(f, "\n    {:?}", row)?;
        }
        write!(f, "\n]")
    }
}
impl fmt::Display for M4xn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for row in self.0.iter() {
            write!(f, "\n    {:?}", row)?;
        }
        write!(f, "\n]")
    }
}

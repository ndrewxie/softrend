use std::cmp::{Ord, Ordering, PartialOrd};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Fx<const SCALE: usize>(i32);
impl<const SCALE: usize> Fx<SCALE> {
    pub fn new(val: i32) -> Self {
        Self(val * Self::scale())
    }
    pub fn from_f32(val: f32) -> Self {
        let precision = 1.0 / Self::scale() as f32;
        Self((val / precision) as i32)
    }
    pub fn as_i32(self) -> i32 {
        self.0 / Self::scale()
    }
    pub fn as_f32(self) -> f32 {
        (self.0 as f32) / (Self::scale() as f32)
    }
    pub fn trunc(self) -> Self {
        Self((self.0 / Self::scale()) * Self::scale())
    }
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
    pub fn is_neg(self) -> bool {
        self.0 < 0
    }
    pub fn is_pos(self) -> bool {
        self.0 > 0
    }
    pub fn scale() -> i32 {
        1 << SCALE
    }
}
impl<const SCALE: usize> Add<Fx<SCALE>> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}
impl<const SCALE: usize> Add<i32> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn add(self, other: i32) -> Self {
        Self(self.0 + other * Self::scale())
    }
}
impl<const SCALE: usize> Sub<Fx<SCALE>> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}
impl<const SCALE: usize> Sub<i32> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn sub(self, other: i32) -> Self {
        Self(self.0 - other * Self::scale())
    }
}
impl<const SCALE: usize> Mul<Fx<SCALE>> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn mul(self, other: Self) -> Self {
        Self(((self.0 as i64 * other.0 as i64) / Self::scale() as i64) as i32)
    }
}
impl<const SCALE: usize> Mul<i32> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn mul(self, other: i32) -> Self {
        Self(self.0 * other)
    }
}
impl<const SCALE: usize> Div<Fx<SCALE>> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn div(self, other: Self) -> Self {
        Self(((self.0 as i64 * Self::scale() as i64) / other.0 as i64) as i32)
    }
}
impl<const SCALE: usize> Div<i32> for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn div(self, other: i32) -> Self {
        Self(self.0 / other)
    }
}
impl<const SCALE: usize> Neg for Fx<SCALE> {
    type Output = Fx<SCALE>;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
impl<const SCALE: usize> From<i32> for Fx<SCALE> {
    fn from(input: i32) -> Self {
        Self(input * Self::scale())
    }
}
impl<const SCALE: usize> From<f32> for Fx<SCALE> {
    fn from(input: f32) -> Self {
        Self((input * Self::scale() as f32) as i32)
    }
}
impl<const SCALE: usize> PartialOrd for Fx<SCALE> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<const SCALE: usize> Ord for Fx<SCALE> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}
impl<const SCALE: usize> std::fmt::Display for Fx<SCALE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fixed({},<{}>)", self.as_f32(), self.0)
    }
}
impl<const SCALE: usize> std::fmt::Debug for Fx<SCALE> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Fixed({},<{}>)", self.as_f32(), self.0)
    }
}
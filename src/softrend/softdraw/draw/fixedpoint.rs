use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};
use std::ops::{Index, IndexMut};
use std::simd::{prelude::*, LaneCount, SupportedLaneCount};

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Fx32<const SCALE: usize>(i32);
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Fx32Simd<const N: usize, const SCALE: usize>(Simd<i32, N>)
where
    LaneCount<N>: SupportedLaneCount;

impl<const SCALE: usize> Fx32<SCALE> {
    pub fn scale() -> i32 {
        1 << SCALE
    }
    pub fn from_float(input: f32) -> Self {
        let float_scale = Self::scale() as f32;
        unsafe { Self((input * float_scale).to_int_unchecked::<i32>()) }
    }
    pub fn to_float(self) -> f32 {
        let recip_scale = (Self::scale() as f32).recip();
        self.0 as f32 * recip_scale
    }
}
impl<const SCALE: usize> PartialOrd for Fx32<SCALE> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

impl<const N: usize, const SCALE: usize> Fx32Simd<N, SCALE>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn scale() -> Simd<i32, N> {
        Simd::<i32, N>::splat(1 << SCALE)
    }
    pub fn from_array(input: [Fx32<SCALE>; N]) -> Self {
        let inner: [i32; N] = std::array::from_fn(|i| input[i].0);
        Self(Simd::<i32, N>::from_array(inner))
    }
    pub fn from_float(input: Simd<f32, N>) -> Self {
        let float_scale = Self::scale().cast::<f32>();
        unsafe { Self((input * float_scale).to_int_unchecked::<i32>()) }
    }
    pub fn splat_float(input: f32) -> Self {
        Self::from_array([Fx32::<SCALE>::from_float(input); N])
    }
    pub fn to_float(self) -> Simd<f32, N> {
        let recip_scale = Self::scale().cast::<f32>().recip();
        self.0.cast::<f32>() * recip_scale
    }
    pub fn ge(self, other: Self) -> Mask<i32, N> {
        self.0.simd_ge(other.0)
    }
    pub fn gt(self, other: Self) -> Mask<i32, N> {
        self.0.simd_gt(other.0)
    }
    pub fn le(self, other: Self) -> Mask<i32, N> {
        self.0.simd_le(other.0)
    }
    pub fn lt(self, other: Self) -> Mask<i32, N> {
        self.0.simd_lt(other.0)
    }
}
impl<const N: usize, const SCALE: usize> Index<usize> for Fx32Simd<N, SCALE>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Fx32<SCALE>;
    fn index(&self, index: usize) -> &Self::Output {
        // Reinterprets the &i32 as an &Fx32<SCALE>
        // Works because the #[repr(transparent)]
        unsafe { &*(&self.0[index] as *const i32 as *const Fx32<SCALE>) }
    }
}
impl<const N: usize, const SCALE: usize> IndexMut<usize> for Fx32Simd<N, SCALE>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        // Reinterprets the &mut i32 as an &mut Fx32<SCALE>
        // Works because the #[repr(transparent)]
        let val = &mut self.0[index];
        let casted = val as *mut i32 as *mut Fx32<SCALE>;
        unsafe { &mut *casted }
    }
}

macro_rules! impl_simple_op_scalar {
    ($optrait:ident, $opfn:ident, $opsym:tt, $inctrait:ident, $incfn:ident, $incsym:tt) => {
        impl<const SCALE: usize> $optrait<Fx32<SCALE>> for Fx32<SCALE> {
            type Output = Fx32<SCALE>;
            fn $opfn(self, other: Fx32<SCALE>) -> Self::Output {
                Self(self.0 $opsym other.0)
            }
        }
        impl<const SCALE: usize> $inctrait<Fx32<SCALE>> for Fx32<SCALE> {
            fn $incfn(&mut self, other: Fx32<SCALE>) {
                self.0 $incsym other.0
            }
        }
    }
}
macro_rules! impl_simple_op_simd {
    ($optrait:ident, $opfn:ident, $opsym:tt, $inctrait:ident, $incfn:ident, $incsym:tt) => {
        impl<const N: usize, const SCALE: usize> $optrait<Fx32Simd<N, SCALE>> for Fx32Simd<N, SCALE> where LaneCount<N>: SupportedLaneCount {
            type Output = Fx32Simd<N, SCALE>;
            fn $opfn(self, other: Fx32Simd<N, SCALE>) -> Self::Output {
                Self(self.0 $opsym other.0)
            }
        }
        impl<const N: usize, const SCALE: usize> $inctrait<Fx32Simd<N, SCALE>> for Fx32Simd<N, SCALE> where LaneCount<N>: SupportedLaneCount {
            fn $incfn(&mut self, other: Fx32Simd<N, SCALE>) where LaneCount<N>: SupportedLaneCount {
                self.0 $incsym other.0
            }
        }
    }
}
impl<const SCALE: usize> Mul<Fx32<SCALE>> for Fx32<SCALE> {
    type Output = Fx32<SCALE>;
    fn mul(self, other: Fx32<SCALE>) -> Self::Output {
        Self(self.0 * other.0 / Self::scale())
    }
}
impl<const SCALE: usize> MulAssign<Fx32<SCALE>> for Fx32<SCALE> {
    fn mul_assign(&mut self, other: Fx32<SCALE>) {
        self.0 *= other.0;
        self.0 /= Self::scale();
    }
}
impl<const N: usize, const SCALE: usize> Mul<Fx32Simd<N, SCALE>> for Fx32Simd<N, SCALE>
where
    LaneCount<N>: SupportedLaneCount,
{
    type Output = Fx32Simd<N, SCALE>;
    fn mul(self, other: Fx32Simd<N, SCALE>) -> Self::Output {
        Self(self.0 * other.0 / Self::scale())
    }
}
impl<const N: usize, const SCALE: usize> MulAssign<Fx32Simd<N, SCALE>> for Fx32Simd<N, SCALE>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn mul_assign(&mut self, other: Fx32Simd<N, SCALE>) {
        self.0 *= other.0;
        self.0 /= Self::scale();
    }
}
impl_simple_op_scalar!(Add, add, +, AddAssign, add_assign, +=);
impl_simple_op_scalar!(Sub, sub, -, SubAssign, sub_assign, -=);
impl_simple_op_simd!(Add, add, +, AddAssign, add_assign, +=);
impl_simple_op_simd!(Sub, sub, -, SubAssign, sub_assign, -=);

pub type Fx28_4 = Fx32<4>;
pub type Fx28_4x4 = Fx32Simd<4, 4>;
pub type Fx28_4x8 = Fx32Simd<8, 4>;

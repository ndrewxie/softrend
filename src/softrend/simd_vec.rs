use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign,
};

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug)]
pub struct SimdVec<T, const N: usize>(pub [T; N]);

macro_rules! impl_simd_traits {
    ($traitname:ident, $traitfunc: ident, $assgntrait:ident, $assgnfunc:ident, $traitop:tt, $assgnop:tt) => {
        impl<T: Copy + $assgntrait, const N: usize> $traitname<SimdVec<T, N>> for SimdVec<T, N> {
            type Output = SimdVec<T, N>;
            fn $traitfunc(mut self, other: Self) -> Self {
                self $assgnop other;
                self
            }
        }
        impl<T: Copy + $assgntrait, const N: usize> $traitname<T> for SimdVec<T, N> {
            type Output = SimdVec<T, N>;
            fn $traitfunc(mut self, other: T) -> Self {
                self $assgnop other;
                self
            }
        }
        impl<T: Copy + $assgntrait, const N: usize> $assgntrait<SimdVec<T, N>> for SimdVec<T, N> {
            fn $assgnfunc(&mut self, rhs: Self) {
                self.0.iter_mut().zip(rhs.0.into_iter()).for_each(|(x, a)| *x $assgnop a);
            }
        }
        impl<T: Copy + $assgntrait, const N: usize> $assgntrait<T> for SimdVec<T, N> {
            fn $assgnfunc(&mut self, rhs: T) {
                self.0.iter_mut().for_each(|x| *x $assgnop rhs);
            }
        }
    }
}

impl_simd_traits!(Add, add, AddAssign, add_assign, +, +=);
impl_simd_traits!(Sub, sub, SubAssign, sub_assign, -, -=);
impl_simd_traits!(Mul, mul, MulAssign, mul_assign, *, *=);
impl_simd_traits!(Div, div, DivAssign, div_assign, /, /=);

impl<T, const N: usize, Idx> Index<Idx> for SimdVec<T, N>
where
    Idx: std::slice::SliceIndex<[T]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}
impl<T, const N: usize> IndexMut<usize> for SimdVec<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut self.0[index]
    }
}

impl<T, const N: usize> SimdVec<T, N> {
    #[inline(always)]
    pub fn iter<'a>(&'a self) -> std::slice::Iter<'a, T> {
        self.0.iter()
    }
    #[inline(always)]
    pub fn iter_mut<'a>(&'a mut self) -> std::slice::IterMut<'a, T> {
        self.0.iter_mut()
    }
}

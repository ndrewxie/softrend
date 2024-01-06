use super::fixedpoint::*;
use std::simd::prelude::*;

#[derive(Clone, Debug)]
pub struct StepDelta<const FIP: usize, const FIA: usize> {
    pub edges: Fx28_4x4,
    pub persp: [f32; FIP],
    pub affine: [f32; FIA],
    pub rcp_z: f32,
}
#[derive(Clone, Debug)]
pub struct Interps<const FIP: usize, const FIA: usize> {
    pub edges: [Fx28_4x4; 3],
    pub persp: [f32x4; FIP],
    pub affine: [f32x4; FIA],
    pub rcp_z: f32x4,
}
#[derive(Clone, Debug)]
pub struct DrawCutoffs {
    pub accept: Fx28_4x4,
    pub reject: Fx28_4x4,
}
#[derive(Debug)]
pub struct SetupData<const FIP: usize, const FIA: usize> {
    pub interps: Interps<FIP, FIA>,
    pub dx: StepDelta<FIP, FIA>,
    pub dy: StepDelta<FIP, FIA>,
    pub cutoffs: DrawCutoffs,
}

impl<const FIP: usize, const FIA: usize> Interps<FIP, FIA> {
    pub fn from(
        rcp_z: f32,
        persp: [f32; FIP],
        affine: [f32; FIA],
        edges: Fx28_4x4,
        dx: &StepDelta<FIP, FIA>,
    ) -> Self {
        let step = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
        Self {
            edges: std::array::from_fn(|i| {
                Fx28_4x4::splat(edges[i])
                    + Fx28_4x4::from_float(step) * Fx28_4x4::splat(dx.edges[i])
            }),
            persp: std::array::from_fn(|i| {
                f32x4::splat(persp[i]) + step * f32x4::splat(dx.persp[i])
            }),
            affine: std::array::from_fn(|i| {
                f32x4::splat(affine[i]) + step * f32x4::splat(dx.affine[i])
            }),
            rcp_z: f32x4::splat(rcp_z) + step * f32x4::splat(dx.rcp_z),
        }
    }
    #[inline(always)]
    pub fn step_by<const SKIP_DRAW: bool>(&mut self, delta: &StepDelta<FIP, FIA>) {
        self.rcp_z += f32x4::splat(delta.rcp_z);
        for (persp, dpersp) in self.persp.iter_mut().zip(delta.persp) {
            *persp += f32x4::splat(dpersp);
        }
        for (affine, daffine) in self.affine.iter_mut().zip(delta.affine) {
            *affine += f32x4::splat(daffine);
        }
        if !SKIP_DRAW {
            for edge in 0..3 {
                self.edges[edge] += Fx28_4x4::splat(delta.edges[edge]);
            }
        }
    }
    #[inline(always)]
    pub fn persp_correct(&self, z_offset: f32) -> [f32x4; FIP] {
        let z = (self.rcp_z - f32x4::splat(z_offset)).recip();
        let mut persp = self.persp;
        persp.iter_mut().for_each(|x| *x *= z);
        persp
    }
    #[inline(always)]
    pub fn draw_mask(&self) -> mask32x4 {
        let zero = Fx28_4x4::splat_float(0.0);
        self.edges[0].ge(zero) & self.edges[1].ge(zero) & self.edges[2].ge(zero)
    }
    #[inline(always)]
    pub fn rejects(&self, cutoffs: &DrawCutoffs) -> bool {
        let cutoff = cutoffs.reject;
        let mask_a = self.edges[0].lt(Fx28_4x4::splat(cutoff[0]));
        let mask_b = self.edges[1].lt(Fx28_4x4::splat(cutoff[1]));
        let mask_c = self.edges[2].lt(Fx28_4x4::splat(cutoff[2]));
        let mask = mask_a | mask_b | mask_c;
        unsafe { mask.test_unchecked(0) }
    }
    #[inline(always)]
    pub fn accepts(&self, cutoffs: &DrawCutoffs) -> bool {
        let cutoff = cutoffs.accept;
        let mask_a = self.edges[0].ge(Fx28_4x4::splat(cutoff[0]));
        let mask_b = self.edges[1].ge(Fx28_4x4::splat(cutoff[1]));
        let mask_c = self.edges[2].ge(Fx28_4x4::splat(cutoff[2]));
        let mask = mask_a & mask_b & mask_c;
        unsafe { mask.test_unchecked(0) }
    }
}
impl DrawCutoffs {
    pub fn scale(&mut self, fac: f32) {
        self.accept *= Fx28_4x4::splat_float(fac);
        self.reject *= Fx28_4x4::splat_float(fac);
    }
}
impl<const FIP: usize, const FIA: usize> StepDelta<FIP, FIA> {
    pub fn scale(&mut self, fac: f32) {
        self.edges *= Fx28_4x4::splat_float(fac);
        self.rcp_z *= fac;
        self.persp.iter_mut().for_each(|x| *x *= fac);
        self.affine.iter_mut().for_each(|x| *x *= fac);
    }
}

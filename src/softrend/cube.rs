use std::simd::prelude::*;

use super::assets::*;
use super::matrices::*;
use super::softdraw::*;

pub struct Cube {
    model: M4x4,
    tex: &'static Texture,
    z_offset: f32,
}

impl Cube {
    pub fn new(
        loc: [f32; 3],
        orientation: [f32; 3],
        scale: [f32; 3],
        tex: &'static Texture,
        camera: M4x4,
        z_offset: f32,
    ) -> Self {
        let model = camera
            * render_mats::translate(loc[0], loc[1], loc[2])
            * render_mats::rot_x(orientation[0])
            * render_mats::rot_y(orientation[1])
            * render_mats::rot_z(orientation[2])
            * render_mats::scale(scale[0], scale[1], scale[2]);
        Self { model, tex, z_offset }
    }
    pub fn draw(self, raster: &Raster) {
        fn v4(arr: [f32; 4]) -> Vec4 {
            Vec4::from_array(arr)
        }
        let mut mesh = Mesh::new(self);
        mesh.add_vertices([
            (v4([1.0, 1.0, 1.0, 1.0]), [0.0, 127.0]),    // 0
            (v4([1.0, 1.0, -1.0, 1.0]), [127.0, 127.0]), // 1
            (v4([1.0, -1.0, 1.0, 1.0]), [127.0, 0.0]),   // 2
            (v4([1.0, -1.0, -1.0, 1.0]), [0.0, 127.0]),  // 3
            (v4([-1.0, 1.0, 1.0, 1.0]), [127.0, 127.0]), // 4
            (v4([-1.0, 1.0, -1.0, 1.0]), [127.0, 0.0]),  // 5
            (v4([-1.0, -1.0, 1.0, 1.0]), [135.0, 0.0]),  // 6
            (v4([-1.0, -1.0, -1.0, 1.0]), [0.0, 20.0]),  // 7
        ]);
        mesh.add_quads([
            [6, 2, 0, 4],
            [7, 5, 1, 3],
            [7, 6, 4, 5],
            [3, 1, 0, 2],
            [7, 3, 2, 6],
            [5, 4, 0, 1],
        ]);
        mesh.draw(raster);
    }
}

impl Shader<2, 0> for Cube {
    type VertexIn = (Vec4, [f32; 2]);
    #[inline(always)]
    fn vert(&self, input: &Self::VertexIn) -> (Vec4, VertexData<2, 0>) {
        let point = &self.model * input.0;
        (point, (input.1, []))
    }
    #[inline(always)]
    fn vd_lerp(
        &self,
        start: &VertexData<2, 0>,
        end: &VertexData<2, 0>,
        t: f32,
    ) -> VertexData<2, 0> {
        let texcoords: [f32; 2] = std::array::from_fn(|i| unsafe {
            let a = *start.0.get_unchecked(i);
            let b = *end.0.get_unchecked(i);
            a + t * (b - a)
        });
        (texcoords, [])
    }
    #[inline(always)]
    fn frag(&self, p_i: &[f32x4; 2], _a_i: &[f32x4; 0]) -> u32x4 {
        let texbuffer = unsafe {
            let pix_ptr = self.tex.0.as_slice().as_ptr() as *mut u32;
            let pix_len = self.tex.0.len() / 4;
            std::slice::from_raw_parts(pix_ptr, pix_len)
        };

        let tx_float = p_i[0].simd_clamp(f32x4::splat(0.0), f32x4::splat(127.0));
        let ty_float = p_i[1].simd_clamp(f32x4::splat(0.0), f32x4::splat(127.0));
        let tx = unsafe { tx_float.to_int_unchecked::<i32>().cast::<u32>() };
        let ty = unsafe { ty_float.to_int_unchecked::<i32>().cast::<u32>() };
        let texcoords = tx + u32x4::splat(TEX_SIZE as u32) * ty;
        let texcoords_usize = texcoords.cast::<usize>();

        let mut pix = u32x4::splat(0);
        for (texel, tc) in pix.as_mut_array().iter_mut().zip(texcoords_usize.as_array()) {
            *texel = unsafe { *texbuffer.get_unchecked(*tc) };
        }

        pix
    }
    #[inline(always)]
    fn blend(&self, frag_color: u32x4, fb_color: u32x4) -> u32x4 {
        frag_color
    }
    #[inline(always)]
    fn z_offset(&self) -> f32 {
        self.z_offset
    }
}

impl std::fmt::Debug for Cube {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cube")
    }
}

mod assets;

use std::any::TypeId;
use std::simd::prelude::*;
use std::simd::*;

const TILE_SIZES: [usize; 3] = [16, 32, 128];
const OUTER_TILE_SIZE: usize = TILE_SIZES[TILE_SIZES.len() - 1];
const OUTER_TILE_INDEX: usize = TILE_SIZES.len() - 1;
const N_TILE_SIZES: usize = TILE_SIZES.len() + 1; // 1x1 tiles

#[repr(C, align(128))]
struct MemAlign([u8; 128]);

pub struct Renderer {
    pixels: &'static mut [u8],
    z_buf: Vec<f32>,
    pub width: usize,
    pub height: usize,
    real_dims: (usize, usize),
}

#[derive(Clone, Debug)]
struct StepInfo {
    dx_draw: f32x4,
    dy_draw: f32x4,
    offset_accept: f32x4,
    offset_reject: f32x4,

    dx_fill: f32x4,
    dy_fill: f32x4,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        let width_aligned = width.next_multiple_of(OUTER_TILE_SIZE);
        let height_aligned = height.next_multiple_of(OUTER_TILE_SIZE);
        let pixels = Self::alloc_framebuffer(width_aligned, height_aligned);
        Self {
            pixels,
            z_buf: vec![0.0; width_aligned * height_aligned],
            width: width_aligned,
            height: height_aligned,
            real_dims: (width, height),
        }
    }
    /// Allocates a framebuffer, aligned to the alignment of MemAlign
    fn alloc_framebuffer(width: usize, height: usize) -> &'static mut [u8] {
        let fb_size = 4 * width * height;
        let fb = vec![0; fb_size + 3 * std::mem::size_of::<MemAlign>()];
        let fb_slice: &'static mut [u8] = Vec::leak(fb);
        unsafe {
            let (_, aligned_fb, _) = fb_slice.align_to_mut::<MemAlign>();
            std::slice::from_raw_parts_mut(
                aligned_fb.as_mut_ptr() as *mut u8,
                fb_size,
            )
        }
    }
    pub fn draw(&mut self, time: usize) -> &[u8] {
        self.pixels.fill(0);
        for _ in 0..10 {
            self.draw_tri(
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1920.0 - 1.0, 0.0, 0.0, 128.0, 0.0],
                [1920.0 - 1.0, 1080.0 - 1.0, 128.0, 128.0, 0.0],
                assets::TEXTURES.get("joemama").unwrap(),
            );
            self.draw_tri(
                [1920.0 - 1.0, 1080.0 - 1.0, 0.0, 0.0, 0.0],
                [0.0, 1080.0 - 1.0, 0.0, 128.0, 0.0],
                [0.0, 0.0, 128.0, 128.0, 0.0],
                assets::TEXTURES.get("joemama").unwrap(),
            );
        }
        &self.pixels
    }
    /// Draws the triangle `abc`, assuming vertices are in clockwise order.
    /// Points are specified with an array of 4 floats: (coord_x, coord_y, tx, ty, z)
    pub fn draw_tri(
        &mut self,
        a: [f32; 5],
        b: [f32; 5],
        c: [f32; 5],
        texture: &assets::Texture,
    ) {
        let [min_x, min_y, max_x, max_y] = Self::compute_aligned_aabb(
            [a[0], b[0], c[0]],
            [a[1], b[1], c[1]],
            self.real_dims.0,
            self.real_dims.1,
            OUTER_TILE_SIZE,
        );

        let top_left = (min_x as f32, min_y as f32);
        let (draw_interps, fill_interps, step_info) =
            StepInfo::setup_tri(&a, &b, &c, top_left);

        self.step_tile(
            (min_x, min_y),
            (max_x, max_y),
            OUTER_TILE_INDEX,
            draw_interps,
            fill_interps,
            &step_info,
            texture,
        );
    }
    fn step_tile(
        &mut self,
        tl: (usize, usize),
        br: (usize, usize),
        tsi: usize,
        mut draw_interps: f32x4,
        mut fill_interps: f32x4,
        step_info: &[StepInfo; N_TILE_SIZES],
        texture: &assets::Texture,
    ) {
        let step: &StepInfo = &step_info[tsi];
        let ts = TILE_SIZES[tsi];
        for y in (tl.1..br.1).step_by(ts) {
            let mut r_draw_interps: f32x4 = draw_interps.clone();
            let mut r_fill_interps = fill_interps.clone();
            for x in (tl.0..br.0).step_by(ts) {
                if step.rejects(r_draw_interps) {
                    r_draw_interps += step.dx_draw;
                    r_fill_interps += step.dx_fill;
                    continue;
                }
                let accepts = step.accepts(r_draw_interps);
                match (tsi, accepts) {
                    (0, true) => {
                        self.fill_inner_tile::<true>(
                            (x, y),
                            r_draw_interps,
                            r_fill_interps,
                            step_info,
                            texture,
                        );
                    }
                    (0, false) => {
                        self.fill_inner_tile::<false>(
                            (x, y),
                            r_draw_interps,
                            r_fill_interps,
                            step_info,
                            texture,
                        );
                    }
                    _ => {
                        self.step_tile(
                            (x, y),
                            (x + ts, y + ts),
                            tsi - 1,
                            r_draw_interps,
                            r_fill_interps,
                            step_info,
                            texture,
                        );
                    }
                }
                r_draw_interps += step.dx_draw;
                r_fill_interps += step.dx_fill;
            }
            draw_interps += step.dy_draw;
            fill_interps += step.dy_fill;
        }
    }
    // Vector version
    #[inline(never)]
    fn fill_inner_tile<const ACCEPTED: bool>(
        &mut self,
        tl: (usize, usize),
        mut draw_interps: f32x4,
        fill_interps: f32x4,
        step_info: &[StepInfo; N_TILE_SIZES],
        texture: &assets::Texture,
    ) {
        let step = &step_info[step_info.len() - 1];
        let mut tx = f32x4::from_array([0.0, 1.0, 2.0, 3.0])
            * f32x4::splat(step.dx_fill[0])
            + f32x4::splat(fill_interps[0]);
        let mut ty = f32x4::from_array([0.0, 1.0, 2.0, 3.0])
            * f32x4::splat(step.dx_fill[1])
            + f32x4::splat(fill_interps[1]);

        let dtx_dx = step.dx_fill[0] * 4.0;
        let dty_dx = step.dx_fill[1] * 4.0;
        let dtx_dy = step.dy_fill[0] - step.dx_fill[0] * TILE_SIZES[0] as f32;
        let dty_dy = step.dy_fill[1] - step.dx_fill[1] * TILE_SIZES[0] as f32;

        let mut indx = 4 * (self.width * tl.1 + tl.0);
        for _ in 0..TILE_SIZES[0] {
            let mut row_indx = indx;
            let mut row_draw_interps =
                if ACCEPTED { f32x4::splat(0.0) } else { draw_interps.clone() };
            for _ in 0..(TILE_SIZES[0] / 4) {
                let line_framebuffer = unsafe {
                    self.pixels.get_unchecked_mut(row_indx..row_indx + 16)
                };

                let int_tx = unsafe {
                    tx.simd_clamp(
                        f32x4::splat(0.0),
                        f32x4::splat((assets::TEX_SIZE - 1) as f32),
                    )
                    .to_int_unchecked::<i32>()
                };
                let int_ty = unsafe {
                    ty.simd_clamp(
                        f32x4::splat(0.0),
                        f32x4::splat((assets::TEX_SIZE - 1) as f32),
                    )
                    .to_int_unchecked::<i32>()
                };
                let texcoords = i32x4::splat(4) * int_tx
                    + i32x4::splat(assets::TEX_SIZE as i32 * 4) * int_ty;

                for (tex_indx, pix) in texcoords
                    .as_array()
                    .iter()
                    .zip(line_framebuffer.chunks_exact_mut(4))
                {
                    if ACCEPTED || row_draw_interps.simd_ge(f32x4::splat(0.0)).all() {
                        let ti = *tex_indx as usize;
                        let tex: &[u8] = unsafe { texture.0.get_unchecked(ti..ti + 4) };
                        pix.copy_from_slice(tex);
                    }
                    if !ACCEPTED {
                        row_draw_interps += step.dx_draw;
                    }
                }
                tx += f32x4::splat(dtx_dx);
                ty += f32x4::splat(dty_dx);
                row_indx += 16;
            }
            tx += f32x4::splat(dtx_dy);
            ty += f32x4::splat(dty_dy);
            if !ACCEPTED {
                draw_interps += step.dy_draw;
            }
            indx += 4 * self.width;
        }
    }
    /// Takes the x and y coordinates for the 3 points of a triangle, and outputs the coordinates
    /// for the top-left and bottom-right points of the axis-aligned bounding box containing the
    /// triangle. The coordinates of this bounding box is guaranteed to be aligned to be a multiple
    /// of `align_to`
    fn compute_aligned_aabb(
        x: [f32; 3],
        y: [f32; 3],
        width: usize,
        height: usize,
        align_to: usize,
    ) -> [usize; 4] {
        let min_x = (min_f32(&x) as usize).clamp(0, width - 1);
        let min_y = (min_f32(&y) as usize).clamp(0, height - 1);
        let max_x = (max_f32(&x).ceil() as usize).clamp(0, width - 1);
        let max_y = (max_f32(&y).ceil() as usize).clamp(0, height - 1);
        [
            (min_x / align_to) * align_to,
            (min_y / align_to) * align_to,
            max_x.next_multiple_of(align_to),
            max_y.next_multiple_of(align_to),
        ]
    }
}

impl StepInfo {
    pub fn new() -> Self {
        Self {
            dx_draw: f32x4::splat(0.0),
            dy_draw: f32x4::splat(0.0),
            offset_accept: f32x4::splat(0.0),
            offset_reject: f32x4::splat(0.0),

            dx_fill: f32x4::splat(0.0),
            dy_fill: f32x4::splat(0.0),
        }
    }
    /// Computes the initial interpolants for the top-left corner (given by tl) of the AABB, as well
    /// as the step deltas. Step deltas are returned as a vec containing the deltas for
    /// varying tile sizes: first smallest to largest, with 1x1 at the end
    fn setup_tri(
        p_a: &[f32; 5],
        p_b: &[f32; 5],
        p_c: &[f32; 5],
        tl: (f32, f32),
    ) -> (f32x4, f32x4, [StepInfo; N_TILE_SIZES]) {
        let recip_tri_area = 1.0
            / Self::area_neg_y(
                (p_a[0] - p_b[0], p_a[1] - p_b[1]),
                (p_c[0] - p_b[0], p_c[1] - p_b[1]),
            );

        let mut step_info = Self::new();
        let mut draw_interps = f32x4::splat(0.0);
        let mut fill_interps = f32x4::splat(0.0);

        let edges: [(&[f32; 5], &[f32; 5], &[f32; 5]); 3] =
            [(p_a, p_b, p_c), (p_b, p_c, p_a), (p_c, p_a, p_b)];
        // Loop over all edges of the triangle in CW order
        for (indx, edge_points) in edges.iter().enumerate() {
            // An edge from a to b, with p_excluded being the 3rd point not involved in the edge
            let (a, b, p_3) = *edge_points;

            // Edge function at the top-left point, and derivatives for this edge function
            let e_tl = Self::area_neg_y(
                (tl.0 - a[0], tl.1 - a[1]),
                (b[0] - a[0], b[1] - a[1]),
            );
            let e_dx = Self::area_neg_y(
                (tl.0 + 1.0 - a[0], tl.1 - a[1]),
                (b[0] - a[0], b[1] - a[1]),
            ) - e_tl;
            let e_dy = Self::area_neg_y(
                (tl.0 - a[0], tl.1 + 1.0 - a[1]),
                (b[0] - a[0], b[1] - a[1]),
            ) - e_tl;

            let e_offsets = [0.0, e_dx, e_dy, e_dx + e_dy];
            draw_interps[indx] = e_tl;
            step_info.dx_draw[indx] = e_dx;
            step_info.dy_draw[indx] = e_dy;
            step_info.offset_accept[indx] = -min_f32(&e_offsets);
            step_info.offset_reject[indx] = -max_f32(&e_offsets);

            for (interp_indx, vertex_interp) in p_3.iter().skip(2).enumerate() {
                let prod: f32 = recip_tri_area * vertex_interp;
                fill_interps[interp_indx] += e_tl * prod;
                step_info.dx_fill[interp_indx] += e_dx * prod;
                step_info.dy_fill[interp_indx] += e_dy * prod;
            }
        }

        // Don't let free 4th draw interp affect trivial reject
        step_info.offset_reject[3] = -1.0;

        let mut step_infos: Vec<_> =
            TILE_SIZES.iter().map(|&x| step_info.resize_tile(x as f32)).collect();
        step_infos.push(step_info);

        let step_infos: [StepInfo; 4] = step_infos.try_into().expect("Error");
        (draw_interps, fill_interps, step_infos)
    }
    pub fn resize_tile(&self, tile_size: f32) -> Self {
        let mut acc = self.clone();
        let tile_size = f32x4::splat(tile_size);
        acc.dx_draw *= tile_size;
        acc.dy_draw *= tile_size;
        acc.offset_accept *= tile_size;
        acc.offset_reject *= tile_size;
        acc.dx_fill *= tile_size;
        acc.dy_fill *= tile_size;
        acc
    }
    pub fn rejects(&self, draw_interps: f32x4) -> bool {
        draw_interps.simd_le(self.offset_reject).any()
    }
    pub fn accepts(&self, draw_interps: f32x4) -> bool {
        draw_interps.simd_ge(self.offset_accept).all()
    }
    /// Computes the area of the triangle defined by the vectors `vec_a`, `vec_b` in 2D
    /// Negates the `y` coordinate because gfx coords
    fn area_neg_y(vec_a: (f32, f32), vec_b: (f32, f32)) -> f32 {
        0.5 * (vec_a.1 * vec_b.0 - vec_a.0 * vec_b.1)
    }
}
fn min_f32<const N: usize>(input: &[f32; N]) -> f32 {
    input.iter().fold(f32::MAX, |a, b| a.min(*b))
}
fn max_f32<const N: usize>(input: &[f32; N]) -> f32 {
    input.iter().fold(f32::MIN, |a: f32, b| a.max(*b))
}

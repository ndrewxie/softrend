use std::cell::{RefCell, RefMut};
use std::simd::prelude::*;

use super::Shader;
use super::Vec4;

const TILE_SIZES: [usize; 2] = [4, 32];
const N_TILE_SIZES: usize = TILE_SIZES.len() + 1; // 1x1 tiles

pub struct Raster {
    pixels: RefCell<&'static mut [u32]>,
    z_buf: RefCell<&'static mut [f32]>,
    fb_dims: (usize, usize),
    screen_dims: (usize, usize),
    raster_tl: (usize, usize),
    near: f32,
}

#[derive(Clone, Debug)]
pub struct StepDelta<const FIP: usize, const FIA: usize> {
    draw: f32x4,
    z: f32,
    persp: [f32; FIP],
    affine: [f32; FIA],
}
#[derive(Clone, Debug)]
struct DrawCutoffs {
    accept: f32x4,
    reject: f32x4,
}
#[derive(Debug)]
struct SetupData<const FIP: usize, const FIA: usize> {
    draw_tl: f32x4,
    interps: Interps<FIP, FIA>,
    dx: StepDelta<FIP, FIA>,
    dy: StepDelta<FIP, FIA>,
    cutoffs: DrawCutoffs,
}
#[derive(Clone, Debug)]
struct Interps<const FIP: usize, const FIA: usize> {
    persp: [f32x4; FIP],
    affine: [f32x4; FIA],
    z: f32x4,
}

#[derive(Debug)]
pub struct RasterAux<'a, S: Shader<FIP, FIA>, const FIP: usize, const FIA: usize> {
    raster: &'a Raster,
    shader: &'a S,
    dx: [StepDelta<FIP, FIA>; N_TILE_SIZES],
    dy: [StepDelta<FIP, FIA>; N_TILE_SIZES],
    cutoffs: [DrawCutoffs; TILE_SIZES.len()],
}

impl<'a, S: Shader<FIP, FIA> + 'a, const FIP: usize, const FIA: usize>
    RasterAux<'a, S, FIP, FIA>
{
    pub fn draw_tri(
        raster: &'a Raster,
        shader: &'a S,
        points: [Vec4; 3],
        persp_interps: [[f32; FIP]; 3],
        affine_interps: [[f32; FIA]; 3],
    ) {
        // Winding order check
        let (ba, bc) = (points[0] - points[1], points[2] - points[1]);
        let area = 0.5 * Self::cross_2d([ba.x(), ba.y()], [bc.x(), bc.y()]);
        if area <= 0.0 {
            return;
        }

        // Compute bounding box
        let ((mut min_x, mut min_y), (max_x, max_y)) =
            Self::tri_aabb(&points, raster.win_dims());
        let (bb_width, bb_height) = (max_x - min_x, max_y - min_y);
        if bb_width == 0 && bb_height == 0 {
            return;
        }

        // Compute starting tile index
        let start_tile = TILE_SIZES
            .iter()
            .position(|&x| 2 * x >= bb_width && 2 * x >= bb_height)
            .unwrap_or(TILE_SIZES.len() - 1);
        let start_tile_size = TILE_SIZES[start_tile];
        // Align start coordinates to tile size
        min_x = (min_x / start_tile_size) * start_tile_size;
        min_y = (min_y / start_tile_size) * start_tile_size;

        let tri_data = Self::frag_deltas(
            (min_x, min_y),
            points,
            persp_interps,
            affine_interps,
            area,
            shader.z_offset(),
        );

        let aux = Self {
            raster: raster,
            shader,
            dx: std::array::from_fn(|i| {
                let mut dx = tri_data.dx.clone();
                (i != 0).then(|| dx.scale(TILE_SIZES[i - 1] as f32));
                dx
            }),
            dy: std::array::from_fn(|i| {
                let mut dy = tri_data.dy.clone();
                (i != 0).then(|| dy.scale(TILE_SIZES[i - 1] as f32));
                dy
            }),
            cutoffs: std::array::from_fn(|i| {
                let mut cutoffs = tri_data.cutoffs.clone();
                cutoffs.scale(TILE_SIZES[i] as f32);
                cutoffs
            }),
        };

        aux.step_tile::<false>(
            (min_x, min_y),
            (max_x, max_y),
            start_tile,
            tri_data.draw_tl,
            tri_data.interps,
        );
    }
    fn step_tile<const ACCEPTED: bool>(
        &self,
        tl: (usize, usize),
        br: (usize, usize),
        tsi: usize,
        mut draw: f32x4,
        mut interps: Interps<FIP, FIA>,
    ) {
        let ts = TILE_SIZES[tsi];
        let cutoffs = &self.cutoffs[tsi];
        let (dx, dy) = (&self.dx[tsi + 1], &self.dy[tsi + 1]);

        let mut y = tl.1;
        while y < br.1 {
            let mut x = tl.0;
            let mut row_interps = interps.clone();
            let mut row_draw = draw;
            while x < br.0 {
                let accepts = cutoffs.accepts(row_draw);
                let rejects = cutoffs.rejects(row_draw);
                macro_rules! tile {
                    (INNER, $x:expr) => {
                        self.fill_inner_tile::<$x>((x, y), row_draw, row_interps.clone())
                    };
                    (NEXT, $level:expr, $x:expr) => {
                        self.step_tile::<$x>(
                            (x, y),
                            (x + ts, y + ts),
                            $level,
                            row_draw,
                            row_interps.clone(),
                        )
                    };
                }

                match (tsi, accepts, rejects) {
                    (_, _, true) => (),
                    (0, false, _) => tile!(INNER, false),
                    (0, true, _) => tile!(INNER, true),
                    (_, true, _) => tile!(NEXT, 0, true),
                    (_, false, _) => tile!(NEXT, tsi - 1, false),
                }
                row_interps.step_by(dx);
                row_draw += dx.draw;
                x += ts;
            }
            interps.step_by(dy);
            draw += dy.draw;
            y += ts;
        }
    }
    fn fill_inner_tile<const ACCEPTED: bool>(
        &self,
        tl: (usize, usize),
        mut draw_interps: f32x4,
        mut fill_interps: Interps<FIP, FIA>,
    ) {
        let raster = self.raster;
        let width = raster.fb_dims().0;
        let (dx, dy) = (&self.dx[0], &self.dy[0]);
        let z_offset = self.shader.z_offset();

        let mut index = tl.1 * width + tl.0;
        let z_indx = tl.1 * width + 4 * tl.0;
        let mut pixels = raster.pixels();
        let mut z_buf = raster.z_buf();
        let z_block = unsafe { z_buf.get_unchecked_mut(z_indx..z_indx + 16) };
        for z_buf_row in z_block.chunks_exact_mut(4) {
            let pix_row = unsafe { pixels.get_unchecked_mut(index..index + 4) };
            let pix_vec = u32x4::from_slice(pix_row);
            let z_buf_vec = f32x4::from_slice(z_buf_row);

            let mut row_draw = draw_interps;

            let persp = fill_interps.persp_div(z_offset);
            let shaded = self.shader.frag(&persp, &fill_interps.affine);
            let blended = self.shader.blend(shaded, pix_vec);

            let mut write_mask = fill_interps.z.simd_gt(z_buf_vec);
            (!ACCEPTED).then(|| unsafe {
                let mut draw_mask = mask32x4::splat(false);
                let zero = f32x4::splat(0.0);
                for i in 0..4 {
                    draw_mask.set_unchecked(i, row_draw.simd_ge(zero).all());
                    row_draw += dx.draw;
                }
                write_mask &= draw_mask;
            });

            write_mask.select(fill_interps.z, z_buf_vec).copy_to_slice(z_buf_row);
            write_mask.select(blended, pix_vec).copy_to_slice(pix_row);

            (!ACCEPTED).then(|| draw_interps += dy.draw);
            fill_interps.step_by(dy);
            index += width;
        }
    }
    fn tri_aabb(
        points: &[Vec4; 3],
        win_dims: (usize, usize),
    ) -> ((usize, usize), (usize, usize)) {
        fn min_clamp(p: &[f32; 3], low: isize, high: isize) -> isize {
            let min = p.iter().fold(f32::MAX, |a, b| a.min(*b));
            (min.floor() as isize).clamp(low, high)
        }
        fn max_clamp(p: &[f32; 3], low: isize, high: isize) -> isize {
            let max = p.iter().fold(f32::MIN, |a: f32, b| a.max(*b));
            (max.ceil() as isize).clamp(low, high)
        }
        let x_coords: [f32; 3] = std::array::from_fn(|i| points[i].x());
        let y_coords: [f32; 3] = std::array::from_fn(|i| points[i].y());
        let win_width = win_dims.0 as isize - 1;
        let win_height = win_dims.1 as isize - 1;
        let min_x = min_clamp(&x_coords, 0, win_width) as usize;
        let min_y = min_clamp(&y_coords, 0, win_height) as usize;
        let max_x = max_clamp(&x_coords, 0, win_width) as usize;
        let max_y = max_clamp(&y_coords, 0, win_height) as usize;

        ((min_x, min_y), (max_x, max_y))
    }
    fn frag_deltas(
        tl: (usize, usize),
        points: [Vec4; 3],
        mut persp_interps: [[f32; FIP]; 3],
        affine_interps: [[f32; FIA]; 3],
        tri_area: f32,
        z_offset: f32,
    ) -> SetupData<FIP, FIA> {
        // Calculates (val_top_left, dx, dy) of an interpolant
        // Expects the interpolant values at each vertex to be packed into
        // the first 3 lanes of a f32x4 (in the correct order), with a 0.0 in
        // the 4th lane
        fn interp_step<const N: usize>(
            draw_tl: f32x4,
            draw_dx: f32x4,
            draw_dy: f32x4,
            point_vals: &[[f32; N]; 3],
            recip_tri_area: f32,
        ) -> ([f32; N], [f32; N], [f32; N]) {
            let mut tl = [0.0; N];
            let mut dx = [0.0; N];
            let mut dy = [0.0; N];
            for i in 0..N {
                for j in 0..3 {
                    tl[i] += point_vals[j][i] * draw_tl[j] * recip_tri_area;
                    dx[i] += point_vals[j][i] * draw_dx[j] * recip_tri_area;
                    dy[i] += point_vals[j][i] * draw_dy[j] * recip_tri_area;
                }
            }
            (tl, dx, dy)
        }

        let recip_tri_area = tri_area.recip();

        let mut draw_tl = f32x4::splat(0.0);
        let mut draw_dx = f32x4::splat(0.0);
        let mut draw_dy = f32x4::splat(0.0);
        let mut accept = f32x4::splat(0.0);
        let mut reject = f32x4::splat(0.0);

        let tl = (tl.0 as f32, tl.1 as f32);
        for i in 0_isize..3 {
            let indx = i as usize;
            let p_from = &points[(i - 2).rem_euclid(3) as usize];
            let p_to = &points[(i - 1).rem_euclid(3) as usize];

            let e_tl = 0.5
                * Self::cross_2d(
                    [tl.0 - p_from.x(), tl.1 - p_from.y()],
                    [p_to.x() - p_from.x(), p_to.y() - p_from.y()],
                );

            let e_dx = 0.5 * (p_to.y() - p_from.y());
            let e_dy = 0.5 * (p_from.x() - p_to.x());

            let e_deltas = [0.0, e_dx, e_dy, e_dx + e_dy];
            accept[indx] = -e_deltas.iter().fold(f32::MAX, |a, b| a.min(*b));
            reject[indx] = -e_deltas.iter().fold(f32::MIN, |a, b| a.max(*b));

            draw_tl[indx] = e_tl;
            draw_dx[indx] = e_dx;
            draw_dy[indx] = e_dy;
        }

        let recip_z: [[f32; 1]; 3] = std::array::from_fn(|i| [points[i].z()]);
        for (rcp_z, pis) in recip_z.iter().zip(persp_interps.iter_mut()) {
            pis.iter_mut().for_each(|x| *x *= rcp_z[0] - z_offset);
        }

        let (z_tl, z_dx, z_dy) =
            interp_step(draw_tl, draw_dx, draw_dy, &recip_z, recip_tri_area);
        let (persp_tl, persp_dx, persp_dy) =
            interp_step(draw_tl, draw_dx, draw_dy, &persp_interps, recip_tri_area);
        let (affine_tl, affine_dx, affine_dy) =
            interp_step(draw_tl, draw_dx, draw_dy, &affine_interps, recip_tri_area);

        let dx = StepDelta { draw: draw_dx, z: z_dx[0], persp: persp_dx, affine: affine_dx };
        let dy = StepDelta { draw: draw_dy, z: z_dy[0], persp: persp_dy, affine: affine_dy };

        let interps = Interps::from(z_tl[0], persp_tl, affine_tl, &dx);

        SetupData { draw_tl, interps, dx, dy, cutoffs: DrawCutoffs { accept, reject } }
    }
    fn cross_2d(a: [f32; 2], b: [f32; 2]) -> f32 {
        a[0] * b[1] - b[0] * a[1]
    }
}

impl Raster {
    pub fn new(raster_tl: (usize, usize), screen_dims: (usize, usize), near: f32) -> Self {
        #[repr(C, align(128))]
        struct FbAlign([u32; 32]);
        #[repr(C, align(128))]
        struct ZAlign([f32; 32]);

        let width_aligned = screen_dims.0.next_multiple_of(*TILE_SIZES.last().unwrap());
        let height_aligned = screen_dims.1.next_multiple_of(*TILE_SIZES.last().unwrap());
        let pixels =
            Self::alloc_framebuffer::<FbAlign, u32>(width_aligned * height_aligned, 0);
        let z_buf =
            Self::alloc_framebuffer::<ZAlign, f32>(width_aligned * height_aligned, 0.0);
        Self {
            pixels: RefCell::from(pixels),
            z_buf: RefCell::from(z_buf),
            screen_dims,
            fb_dims: (width_aligned, height_aligned),
            raster_tl,
            near,
        }
    }
    /// Allocates a slice, aligned to the alignment of `Align` and filled with `fill`.
    /// Leaks memory, but as long as it isn't called repeatedly that's OK.
    fn alloc_framebuffer<Align, T: Copy>(size: usize, fill: T) -> &'static mut [T] {
        let fb = vec![fill; size + 4 * std::mem::size_of::<Align>()];
        let fb_slice: &'static mut [T] = Vec::leak(fb);
        unsafe {
            let (_, aligned_fb, _) = fb_slice.align_to_mut::<Align>();
            std::slice::from_raw_parts_mut(aligned_fb.as_mut_ptr() as *mut T, size)
        }
    }
    pub fn win_dims(&self) -> (usize, usize) {
        self.screen_dims
    }
    pub fn fb_dims(&self) -> (usize, usize) {
        self.fb_dims
    }
    pub fn tl(&self) -> (usize, usize) {
        self.raster_tl
    }
    pub fn pixels(&self) -> RefMut<&'static mut [u32]> {
        self.pixels.borrow_mut()
    }
    pub fn z_buf(&self) -> RefMut<&'static mut [f32]> {
        self.z_buf.borrow_mut()
    }
    pub fn near(&self) -> f32 {
        self.near
    }
    pub fn clear(&self) {
        self.pixels().fill(0);
        self.z_buf().fill(0.0);
    }
    pub fn copy_to_brga_u32<B>(&self, buffer: &mut B)
    where
        B: std::ops::DerefMut<Target = [u32]>,
    {
        let pixels = self.pixels();
        let pixels = unsafe {
            let pix_ptr = pixels.as_ptr() as *mut u8;
            let pix_len = pixels.len() * 4;
            std::slice::from_raw_parts(pix_ptr, pix_len)
        };

        for (in_row, out_row) in pixels
            .chunks_exact(4 * self.fb_dims().0)
            .zip(buffer.chunks_exact_mut(self.win_dims().0))
        {
            for (in_pix, out_pix) in in_row.chunks_exact(4).zip(out_row.iter_mut()) {
                let red = in_pix[0] as u32;
                let green = in_pix[1] as u32;
                let blue = in_pix[2] as u32;
                *out_pix = blue | (green << 8) | (red << 16);
            }
        }
    }
}

impl DrawCutoffs {
    pub fn rejects(&self, draw_interps: f32x4) -> bool {
        draw_interps.simd_lt(self.reject).any()
    }
    pub fn accepts(&self, draw_interps: f32x4) -> bool {
        draw_interps.simd_ge(self.accept).all()
    }
    pub fn scale(&mut self, fac: f32) {
        self.accept *= f32x4::splat(fac);
        self.reject *= f32x4::splat(fac);
    }
}

impl<const FIP: usize, const FIA: usize> StepDelta<FIP, FIA> {
    pub fn scale(&mut self, fac: f32) {
        self.draw *= f32x4::splat(fac);
        self.z *= fac;
        self.persp.iter_mut().for_each(|x| *x *= fac);
        self.affine.iter_mut().for_each(|x| *x *= fac);
    }
}

impl<const FIP: usize, const FIA: usize> Interps<FIP, FIA> {
    pub fn from(
        z: f32,
        persp: [f32; FIP],
        affine: [f32; FIA],
        dx: &StepDelta<FIP, FIA>,
    ) -> Self {
        let step = f32x4::from_array([0.0, 1.0, 2.0, 3.0]);
        Self {
            z: f32x4::splat(z) + step * f32x4::splat(dx.z),
            persp: std::array::from_fn(|i| {
                f32x4::splat(persp[i]) + step * f32x4::splat(dx.persp[i])
            }),
            affine: std::array::from_fn(|i| {
                f32x4::splat(affine[i]) + step * f32x4::splat(dx.affine[i])
            }),
        }
    }
    pub fn step_by(&mut self, delta: &StepDelta<FIP, FIA>) {
        self.z += f32x4::splat(delta.z);
        for (persp, dpersp) in self.persp.iter_mut().zip(delta.persp) {
            *persp += f32x4::splat(dpersp);
        }
        for (affine, daffine) in self.affine.iter_mut().zip(delta.affine) {
            *affine += f32x4::splat(daffine);
        }
    }
    pub fn persp_div(&self, z_offset: f32) -> [f32x4; FIP] {
        let z = (self.z - f32x4::splat(z_offset)).recip();
        let mut persp = self.persp;
        persp.iter_mut().for_each(|x| *x *= z);
        persp
    }
}

impl std::fmt::Debug for Raster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Raster")
    }
}

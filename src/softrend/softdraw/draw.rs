mod config;
mod fixedpoint;
mod raster;
mod steps;

use super::*;
use config::*;
use fixedpoint::*;
pub use raster::*;
use steps::*;

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
        let area_usize = area.round() as usize;

        // Compute bounding box
        let ((mut min_x, mut min_y), (max_x, max_y)) =
            Self::tri_aabb(&points, raster.win_dims());
        let (bb_width, bb_height) = (max_x - min_x, max_y - min_y);
        if bb_width == 0 || bb_height == 0 {
            return;
        }

        // Compute starting tile index
        let start_tile = (0..TILE_SIZES.len())
            .rev()
            .find(|&ti| {
                let size = TILE_SIZES[ti];
                2 * size <= bb_width && 2 * size <= bb_height && 4 * area_usize <= size * size
            })
            .unwrap_or(0);
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
            raster,
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

        aux.step_tile::<false>((min_x, min_y), (max_x, max_y), start_tile, tri_data.interps);
    }
    fn step_tile<const ACCEPTED: bool>(
        &self,
        tl: (usize, usize),
        br: (usize, usize),
        tsi: usize,
        mut interps: Interps<FIP, FIA>,
    ) {
        let ts = TILE_SIZES[tsi];
        let cutoffs = &self.cutoffs[tsi];
        let (dx, dy) = (&self.dx[tsi + 1], &self.dy[tsi + 1]);

        let mut y = tl.1;
        while y < br.1 {
            let mut x = tl.0;
            let mut row_interps = interps.clone();
            while x < br.0 {
                let accepts = ACCEPTED || row_interps.accepts(&cutoffs);
                let rejects = (!ACCEPTED) && row_interps.rejects(&cutoffs);
                macro_rules! tile {
                    (INNER, $x:expr) => {
                        self.fill_inner_tile::<$x>((x, y), row_interps.clone())
                    };
                    (NEXT, $level:expr, $x:expr) => {
                        self.step_tile::<$x>(
                            (x, y),
                            (x + ts, y + ts),
                            $level,
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
                row_interps.step_by::<ACCEPTED>(dx);
                x += ts;
            }
            interps.step_by::<ACCEPTED>(dy);
            y += ts;
        }
    }
    fn fill_inner_tile<const ACCEPTED: bool>(
        &self,
        tl: (usize, usize),
        mut interps: Interps<FIP, FIA>,
    ) {
        let raster = self.raster;
        let width = raster.fb_dims().0;
        let dy = &self.dy[0];
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

            let persp = interps.persp_correct(z_offset);
            let shaded = self.shader.frag(&persp, &interps.affine);
            let blended = self.shader.blend(shaded, pix_vec);

            let mut write_mask = interps.rcp_z.simd_gt(z_buf_vec);
            if !ACCEPTED {
                write_mask &= interps.draw_mask();
            }

            write_mask.select(interps.rcp_z, z_buf_vec).copy_to_slice(z_buf_row);
            write_mask.select(blended, pix_vec).copy_to_slice(pix_row);

            interps.step_by::<ACCEPTED>(dy);
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
        let fxzero = Fx28_4::from_float(0.0);
        let fxszero = Fx28_4x4::splat_float(0.0);

        let (mut draw_tl, mut draw_dx, mut draw_dy) = (fxszero, fxszero, fxszero);
        let (mut accept, mut reject) = (fxszero, fxszero);

        let tl = (tl.0 as f32, tl.1 as f32);
        for i in 0_isize..3 {
            let indx = i as usize;
            let p_from = &points[(i - 2).rem_euclid(3) as usize];
            let p_to = &points[(i - 1).rem_euclid(3) as usize];

            let (x0, y0) = (Fx28_4::from_float(p_from.x()), Fx28_4::from_float(p_from.y()));
            let (x1, y1) = (Fx28_4::from_float(p_to.x()), Fx28_4::from_float(p_to.y()));

            let dx = Fx28_4::from_float(0.5) * (y1 - y0);
            let dy = Fx28_4::from_float(0.5) * (x0 - x1);
            let e_tl =
                (Fx28_4::from_float(tl.0) - x0) * dx + (Fx28_4::from_float(tl.1) - y0) * dy;

            accept[indx] -= if dx < fxzero { dx } else { fxzero };
            reject[indx] -= if dx < fxzero { fxzero } else { dx };
            accept[indx] -= if dy < fxzero { dy } else { fxzero };
            reject[indx] -= if dy < fxzero { fxzero } else { dy };

            draw_tl[indx] = e_tl;
            draw_dx[indx] = dx;
            draw_dy[indx] = dy;
        }

        let recip_z: [[f32; 1]; 3] = std::array::from_fn(|i| [points[i].z()]);
        for (rcp_z, pis) in recip_z.iter().zip(persp_interps.iter_mut()) {
            pis.iter_mut().for_each(|x| *x *= rcp_z[0] - z_offset);
        }

        let (f_draw_tl, f_draw_dx, f_draw_dy) =
            (draw_tl.to_float(), draw_dx.to_float(), draw_dy.to_float());
        let (z_tl, z_dx, z_dy) =
            interp_step(f_draw_tl, f_draw_dx, f_draw_dy, &recip_z, recip_tri_area);
        let (persp_tl, persp_dx, persp_dy) =
            interp_step(f_draw_tl, f_draw_dx, f_draw_dy, &persp_interps, recip_tri_area);
        let (affine_tl, affine_dx, affine_dy) =
            interp_step(f_draw_tl, f_draw_dx, f_draw_dy, &affine_interps, recip_tri_area);

        let dx =
            StepDelta { edges: draw_dx, rcp_z: z_dx[0], persp: persp_dx, affine: affine_dx };
        let dy =
            StepDelta { edges: draw_dy, rcp_z: z_dy[0], persp: persp_dy, affine: affine_dy };

        let interps = Interps::from(z_tl[0], persp_tl, affine_tl, draw_tl, &dx);

        SetupData { interps, dx, dy, cutoffs: DrawCutoffs { accept, reject } }
    }
    fn cross_2d(a: [f32; 2], b: [f32; 2]) -> f32 {
        a[0] * b[1] - b[0] * a[1]
    }
}

#![allow(dead_code)]

const SMALL_MODEL_SIZE: usize = 32;
// In NDC coords, clip when |{x, y}| >= GUARD_BAND_TOLERANCE
const GUARD_BAND_TOLERANCE: f32 = 1.5;

use super::matrices::*;
use smallvec::SmallVec;
use std::simd::prelude::*;

mod draw;
pub use draw::Raster;
use draw::*;

pub type VertexData<const FIP: usize, const FIA: usize> = ([f32; FIP], [f32; FIA]);
pub trait Shader<const FIP: usize, const FIA: usize>: std::fmt::Debug {
    type VertexIn: Clone + Send;
    /// The vertex shader.
    /// Input: Vertex data
    /// Output: Vertex location in clip space coordinates, and fragment
    ///         interpolant values (perspective and affine) at the vertex
    /// Clip space coordinates are the coordinates right before the perspective
    /// divide, which will then convert clip space coordinates into NDC coordinates.
    /// Do not perform the perspective divide, the renderer will handle it.
    /// NDC coordinate format:
    ///     * -1 <= x <= 1
    ///     * -1 <= y <= 1
    ///     * 0 (far) <= z <= 1 (near)
    ///     * w: Original z coordinate, in world coordinates
    ///     NOTE: If perspective correct interpolants are used, the z'(z) function
    ///     is assumed to be affine. See z_offset.
    fn vert(&self, input: &Self::VertexIn) -> (Vec4, VertexData<FIP, FIA>);
    /// Linearly interpolate the vertex data between `start` and `end`
    /// 0 <= t <= 1
    fn vd_lerp(
        &self,
        start: &VertexData<FIP, FIA>,
        end: &VertexData<FIP, FIA>,
        t: f32,
    ) -> VertexData<FIP, FIA>;
    /// Fragment shader. Inputs are perspective corrected interpolants (p_i)
    /// and affinely interpolated interpolants (a_i)
    /// Output should be a pixel color
    fn frag(&self, p_i: &[f32x4; FIP], a_i: &[f32x4; FIA]) -> u32x4;
    /// Blend shader. Inputs are the fragment color (coming from fragment shader)
    /// and the current framebuffer color. Output should be blended pixel color
    fn blend(&self, frag_color: u32x4, fb_color: u32x4) -> u32x4;
    /// If perspective correct interpolants are used, it is assumed that
    /// the z coord is transformed as z' = (a/z) + b (with the perspective
    /// divide being handled automatically). z_offset is the constant `b`
    /// value.
    /// If perspective correct interpolants aren't used, set to 0.0
    fn z_offset(&self) -> f32;
}

pub struct Mesh<S: Shader<FIP, FIA>, const FIP: usize, const FIA: usize> {
    vertices: SmallVec<[S::VertexIn; SMALL_MODEL_SIZE]>,
    polys: SmallVec<[SmallVec<[usize; 4]>; SMALL_MODEL_SIZE]>,
    shader: S,
}

impl<S: Shader<FIP, FIA>, const FIP: usize, const FIA: usize> Mesh<S, FIP, FIA> {
    pub fn new(shader: S) -> Self {
        Self { vertices: SmallVec::new(), polys: SmallVec::new(), shader }
    }
    pub fn add_vertices(&mut self, vertices: impl IntoIterator<Item = S::VertexIn>) {
        self.vertices.extend(vertices);
    }
    pub fn add_tris(&mut self, tris: impl IntoIterator<Item = [usize; 3]>) {
        self.polys.extend(tris.into_iter().map(|x| SmallVec::from_slice(&x)));
    }
    pub fn add_quads(&mut self, quads: impl IntoIterator<Item = [usize; 4]>) {
        self.polys.extend(quads.into_iter().map(|x| SmallVec::from_slice(&x)));
    }
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.polys.clear();
    }
    pub fn draw(&self, raster: &Raster) {
        let mut vertex_coords: SmallVec<[_; SMALL_MODEL_SIZE]> = SmallVec::new();
        let mut vertex_attrs: SmallVec<[_; SMALL_MODEL_SIZE]> = SmallVec::new();
        if self.vertices.len() > SMALL_MODEL_SIZE {
            vertex_coords.reserve_exact(SMALL_MODEL_SIZE - self.vertices.len());
            vertex_attrs.reserve_exact(SMALL_MODEL_SIZE - self.vertices.len());
        }
        for vert in self.vertices.iter() {
            let (loc, attr) = self.shader.vert(vert);
            vertex_coords.push(loc);
            vertex_attrs.push(attr);
        }

        let mut clip_points = [Vec4::from_array([0.0; 4]); 32];
        let mut clip_attrs = [([0.0; FIP], [0.0; FIA]); 32];
        let mut clip_accepts = [false; 32];
        let mut num_points;
        for poly in self.polys.iter() {
            if poly.len() < 3 {
                continue;
            }
            let ab = vertex_coords[poly[1]] - vertex_coords[poly[0]];
            let ac = vertex_coords[poly[poly.len() - 1]] - vertex_coords[poly[0]];
            if ac.cross_3d(&ab).z() <= 0.0 {
                continue;
            }

            num_points = poly.len();
            for indx in 0..num_points {
                clip_points[indx] = vertex_coords[poly[indx]];
                clip_attrs[indx] = vertex_attrs[poly[indx]];
            }
            #[rustfmt::skip]
            let normals = [
                Vec4::from_array([0.0,  0.0,  -1.0, 1.0]), // Near plane
                Vec4::from_array([0.0,  0.0,  1.0,  0.0]), // Far plane
                Vec4::from_array([1.0,  0.0,  0.0,  GUARD_BAND_TOLERANCE]), // Min-x plane
                Vec4::from_array([-1.0, 0.0,  0.0,  GUARD_BAND_TOLERANCE]), // Max-x plane
                Vec4::from_array([0.0,  1.0,  0.0,  GUARD_BAND_TOLERANCE]), // Min-y plane
                Vec4::from_array([0.0,  -1.0, 0.0,  GUARD_BAND_TOLERANCE]), // Max-y plane
            ];
            for normal in normals {
                self.clip_poly(
                    &mut clip_points,
                    &mut clip_attrs,
                    &mut clip_accepts,
                    &mut num_points,
                    normal,
                );
            }

            let (raster_x, raster_y) = raster.tl();
            let (width, height) = raster.win_dims();
            let (halfwidth, halfheight) = (0.5 * width as f32, 0.5 * height as f32);
            let max_dim = halfwidth.max(halfheight);
            for p in clip_points.iter_mut().take(num_points) {
                p.w_div();
                p[0] = p[0] * max_dim + halfwidth - raster_x as f32;
                p[1] = -p[1] * max_dim + halfheight - raster_y as f32;
            }

            let mut win = 1;
            // Clipping a convex poly against near plane preserves convexity
            // so a simple fan triangulation works
            // May produce slivers, however
            while win + 1 < num_points {
                let (pa, ta) = (clip_points[0], &clip_attrs[0]);
                let (pb, tb) = (clip_points[win], &clip_attrs[win]);
                let (pc, tc) = (clip_points[win + 1], &clip_attrs[win + 1]);
                RasterAux::draw_tri(
                    raster,
                    &self.shader,
                    [pa, pc, pb],
                    [ta.0, tc.0, tb.0],
                    [ta.1, tc.1, tb.1],
                );
                win += 1;
            }
        }
    }
    /// Clips a convex polygon in homogenous coordinates, against a plane with
    /// a homogenous normal specified by `normal`
    fn clip_poly<const N: usize>(
        &self,
        points: &mut [Vec4; N],
        attrs: &mut [VertexData<FIP, FIA>; N],
        accepts: &mut [bool; N],
        num_points: &mut usize,
        normal: Vec4,
    ) {
        fn insert<T: Copy>(arr: &mut [T], len: usize, val: T, at: usize) {
            for indx in (at..len).rev() {
                arr[indx + 1] = arr[indx];
            }
            arr[at] = val;
        }
        fn remove<T: Copy>(arr: &mut [T], len: usize, at: usize) {
            for indx in at..(len - 1) {
                arr[indx] = arr[indx + 1];
            }
        }

        for i in 0..*num_points {
            accepts[i] = points[i].dot(&normal) >= 0.0;
        }
        if accepts.iter().take(*num_points).all(|&x| x) {
            // No clipping needed
            return;
        }
        if accepts.iter().take(*num_points).all(|&x| !x) {
            // Entirely culled
            *num_points = 0;
            return;
        }
        let first_valid = accepts.iter().position(|&x| x).unwrap();

        let mut index = first_valid;
        let mut counter = 0;
        let num_iters = *num_points;
        while counter < num_iters {
            counter += 1;
            let cei = index % *num_points;
            let nei = (index + 1) % *num_points;
            if accepts[cei] && accepts[nei] {
                index += 1;
                continue;
            }
            if !accepts[cei] && !accepts[nei] {
                remove(points, *num_points, cei);
                remove(attrs, *num_points, cei);
                remove(accepts, *num_points, cei);
                *num_points -= 1;
                continue;
            }

            let t = points[cei].dot(&normal) / (points[cei] - points[nei]).dot(&normal);
            let p_inter = points[cei] + (points[nei] - points[cei]) * t;
            let a_inter = self.shader.vd_lerp(&attrs[cei], &attrs[nei], t);

            if accepts[cei] && !accepts[nei] {
                insert(points, *num_points, p_inter, nei);
                insert(attrs, *num_points, a_inter, nei);
                insert(accepts, *num_points, true, nei);
                *num_points += 1;
                index = nei + 1;
                continue;
            }
            if !accepts[cei] && accepts[nei] {
                points[cei] = p_inter;
                attrs[cei] = a_inter;
                accepts[cei] = true;
                index += 1;
                continue;
            }
        }
    }
}

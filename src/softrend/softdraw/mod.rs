#![allow(dead_code)]

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
    vertices: Vec<S::VertexIn>,
    polys: Vec<SmallVec<[usize; 4]>>,
    shader: S,
}

impl<S: Shader<FIP, FIA>, const FIP: usize, const FIA: usize> Mesh<S, FIP, FIA> {
    pub fn new(shader: S) -> Self {
        Self { vertices: Vec::with_capacity(128), polys: Vec::with_capacity(32), shader }
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
        let (mut vertex_coords, mut vertex_attrs) = (Vec::new(), Vec::new());
        vertex_coords.reserve_exact(self.vertices.len());
        vertex_attrs.reserve_exact(self.vertices.len());

        for vert in self.vertices.iter() {
            let (loc, attr) = self.shader.vert(vert);
            vertex_coords.push(loc);
            vertex_attrs.push(attr);
        }

        // Draw triangles with near clipping
        let mut clip_points: SmallVec<[Vec4; 4]> = SmallVec::with_capacity(4);
        let mut clip_attrs: SmallVec<[VertexData<FIP, FIA>; 4]> = SmallVec::with_capacity(4);
        let mut clip_accepts: SmallVec<[bool; 4]> = SmallVec::with_capacity(4);
        for poly in self.polys.iter() {
            clip_points.clear();
            clip_attrs.clear();

            for &p_indx in poly {
                clip_points.push(vertex_coords[p_indx]);
                clip_attrs.push(vertex_attrs[p_indx]);
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
                self.clip_poly(&mut clip_points, &mut clip_attrs, &mut clip_accepts, normal);
            }

            let (raster_x, raster_y) = raster.tl();
            let (width, height) = raster.win_dims();
            let (halfwidth, halfheight) = (0.5 * width as f32, 0.5 * height as f32);
            let max_dim = halfwidth.max(halfheight);
            for p in clip_points.iter_mut() {
                p.w_div();
                p[0] = p[0] * max_dim + halfwidth - raster_x as f32;
                p[1] = -p[1] * max_dim + halfheight - raster_y as f32;
            }

            let mut win = 1;
            // Clipping a convex poly against near plane preserves convexity
            // so a simple fan triangulation works
            // May produce slivers, however
            while win + 1 < clip_points.len() {
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
    fn clip_poly(
        &self,
        points: &mut SmallVec<[Vec4; 4]>,
        attrs: &mut SmallVec<[VertexData<FIP, FIA>; 4]>,
        accepts: &mut SmallVec<[bool; 4]>,
        normal: Vec4,
    ) {
        accepts.clear();
        points.iter().for_each(|p| accepts.push(p.dot(&normal) >= 0.0));
        if accepts.iter().all(|&x| x) {
            // No clipping needed
            return;
        }
        if accepts.iter().all(|&x| !x) {
            // Entirely culled
            points.clear();
            attrs.clear();
            return;
        }
        let first_valid = accepts.iter().position(|&x| x).unwrap();

        let mut index = first_valid;
        let mut counter = 0;
        let num_iters = points.len();
        while counter < num_iters {
            counter += 1;
            let cei = index % points.len();
            let nei = (index + 1) % points.len();
            if accepts[cei] && accepts[nei] {
                index += 1;
                continue;
            }
            if !accepts[cei] && !accepts[nei] {
                points.remove(cei);
                attrs.remove(cei);
                accepts.remove(cei);
                continue;
            }

            let t = points[cei].dot(&normal) / (points[cei] - points[nei]).dot(&normal);
            let p_inter = points[cei] + (points[nei] - points[cei]) * t;
            let a_inter = self.shader.vd_lerp(&attrs[cei], &attrs[nei], t);

            if accepts[cei] && !accepts[nei] {
                points.insert(nei, p_inter);
                attrs.insert(nei, a_inter);
                accepts.insert(nei, true);
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

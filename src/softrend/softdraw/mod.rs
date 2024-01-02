#![allow(dead_code)]
use super::matrices::*;
use std::simd::prelude::*;

mod draw;
pub use draw::Raster;
use draw::*;

pub type VertexData<const FIP: usize, const FIA: usize> = ([f32; FIP], [f32; FIA]);
pub trait Shader<const FIP: usize, const FIA: usize>: std::fmt::Debug {
    type VertexIn: Clone + Send;
    /// Vertex shader. Takes in a `VertexIn` and outputs the transformed location
    /// and data of the vertex. Perspective divide should not be performed
    /// on the transformed location: it should be left in clip space coordinates,
    /// and will be transformed into NDC coordinates following a perspective divide.
    /// NDC coordinate format:
    /// * -1 <= x <= 1
    /// * -1 <= y <= 1
    /// * 0 (far) <= z <= 1 (near)
    /// * w: z coordinate in world coordinates
    /// NOTE: If perspective correct interpolants are used, see z_offset
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
    tris: Vec<[usize; 3]>,
    shader: S,
}

impl<S: Shader<FIP, FIA>, const FIP: usize, const FIA: usize> Mesh<S, FIP, FIA> {
    pub fn new(shader: S) -> Self {
        Self { vertices: Vec::with_capacity(128), tris: Vec::with_capacity(32), shader }
    }
    pub fn add_vertices(&mut self, vertices: impl IntoIterator<Item = S::VertexIn>) {
        self.vertices.extend(vertices);
    }
    pub fn add_tris(&mut self, tris: impl IntoIterator<Item = [usize; 3]>) {
        self.tris.extend(tris);
    }
    pub fn add_quads(&mut self, quads: impl IntoIterator<Item = [usize; 4]>) {
        for quad in quads {
            self.add_tris([[quad[0], quad[1], quad[2]], [quad[2], quad[3], quad[0]]])
        }
    }
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.tris.clear();
    }
    pub fn draw(&self, raster: &Raster) {
        let near = raster.near();
        let (mut vertex_coords, mut vertex_attrs) = (Vec::new(), Vec::new());
        vertex_coords.reserve_exact(self.vertices.len());
        vertex_attrs.reserve_exact(self.vertices.len());

        for vert in self.vertices.iter() {
            let (loc, attr) = self.shader.vert(vert);
            vertex_coords.push(loc);
            vertex_attrs.push(attr);
        }

        // Draw triangles with near clipping
        // TODO: If Raster is migrated to fixed point, add guard band clipping
        let mut clip_points: Vec<Vec4> = Vec::with_capacity(4);
        let mut clip_attrs: Vec<VertexData<FIP, FIA>> = Vec::with_capacity(4);
        for tri in self.tris.iter() {
            clip_points.clear();
            clip_attrs.clear();

            let mut p_accepts = [true; 3];
            for (&p, p_c) in tri.iter().zip(p_accepts.iter_mut()) {
                *p_c = vertex_coords[p].w() >= near;
            }
            if p_accepts.iter().all(|&x| !x) {
                // All Z-rejected
                continue;
            }
            if p_accepts.iter().all(|&x| x) {
                // None Z-rejected
                for &p in tri.iter() {
                    clip_points.push(vertex_coords[p]);
                    clip_attrs.push(vertex_attrs[p]);
                }
            } else {
                // Some Z-rejected
                let first_valid = p_accepts.iter().position(|&x| x).unwrap();
                clip_points.push(vertex_coords[tri[first_valid]]);
                clip_attrs.push(vertex_attrs[tri[first_valid]]);
                for i in 1..=3_isize {
                    // Current element index
                    let cei = (first_valid as isize + i).rem_euclid(3) as usize;
                    // Last element index
                    let lei = (first_valid as isize + i - 1).rem_euclid(3) as usize;
                    // If cei => lei crosses the near clipping plane
                    let crosses_near = p_accepts[cei] != p_accepts[lei];
                    if !crosses_near {
                        if !p_accepts[cei] {
                            // Two points behind near do not produce a new point
                            continue;
                        }
                        clip_points.push(vertex_coords[tri[cei]]);
                        clip_attrs.push(vertex_attrs[tri[cei]]);
                        continue;
                    }

                    let vertex = vertex_coords[tri[cei]];
                    let last_vertex = vertex_coords[tri[lei]];
                    let attrs = &vertex_attrs[tri[cei]];
                    let last_attrs = &vertex_attrs[tri[lei]];

                    // No div by zero, as crosses_near => zc != zl
                    let t = (near - last_vertex.w()) / (vertex.w() - last_vertex.w());
                    let clip_vert = last_vertex + (vertex - last_vertex) * t;
                    let clip_attr = self.shader.vd_lerp(last_attrs, attrs, t);
                    clip_points.push(clip_vert);
                    clip_attrs.push(clip_attr);
                    if p_accepts[cei] {
                        clip_points.push(vertex);
                        clip_attrs.push(*attrs);
                    }
                }
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
}

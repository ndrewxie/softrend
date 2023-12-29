mod assets;
mod matrices;
mod rasterizer;

use matrices::*;
use rasterizer::Rasterizer;

const NEAR: f32 = 1.5;
const FAR: f32 = 128.0;
const FOV: f32 = std::f32::consts::PI * 0.7;

type Tex = &'static assets::Texture;

pub struct Renderer {
    raster: Rasterizer,
    /// Window dimensions (width, height)
    window_dims: (usize, usize),
    /// (x, y, z) camera location in world coordinates
    cam_loc: (f32, f32, f32),
    /// (rotation, inclination) camera orientation
    cam_orient: (f32, f32),

    vertices: M4xn,
    vertex_attrs: Vec<[f32; 2]>,
    tris: Vec<[usize; 3]>,
    bound_tex: Tex,

    world_proj: M4x4,
    viewport: M4x4,
    mouse_sens: f32,
    move_sens: f32,
}

impl Renderer {
    pub fn new(
        width: usize,
        height: usize,
        mouse_sens: f32,
        move_sens: f32,
    ) -> Self {
        Self {
            raster: Rasterizer::new(width, height),
            window_dims: (width, height),
            cam_loc: (0.0, 0.0, 0.0),
            cam_orient: (0.0, 0.0),

            vertices: M4xn::with_capacity(128),
            tris: Vec::with_capacity(64),
            vertex_attrs: Vec::with_capacity(64),
            bound_tex: assets::TEXTURES.get("white").unwrap(),

            world_proj: M4x4::new(),
            viewport: M4x4::new(),
            mouse_sens,
            move_sens,
        }
    }
    pub fn move_cam(&mut self, dx: f32, dz: f32) {
        let forward = Vec4::from_array([dx, 0.0, dz, 1.0]) * self.move_sens;
        let forward = self.compute_cam_rot() * forward;
        self.cam_loc.0 += forward[0];
        self.cam_loc.1 += forward[1];
        self.cam_loc.2 += forward[2];
    }
    pub fn rot_cam(&mut self, dazu: f32, dinc: f32) {
        let inc_limit = std::f32::consts::PI * 0.5;

        self.cam_orient.0 += dazu * self.mouse_sens;
        self.cam_orient.1 += dinc * self.mouse_sens;
        self.cam_orient.1 =
            self.cam_orient.1.clamp(-inc_limit + 0.05, inc_limit - 0.05);
    }
    pub fn draw_frame(&mut self, time: u128) -> &Rasterizer {
        self.raster.clear();
        self.compute_camera();

        fn draw_z_test(rend: &mut Renderer, time: u128) {
            let xr = 0.5 * std::f32::consts::PI * (time as f32 / 4500.0);
            let zr = 0.5 * std::f32::consts::PI * (time as f32 / 4000.0);
            let yr = 0.5 * std::f32::consts::PI * (time as f32 / 3500.0);
            rend.draw_cube(
                [0.0, 0.0, 200.0],
                [0.0, 0.0, 0.0],
                [100.0; 3],
                assets::TEXTURES.get("joemama").unwrap(),
            );
            rend.draw_cube(
                [0.0, 0.0, 30.0],
                [xr, yr, zr],
                [6.0; 3],
                assets::TEXTURES.get("joemama").unwrap(),
            );
            rend.draw_cube(
                [0.0, 0.0, 30.0],
                [zr + 0.75, xr, yr + 2.75],
                [6.0; 3],
                assets::TEXTURES.get("checkerboard").unwrap(),
            );
        }
        fn draw_fill_test(rend: &mut Renderer, time: u128) {
            for _ in 0..10 {
                rend.raster.draw_tri(
                    [0.0, 0.0, 10.0, 0.0, 0.0],
                    [1920.0 - 1.0, 0.0, 10.0, 127.0, 0.0],
                    [1920.0 - 1.0, 1080.0 - 1.0, 10.0, 127.0, 127.0],
                    assets::TEXTURES.get("joemama").unwrap(),
                );
                rend.raster.draw_tri(
                    [1920.0 - 1.0, 1080.0 - 1.0, 10.0, 127.0, 127.0],
                    [0.0, 1080.0 - 1.0, 10.0, 0.0, 127.0],
                    [0.0, 0.0, 10.0, 0.0, 0.0],
                    assets::TEXTURES.get("joemama").unwrap(),
                );
            }
        }
        fn draw_near_clip_test(rend: &mut Renderer, time: u128) {
            for x in (-50..=50).step_by(10) {
                for y in (-50..=50).step_by(10) {
                    rend.draw_cube(
                        [x as f32, y as f32, 10.0],
                        [0.0, 0.0, 0.0],
                        [5.0; 3],
                        assets::TEXTURES.get("checkerboard").unwrap(),
                    );
                }
            }
        }
        fn draw_floor_test(rend: &mut Renderer, time: u128) {
            rend.draw_cube(
                [0.0, -100.0, 0.0],
                [0.0, 0.0, 0.0],
                [1000.0, 0.0, 1000.0],
                assets::TEXTURES.get("joemama").unwrap(),
            );
        }
        draw_near_clip_test(self, time);
        &self.raster
    }
    fn compute_camera(&mut self) {
        let max_dim =
            0.5 * std::cmp::max(self.window_dims.0, self.window_dims.1) as f32;

        let trans = render_mats::translate(
            -self.cam_loc.0,
            -self.cam_loc.1,
            -self.cam_loc.2,
        );
        let cam_rot = self.compute_cam_rot();
        let proj = render_mats::proj(NEAR, FAR, FOV);
        let z_shift = render_mats::translate(
            0.0,
            0.0,
            -render_mats::proj_z_offset(NEAR, FAR, FOV),
        );
        let screen_scale = render_mats::scale(max_dim, -max_dim, 1.0);
        let screen_center = render_mats::translate(
            0.5 * self.window_dims.0 as f32,
            0.5 * self.window_dims.1 as f32,
            0.0,
        );

        self.viewport = screen_center * screen_scale * z_shift;
        self.world_proj = proj * cam_rot * trans;
    }
    fn compute_cam_rot(&self) -> M4x4 {
        let azu_rot = render_mats::rot_y(self.cam_orient.0);
        let alt_rot = render_mats::rot_x(self.cam_orient.1);
        alt_rot * azu_rot
    }
    #[rustfmt::skip]
    fn draw_cube(&mut self, loc: [f32; 3], orientation: [f32; 3], s: [f32; 3], tex: Tex) {
        fn v4(arr: [f32; 4]) -> Vec4 {
            Vec4::from_array(arr)
        }

        let rot = render_mats::translate(loc[0], loc[1], loc[2])
            * render_mats::rot_x(orientation[0])
            * render_mats::rot_y(orientation[1])
            * render_mats::rot_z(orientation[2])
            * render_mats::translate(-loc[0], -loc[1], -loc[2]);

        self.add_vert(v4([loc[0] + s[0], loc[1] + s[1], loc[2] + s[2], 1.0]), [0.0, 0.0]); // 0
        self.add_vert(v4([loc[0] + s[0], loc[1] + s[1], loc[2] - s[2], 1.0]), [127.0, 0.0]); // 1
        self.add_vert(v4([loc[0] + s[0], loc[1] - s[1], loc[2] + s[2], 1.0]), [0.0, 127.0]); // 2
        self.add_vert(v4([loc[0] + s[0], loc[1] - s[1], loc[2] - s[2], 1.0]), [127.0, 127.0]); // 3
        self.add_vert(v4([loc[0] - s[0], loc[1] + s[1], loc[2] + s[2], 1.0]), [127.0, 0.0]); // 4
        self.add_vert(v4([loc[0] - s[0], loc[1] + s[1], loc[2] - s[2], 1.0]), [0.0, 127.0]); // 5
        self.add_vert(v4([loc[0] - s[0], loc[1] - s[1], loc[2] + s[2], 1.0]), [127.0, 0.0]); // 6
        self.add_vert(v4([loc[0] - s[0], loc[1] - s[1], loc[2] - s[2], 1.0]), [0.0, 0.0]); // 7

        self.add_quad(6, 2, 0, 4);
        self.add_quad(7, 5, 1, 3);
        self.add_quad(7, 6, 4, 5);
        self.add_quad(3, 1, 0, 2);
        self.add_quad(7, 3, 2, 6);
        self.add_quad(5, 4, 0, 1);

        self.bind_tex(assets::TEXTURES.get("joemama").unwrap());

        self.draw_model(&rot);
    }
    fn draw_model(&mut self, model: &M4x4) {
        let mut clip_points: M4xn = M4xn::with_capacity(4);
        let mut clip_tex: Vec<[f32; 2]> = Vec::with_capacity(4);
        model.mul_packed(&mut self.vertices);
        self.world_proj.mul_packed(&mut self.vertices);
        for tri in self.tris.iter() {
            clip_points.clear();
            clip_tex.clear();
            // Use w coordinate to infer clip/no clip, as projection copies
            // z to w
            let mut p_accepts = [true; 3];
            for (p, p_clip) in tri.iter().zip(p_accepts.iter_mut()) {
                *p_clip = self.vertices[*p][3] >= NEAR;
            }
            if p_accepts.iter().all(|x| !*x) {
                continue;
            }
            if p_accepts.iter().all(|x| *x) {
                // No clipping needed
                for p in tri.iter() {
                    clip_points.push(self.vertices[*p]);
                    clip_tex.push(self.vertex_attrs[*p]);
                }
            } else {
                // Clipping
                let first_valid = p_accepts.iter().position(|x| *x).unwrap();
                clip_points.push(self.vertices[tri[first_valid]]);
                clip_tex.push(self.vertex_attrs[tri[first_valid]]);
                for i in 1..=3_isize {
                    let cei = (first_valid as isize + i).rem_euclid(3) as usize;
                    let lei = (first_valid as isize + i - 1).rem_euclid(3) as usize;
                    let crosses_near = p_accepts[cei] != p_accepts[lei];
                    if !crosses_near {
                        if !p_accepts[cei] {
                            // Two points behind near do not produce a new point
                            continue;
                        }
                        clip_points.push(self.vertices[tri[cei]]);
                        clip_tex.push(self.vertex_attrs[tri[cei]]);
                        continue;
                    }

                    let vertex = self.vertices[tri[cei]];
                    let last_vertex = self.vertices[tri[lei]];
                    let attrs = self.vertex_attrs[tri[cei]];
                    let last_attrs = self.vertex_attrs[tri[lei]];

                    // No div by zero, as crosses_near => zc != zl
                    let t = (NEAR - last_vertex[3]) / (vertex[3] - last_vertex[3]);
                    let p = last_vertex + (vertex - last_vertex) * t;
                    let mut tex = last_attrs;
                    for (tex, curr_tex) in tex.iter_mut().zip(attrs) {
                        *tex = *tex + t * (curr_tex - *tex);
                    }
                    clip_points.push(p);
                    clip_tex.push(tex);
                    if p_accepts[cei] {
                        clip_points.push(vertex);
                        clip_tex.push(attrs);
                    }
                }
            }
            clip_points.w_divide();
            self.viewport.mul_packed(&mut clip_points);
            let mut win = 1;
            // Clipping a convex poly against near plane preserves convex-ness
            // so a simple fan triangulation works
            // May produce slivers, however
            while win + 1 < clip_points.n() {
                let (pa, ta) = (&clip_points[0], &clip_tex[0]);
                let (pb, tb) = (&clip_points[win], &clip_tex[win]);
                let (pc, tc) = (&clip_points[win + 1], &clip_tex[win + 1]);
                self.raster.draw_tri(
                    [pa[0], pa[1], pa[2], ta[0], ta[1]],
                    [pb[0], pb[1], pb[2], tb[0], tb[1]],
                    [pc[0], pc[1], pc[2], tc[0], tc[1]],
                    self.bound_tex,
                );
                win += 1;
            }
        }
        self.clear_obj();
    }
    pub fn add_vert(&mut self, p: Vec4, attr: [f32; 2]) -> usize {
        self.vertices.push(p);
        self.vertex_attrs.push(attr);
        self.vertices.n() - 1
    }
    pub fn add_tri(&mut self, a: usize, b: usize, c: usize) {
        self.tris.push([a, b, c]);
    }
    pub fn add_quad(&mut self, a: usize, b: usize, c: usize, d: usize) {
        self.tris.push([a, b, c]);
        self.tris.push([c, d, a])
    }
    pub fn bind_tex(&mut self, tex: Tex) {
        self.bound_tex = tex;
    }
    pub fn clear_obj(&mut self) {
        self.vertices.clear();
        self.tris.clear();
        self.vertex_attrs.clear();
    }
}

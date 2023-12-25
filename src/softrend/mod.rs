mod assets;
mod matrices;
mod rasterizer;

use matrices::*;
use rasterizer::Rasterizer;

const NEAR: f32 = 4.0;
const FAR: f32 = 128.0;
const FOV: f32 = std::f32::consts::PI * 0.5;

pub struct Renderer {
    raster: Rasterizer,
    /// Window dimensions (width, height)
    window_dims: (usize, usize),
    /// (x, y, z) camera location in world coordinates
    cam_loc: (f32, f32, f32),
    /// (rotation, inclination) camera orientation
    cam_orient: (f32, f32),

    camera: M4x4,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            raster: Rasterizer::new(width, height, NEAR / FAR),
            window_dims: (width, height),
            cam_loc: (0.0, 0.0, 0.0),
            cam_orient: (0.0, 0.0),

            camera: M4x4::new(),
        }
    }
    pub fn move_cam(&mut self, dx: f32, dy: f32, dz: f32) {
        self.cam_loc.0 += dx;
        self.cam_loc.1 += dy;
        self.cam_loc.2 += dz;
    }
    pub fn rot_cam(&mut self, dazu: f32, dinc: f32) {
        self.cam_orient.0 += dazu;
        self.cam_orient.1 += dinc;
    }
    pub fn draw_frame(&mut self, time: u128) -> &Rasterizer {
        self.raster.clear();
        self.compute_camera();
        let xr = 0.5 * std::f32::consts::PI * (time as f32 / 4500.0);
        let zr = 0.5 * std::f32::consts::PI * (time as f32 / 4000.0);
        let yr = 0.5 * std::f32::consts::PI * (time as f32 / 3500.0);
        self.draw_cube(
            [0.0, 0.0, 200.0],
            [0.0, 0.0, 0.0],
            100.0,
            assets::TEXTURES.get("joemama").unwrap(),
        );
        self.draw_cube(
            [0.0, 0.0, 30.0],
            [xr, yr, zr],
            6.0,
            assets::TEXTURES.get("joemama").unwrap(),
        );
        self.draw_cube(
            [0.0, 0.0, 30.0],
            [zr + 0.75, xr, yr + 2.75],
            6.0,
            assets::TEXTURES.get("checkerboard").unwrap(),
        );
        self.draw_cube(
            [0.0, 0.0, 30.0],
            [yr + 1.35, zr + 2.0, xr],
            6.0,
            assets::TEXTURES.get("white").unwrap(),
        );
        self.draw_cube(
            [0.0, 0.0, 30.0],
            [0.5 * zr, yr + 2.75, xr],
            6.0,
            assets::TEXTURES.get("amogus").unwrap(),
        );
        /*
        for _ in 0..10 {
            /*
            for y in (0..1080).step_by(100) {
                for x in (0..1920).step_by(100) {
                    let (x0, y0) = (x as f32, y as f32);
                    let (x1, y1) = (x0 + 100.0, y0 + 100.0);
                    self.raster.draw_tri(
                        [x0, y0, 0.0, 0.0, 0.0],
                        [x1, y0, 0.0, 127.0, 0.0],
                        [x1, y1, 127.0, 127.0, 0.0],
                        assets::TEXTURES.get("checkerboard").unwrap(),
                    );
                    self.raster.draw_tri(
                        [x1, y1, 0.0, 0.0, 0.0],
                        [x0, y1, 0.0, 127.0, 0.0],
                        [x0, y0, 127.0, 127.0, 0.0],
                        assets::TEXTURES.get("checkerboard").unwrap(),
                    );
                }
            }
            */
            self.raster.draw_tri(
                [0.0, 0.0, 10.0, 0.0, 0.0],
                [1920.0 - 1.0, 0.0, 10.0, 127.0, 0.0],
                [1920.0 - 1.0, 1080.0 - 1.0, 10.0, 127.0, 127.0],
                assets::TEXTURES.get("joemama").unwrap(),
            );
            self.raster.draw_tri(
                [1920.0 - 1.0, 1080.0 - 1.0, 10.0, 127.0, 127.0],
                [0.0, 1080.0 - 1.0, 10.0, 0.0, 127.0],
                [0.0, 0.0, 10.0, 0.0, 0.0],
                assets::TEXTURES.get("joemama").unwrap(),
            );
        }
        */
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
        let azu_rot = render_mats::rot_y(self.cam_orient.0);
        let alt_rot = render_mats::rot_x(self.cam_orient.1);
        let proj = render_mats::proj(NEAR, FAR, FOV);
        let screen_scale = render_mats::scale(max_dim, -max_dim, 1.0);
        let screen_center = render_mats::translate(
            0.5 * self.window_dims.0 as f32,
            0.5 * self.window_dims.1 as f32,
            0.0,
        );

        self.camera =
            screen_center * screen_scale * proj * alt_rot * azu_rot * trans;
    }
    pub fn draw_cube(
        &mut self,
        center: [f32; 3],
        orientation: [f32; 3],
        size: f32,
        tex: &'static assets::Texture,
    ) {
        let rot = render_mats::translate(center[0], center[1], center[2])
            * render_mats::rot_x(orientation[0])
            * render_mats::rot_y(orientation[1])
            * render_mats::rot_z(orientation[2])
            * render_mats::translate(-center[0], -center[1], -center[2]);
        let mut vertices = M4xn::from_vec(vec![
            [center[0] + size, center[1] + size, center[2] + size, 1.0], // 0
            [center[0] + size, center[1] + size, center[2] - size, 1.0], // 1
            [center[0] + size, center[1] - size, center[2] + size, 1.0], // 2
            [center[0] + size, center[1] - size, center[2] - size, 1.0], // 3
            [center[0] - size, center[1] + size, center[2] + size, 1.0], // 4
            [center[0] - size, center[1] + size, center[2] - size, 1.0], // 5
            [center[0] - size, center[1] - size, center[2] + size, 1.0], // 6
            [center[0] - size, center[1] - size, center[2] - size, 1.0], // 7
        ]);
        rot.mul_packed(&mut vertices);
        self.camera.mul_packed(&mut vertices);
        vertices.w_divide();

        // Back
        self.draw_quadface(vertices[6], vertices[2], vertices[0], vertices[4], tex);
        // Front
        self.draw_quadface(vertices[7], vertices[5], vertices[1], vertices[3], tex);

        // Left
        self.draw_quadface(vertices[7], vertices[6], vertices[4], vertices[5], tex);
        // Right
        self.draw_quadface(vertices[3], vertices[1], vertices[0], vertices[2], tex);

        // Bottom
        self.draw_quadface(vertices[7], vertices[3], vertices[2], vertices[6], tex);
        // Top
        self.draw_quadface(vertices[5], vertices[4], vertices[0], vertices[1], tex);
    }
    fn draw_quadface(
        &mut self,
        p1: [f32; 4],
        p2: [f32; 4],
        p3: [f32; 4],
        p4: [f32; 4],
        tex: &'static assets::Texture,
    ) {
        self.raster.draw_tri(
            [p1[0], p1[1], p1[2], 0.0, 0.0],
            [p2[0], p2[1], p2[2], 127.0, 0.0],
            [p3[0], p3[1], p3[2], 127.0, 127.0],
            tex,
        );
        self.raster.draw_tri(
            [p3[0], p3[1], p3[2], 127.0, 127.0],
            [p4[0], p4[1], p4[2], 0.0, 127.0],
            [p1[0], p1[1], p1[2], 0.0, 0.0],
            tex,
        );
    }
}

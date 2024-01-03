mod assets;
mod cube;
mod matrices;
mod softdraw;

use cube::*;
use matrices::*;
use softdraw::*;

const NEAR: f32 = 1.5;
const FAR: f32 = 128.0;
const FOV: f32 = std::f32::consts::PI * 0.5;

type Tex = &'static assets::Texture;

pub struct Renderer {
    raster: Raster,
    /// (x, y, z) camera location in world coordinates
    cam_loc: (f32, f32, f32),
    /// (rotation, inclination) camera orientation
    cam_orient: (f32, f32),

    camera: M4x4,
    z_offset: f32,
    mouse_sens: f32,
    move_sens: f32,
}

impl Renderer {
    pub fn new(width: usize, height: usize, mouse_sens: f32, move_sens: f32) -> Self {
        Self {
            raster: Raster::new((0, 0), (width, height)),
            cam_loc: (0.0, 0.0, 0.0),
            cam_orient: (0.0, 0.0),

            camera: M4x4::new(),
            z_offset: 0.0,
            mouse_sens,
            move_sens,
        }
    }
    pub fn move_cam(&mut self, dx: f32, dz: f32) {
        let azu_rot = render_mats::rot_y(-self.cam_orient.0);
        let alt_rot = render_mats::rot_x(-self.cam_orient.1);
        let inv_cam_rot = azu_rot * alt_rot;

        let mut forward = Vec4::from_array([0.0, 0.0, 1.0, 1.0]);
        forward = &inv_cam_rot * forward;
        let left = Vec4::from_array([0.0, 1.0, 0.0, 1.0]).cross_3d(&forward);

        let delta = (left * dx + forward * dz) * self.move_sens;
        self.cam_loc.0 += delta[0];
        self.cam_loc.1 += delta[1];
        self.cam_loc.2 += delta[2];
    }
    pub fn rot_cam(&mut self, dazu: f32, dinc: f32) {
        let inc_limit = std::f32::consts::PI * 0.5;

        self.cam_orient.0 += dazu * self.mouse_sens;
        self.cam_orient.1 += dinc * self.mouse_sens;
        self.cam_orient.1 = self.cam_orient.1.clamp(-inc_limit + 0.05, inc_limit - 0.05);
    }
    pub fn draw_frame<B>(&mut self, time: u128, buffer: &mut B)
    where
        B: std::ops::DerefMut<Target = [u32]>,
    {
        self.raster.clear();
        self.compute_camera();

        #[allow(dead_code)]
        fn draw_near_clip_test(rend: &mut Renderer, time: u128) {
            let xr = 0.5 * std::f32::consts::PI * (time as f32 / 4500.0);
            let zr = 0.5 * std::f32::consts::PI * (time as f32 / 4000.0);
            let yr = 0.5 * std::f32::consts::PI * (time as f32 / 3500.0);
            for x in (-100..=100).step_by(10) {
                for y in (-100..=100).step_by(10) {
                    rend.draw_cube(
                        [x as f32, y as f32, 20.0],
                        [xr, yr, zr],
                        [5.0; 3],
                        assets::TEXTURES.get("joemama").unwrap(),
                    );
                }
            }
        }
        #[allow(dead_code)]
        fn draw_setup_stress_test(rend: &mut Renderer, _time: u128) {
            let rot = std::f32::consts::PI * 0.25;
            for x in (-100..=100).step_by(1) {
                for y in (-100..=100).step_by(1) {
                    rend.draw_cube(
                        [x as f32 * 0.5, y as f32 * 0.5, 20.0],
                        [rot, rot, rot],
                        [0.5; 3],
                        assets::TEXTURES.get("joemama").unwrap(),
                    );
                }
            }
        }
        #[allow(dead_code)]
        fn draw_floor_test(rend: &mut Renderer, _time: u128) {
            rend.draw_cube(
                [0.0, -50.0, 0.0],
                [0.0, 0.0, 0.0],
                [1000.0, 0.0, 1000.0],
                assets::TEXTURES.get("amogus").unwrap(),
            );
        }

        //draw_setup_stress_test(self, time);
        draw_near_clip_test(self, time);
        //draw_floor_test(self, time);
        self.raster.copy_to_brga_u32(buffer);
    }
    fn compute_camera(&mut self) {
        let trans = render_mats::translate(-self.cam_loc.0, -self.cam_loc.1, -self.cam_loc.2);
        let azu_rot = render_mats::rot_y(self.cam_orient.0);
        let alt_rot = render_mats::rot_x(self.cam_orient.1);
        let cam_rot = alt_rot * azu_rot;
        let proj = render_mats::proj(NEAR, FAR, FOV);

        self.camera = proj * cam_rot * trans;
        self.z_offset = render_mats::proj_z_offset(NEAR, FAR, FOV);
    }
    #[rustfmt::skip]
    fn draw_cube(&self, loc: [f32; 3], orientation: [f32; 3], scale: [f32; 3], tex: Tex) {
        let cube = Cube::new(loc, orientation, scale, tex, self.camera.clone(), self.z_offset);
        cube.draw(&self.raster);
    }
}

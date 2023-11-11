mod fixedpoint;
use fixedpoint::Fx;

pub struct Renderer {
    pixels: Vec<u8>,
    z_buf: Vec<f32>,
    width: usize,
    height: usize,
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            pixels: vec![0; 4 * width * height],
            z_buf: vec![0.0; width * height],
            width,
            height,
        }
    }
    pub fn draw(&mut self, time: usize) -> &[u8] {
        for _ in 0..10 {
            self.draw_tri((0.0, 0.0), (1920.0-1.0, 0.0), (1920.0-1.0, 1080.0-1.0), 255, 255, 0);
            self.draw_tri((1920.0-1.0, 1080.0-1.0), (0.0, 1080.0-1.0), (0.0, 0.0), 0, 255, 255);
        }
        &self.pixels
    }
    pub fn draw_tri(&mut self, a: (f32, f32), b: (f32, f32), c: (f32, f32), cr: u8, cg: u8, cb: u8) {
        let (ax, ay) = (Fx::<4>::from(a.0), Fx::<4>::from(a.1));
        let (bx, by) = (Fx::<4>::from(b.0), Fx::<4>::from(b.1));
        let (cx, cy) = (Fx::<4>::from(c.0), Fx::<4>::from(c.1));

        let start_x = std::cmp::min(std::cmp::min(ax, bx), cx);
        let start_y = std::cmp::min(std::cmp::min(ay, by), cy);
        let end_x = std::cmp::max(std::cmp::max(ax, bx), cx);
        let end_y = std::cmp::max(std::cmp::max(ay, by), cy);
        let mut x = start_x;
        let mut y = start_y;

        let ab = Self::setup_edge_func((ax, ay), (bx, by), (x, y));
        let bc = Self::setup_edge_func((bx, by), (cx, cy), (x, y));
        let ca = Self::setup_edge_func((cx, cy), (ax, ay), (x, y));
        
        let mut edge_funcs = [ab.0, bc.0, ca.0];
        let dx = [ab.1.0, bc.1.0, ca.1.0];
        let dy = [ab.1.1, bc.1.1, ca.1.1];

        let mut index = y.as_i32() as usize * self.width + x.as_i32() as usize;

        while y <= end_y {
            let mut edge_funcs_row = edge_funcs.clone();
            let mut row_index = index;
            while x <= end_x {
                if edge_funcs_row.iter().all(|x| !x.is_neg()) {
                    self.pixels[row_index] = cr;
                    self.pixels[row_index+1] = cg;
                    self.pixels[row_index+2] = cb;
                    self.pixels[row_index+3] = 0;
                }
                edge_funcs_row.iter_mut().zip(dx.iter()).for_each(|(x, dx)| *x = *x + *dx);
                x = x + 1;
                row_index += 4;
            }
            y = y + 1;
            x = start_x;
            index += 4 * self.width;
            edge_funcs.iter_mut().zip(dy.iter()).for_each(|(x, dy)| *x = *x + *dy);
        }
    }
    fn setup_edge_func(a: (Fx<4>, Fx<4>), b: (Fx<4>, Fx<4>), p: (Fx<4>, Fx<4>)) -> (Fx<4>, (Fx<4>, Fx<4>)) {
        let val = p.0 * (a.1 - b.1) + p.1 * (b.0 - a.0) + a.0 * (b.1 - a.1) + a.1 * (a.0 - b.0);
        let dx = a.1 - b.1;
        let dy = b.0 - a.0;
        (val, (dx, dy))
    }
}
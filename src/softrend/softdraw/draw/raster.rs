use super::config::*;
use std::cell::{RefCell, RefMut};

pub struct Raster {
    pixels: RefCell<&'static mut [u32]>,
    z_buf: RefCell<&'static mut [f32]>,
    fb_dims: (usize, usize),
    screen_dims: (usize, usize),
    raster_tl: (usize, usize),
}

impl Raster {
    pub fn new(raster_tl: (usize, usize), screen_dims: (usize, usize)) -> Self {
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
impl std::fmt::Debug for Raster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Raster")
    }
}

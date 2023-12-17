#![feature(portable_simd)]

use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Fullscreen, WindowBuilder};

mod softrend;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Rc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let monitor = event_loop.available_monitors().next().expect("no monitor found!");
    let fullscreen = Some(Fullscreen::Borderless(Some(monitor.clone())));
    window.set_fullscreen(fullscreen);

    let context = unsafe { softbuffer::Context::new(&window).unwrap() };
    let mut surface =
        unsafe { softbuffer::Surface::new(&context, &window).unwrap() };

    let size = window.inner_size();
    let (Some(width), Some(height)) =
        (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
    else {
        return;
    };
    surface.resize(width, height).unwrap();

    let game_start = Instant::now();
    let mut renderer =
        softrend::Renderer::new(size.width as usize, size.height as usize);

    let mut last_frame = Instant::now();
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Wait);
            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::RedrawRequested => {
                        let fps = 1000 / (last_frame.elapsed().as_millis() + 1);
                        println!("fps: {}", fps);
                        last_frame = Instant::now();

                        let mut buffer: softbuffer::Buffer<'_> =
                            surface.buffer_mut().unwrap();
                        renderer
                            .draw_frame(game_start.elapsed().as_millis())
                            .copy_to_brga_u32(&mut buffer);

                        buffer.present().unwrap();
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed && !event.repeat {
                            println!("Pressed {:?}", event.physical_key);
                            match event.physical_key {
                                PhysicalKey::Code(KeyCode::Escape) => {
                                    elwt.exit();
                                }
                                PhysicalKey::Code(KeyCode::KeyW) => {
                                    renderer.move_cam(0.0, 0.0, 1.0);
                                }
                                PhysicalKey::Code(KeyCode::KeyA) => {
                                    renderer.move_cam(-1.0, 0.0, 0.0);
                                }
                                PhysicalKey::Code(KeyCode::KeyD) => {
                                    renderer.move_cam(1.0, 0.0, 0.0);
                                }
                                PhysicalKey::Code(KeyCode::KeyS) => {
                                    renderer.move_cam(0.0, 0.0, -1.0);
                                }
                                PhysicalKey::Code(KeyCode::KeyQ) => {
                                    renderer.move_cam(0.0, 1.0, 0.0);
                                }
                                PhysicalKey::Code(KeyCode::KeyE) => {
                                    renderer.move_cam(0.0, -1.0, 0.0);
                                }
                                PhysicalKey::Code(KeyCode::ArrowRight) => {
                                    renderer
                                        .rot_cam(-std::f32::consts::PI * 0.05, 0.0);
                                }
                                PhysicalKey::Code(KeyCode::ArrowLeft) => {
                                    renderer
                                        .rot_cam(std::f32::consts::PI * 0.05, 0.0);
                                }
                                PhysicalKey::Code(KeyCode::ArrowUp) => {
                                    renderer
                                        .rot_cam(0.0, std::f32::consts::PI * 0.05);
                                }
                                PhysicalKey::Code(KeyCode::ArrowDown) => {
                                    renderer
                                        .rot_cam(0.0, -std::f32::consts::PI * 0.05);
                                }
                                _ => (),
                            }
                        }
                    }
                    _ => (),
                }
            }
        })
        .unwrap();
}

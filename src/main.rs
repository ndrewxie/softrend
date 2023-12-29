#![feature(portable_simd)]

use std::collections::HashMap;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, WindowBuilder};

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
    let mut renderer = softrend::Renderer::new(
        size.width as usize,
        size.height as usize,
        0.001,
        50.0,
    );

    window.set_cursor_grab(CursorGrabMode::Confined).expect("Cannot grab cursor");
    window.set_cursor_visible(false);

    let mut last_frame = Instant::now();
    let mut player_move_state = (0.0, 0.0);
    let mut key_holds = HashMap::new();
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Wait);
            if let Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } = &event
            {
                renderer.rot_cam(delta.0 as f32, delta.1 as f32);
            }
            if let Event::WindowEvent { event, .. } = &event {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::RedrawRequested => {
                        let fps =
                            1000.0 / (last_frame.elapsed().as_millis() as f32 + 1.0);
                        renderer.move_cam(
                            player_move_state.0 / fps,
                            player_move_state.1 / fps,
                        );
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
                        if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            elwt.exit();
                        }
                        let key_map = HashMap::from([
                            (PhysicalKey::Code(KeyCode::KeyW), (0.0, 1.0)),
                            (PhysicalKey::Code(KeyCode::KeyS), (0.0, -1.0)),
                            (PhysicalKey::Code(KeyCode::KeyA), (-1.0, 0.0)),
                            (PhysicalKey::Code(KeyCode::KeyD), (1.0, 0.0)),
                        ]);
                        let key_delta = key_map.get(&event.physical_key);
                        match (event.state, key_delta) {
                            (ElementState::Pressed, Some(delta)) => {
                                if !*key_holds
                                    .entry(event.physical_key)
                                    .or_insert(false)
                                {
                                    key_holds.insert(event.physical_key, true);
                                    player_move_state.0 += delta.0;
                                    player_move_state.1 += delta.1;
                                }
                            }
                            (ElementState::Released, Some(delta)) => {
                                if *key_holds
                                    .entry(event.physical_key)
                                    .or_insert(true)
                                {
                                    key_holds.insert(event.physical_key, false);
                                    player_move_state.0 -= delta.0;
                                    player_move_state.1 -= delta.1;
                                }
                            }
                            _ => (),
                        }
                    }
                    _ => (),
                }
            }
        })
        .unwrap();
}

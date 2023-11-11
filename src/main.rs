use std::num::NonZeroU32;
use std::rc::Rc;
use std::time::Instant;
use winit::event::{ElementState, Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::{PhysicalKey, KeyCode};
use winit::window::{Fullscreen, WindowBuilder};

mod softrend;

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window = Rc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let monitor = event_loop
        .available_monitors()
        .next()
        .expect("no monitor found!");
    let fullscreen = Some(Fullscreen::Borderless(Some(monitor.clone())));
    window.set_fullscreen(fullscreen);

    let context = unsafe { softbuffer::Context::new(&window).unwrap() };
    let mut surface = unsafe { softbuffer::Surface::new(&context, &window).unwrap() };

    let size = window.inner_size();
    let (Some(width), Some(height)) = (NonZeroU32::new(size.width), NonZeroU32::new(size.height))
    else {
        return;
    };
    surface.resize(width, height).unwrap();

    let mut renderer = softrend::Renderer::new(size.width as usize, size.height as usize);

    let app_start = Instant::now();
    let mut last_frame = Instant::now();
    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Wait);

            if let Event::WindowEvent { event, .. } = event {
                match event {
                    WindowEvent::CloseRequested => elwt.exit(),
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        println!("fps: {}", 1000 / ((now - last_frame).as_millis() + 1));
                        last_frame = now;

                        let pixels = renderer.draw((now - app_start).as_millis() as usize);
                        let mut buffer = surface.buffer_mut().unwrap();
                        for (c_in, p_out) in pixels.chunks_exact(4).zip(buffer.iter_mut()) {
                            let red = c_in[0] as u32;
                            let green = c_in[1] as u32;
                            let blue = c_in[2] as u32;
                            *p_out = blue | (green << 8) | (red << 16);
                        }
                        buffer.present().unwrap();
                        window.request_redraw();
                    }
                    WindowEvent::KeyboardInput { event, .. } => {
                        if event.state == ElementState::Pressed && !event.repeat {
                            println!("Pressed {:?}", event.physical_key);
                            if (event.physical_key == PhysicalKey::Code(KeyCode::Escape)) {
                                elwt.exit();
                            }
                        }
                    }
                    _ => (),
                }
            }
        })
        .unwrap();
}

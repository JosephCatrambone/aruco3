
use aruco3::{ARDictionary, Detector, DetectorConfig};
use imageproc;
use image;
use kamera::Camera;
use minifb::{Key, Window, WindowOptions};
use std::time::{Instant, Duration};

fn main() {
	let camera = Camera::new_default_device();
	camera.start();

	// Get our detector ready:
	let detector = Detector {
		config: Default::default(),
		dictionary: ARDictionary::new_from_named_dict("ARUCO"),
	};

	// Read a starter frame:
	let Some(frame) = camera.wait_for_frame() else { return }; // always blockingly waiting for next new frame
	let (w, h) = frame.size_u32();

	// Allocate and open our window:
	let mut window_buffer: Vec<u32> = vec![0; (w * h) as usize];
	let mut window = Window::new("Test - ESC to exit", w as usize, h as usize, WindowOptions::default(),).expect("Failed to open system window.");

	let mut img = image::RgbaImage::from_pixel(1, 1, [0, 0, 0, 0].into());
	while window.is_open() && !window.is_key_down(Key::Escape) {
		let Some(frame) = camera.wait_for_frame() else { return };
		let (new_w, new_h) = frame.size_u32();
		if new_w != w || new_h != h { return; } // Image size changed. Abort.

		// Images come in as ARGB.
		let data = frame.data();
		let buffer = data.data_u8();
		//let argb_image = image::RgbaImage::from_raw(w, h, buffer.to_vec()).expect("Failed to unwrap camera image.");
		// Copy the buffer to our image AND to our window.
		img = image::RgbaImage::from_fn(w, h, |x, y| {
			let idx = ((x+(y*w))*4) as usize;
			let a = buffer[idx+3];
			let r = buffer[idx+2];
			let g = buffer[idx+1];
			let b = buffer[idx+0];
			// Copy the image data to the screen and return a pixel.
			if let Some(pixel) = window_buffer.get_mut((x+y*w) as usize) {
				// 0xAARRGGBB
				//*pixel = (r as u32) << 24 | (g as u32) << 16 | (b as u32) << 8 | (a as u32);
				*pixel = (a as u32) << 24 | (r as u32) << 16 | (g as u32) << 8 | (b as u32);
			}

			image::Rgba([r, g, b, a].into())
		});

		// Detect:
		let start_detection = Instant::now();
		let detections = detector.detect(img.into());
		let end_detection = Instant::now();
		println!("Detection took {:?} for {} markers.", (end_detection - start_detection), detections.markers.len());
		for d in detections.markers.iter() {
			for (x, y) in &d.corners {
				if let Some(pixel) = window_buffer.get_mut((x + (y*w)) as usize) {
					*pixel = 0xFFFF00FF;
				}
			}
		}

		window
			.update_with_buffer(&window_buffer, w as usize, h as usize)
			.expect("Failed to update window buffer.");

	}
	//img.save("test.png").expect("Saved PNG.");

	camera.stop() // or drop it
}
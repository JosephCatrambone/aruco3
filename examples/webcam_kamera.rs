
use aruco3::{ARDictionary, Detector, DetectorConfig, pose};
use imageproc;
use image;
use kamera::Camera;
use minifb::{Key, Window, WindowOptions};
use std::time::Instant;

fn main() {
	let camera = Camera::new_default_device();
	camera.start();

	// Get our detector ready:
	let detector = Detector {
		config: DetectorConfig::default(),
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
			for i in 0..4 {
				lazy_line(&d.corners[i], &d.corners[(i+1)%4], 0xFFFF00FF, &mut window_buffer, w, h);
			}
		}

		// Compute pose:
		let marker_points = vec![(0.0, 0.0, 0.0f32), (1.0f32, 0.0, 0.0), (0.0, 1.0f32, 0.0), (0.0, 0.0, 1.0f32)];
		for d in detections.markers.iter() {
			let (pose1, _) = pose::solve_with_undistorted_points(&d.corners, 40.0f32, (w, h));
			let unproj_pts = pose1.apply_transform_to_points(&marker_points);
			draw_axes(&unproj_pts, &mut window_buffer, w, h);
		}

		window
			.update_with_buffer(&window_buffer, w as usize, h as usize)
			.expect("Failed to update window buffer.");

	}
	//img.save("test.png").expect("Saved PNG.");

	camera.stop() // or drop it
}

fn lazy_line(start: &(u32, u32), end: &(u32, u32), color: u32, buffer: &mut Vec<u32>, buffer_width: u32, buffer_height: u32) {
	// This is dumb and almost certainly slower than bresnehan's algorithm, but it's simple.
	let mut dx = end.0 as f32 - start.0 as f32;
	let mut dy = end.1 as f32 - start.1 as f32;
	let steps = (dx*dx + dy*dy).sqrt().ceil().max(1.0f32);
	let mut px = start.0 as f32;
	let mut py = start.1 as f32;
	dx /= steps;
	dy /= steps;
	for _ in 0..(steps as u32) {
		if px < 0f32 || py < 0f32 || px > buffer_width as f32 || py > buffer_height as f32 { continue; } // Since the corner detections _have_ to be inside the image this should be safe, but...
		let idx = px as u32 + (py as u32 * buffer_width);
		if let Some(pixel) = buffer.get_mut(idx as usize) {
			*pixel = color;
		}
		px += dx;
		py += dy;
	}
}

fn draw_axes(axes: &Vec<(f32, f32, f32)>, buffer: &mut Vec<u32>, buffer_width: u32, buffer_height: u32) {
	let to_u32 = |p: &(f32, f32, f32)| { (p.0.max(0.0) as u32, p.1.max(0.0) as u32) };
	lazy_line(&to_u32(&axes[0]), &to_u32(&axes[1]), 0xFFFF0000, buffer, buffer_width, buffer_height);
	lazy_line(&to_u32(&axes[0]), &to_u32(&axes[2]), 0xFF00FF00, buffer, buffer_width, buffer_height);
	lazy_line(&to_u32(&axes[0]), &to_u32(&axes[3]), 0xFF0000FF, buffer, buffer_width, buffer_height);
}
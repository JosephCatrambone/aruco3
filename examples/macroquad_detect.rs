use aruco3::{ARDictionary, Detection, Detector, DetectorConfig, PoseEstimator};
use macroquad::prelude::*;
use image as image_rs;
use imageproc;

const MARKER_IMAGE_SIZE: u16 = 512;
const FRAMES_BETWEEN_ESTIMATES: u8 = 30;
const MARKER_ID: usize = 69;

#[macroquad::main("3D")]
async fn main() {
	//let rust_logo = load_texture("examples/rust.png").await.unwrap();
	let detector = Detector {
		config: DetectorConfig {
			..Default::default()
		},
		dictionary: ARDictionary::new_from_named_dict("ARUCO_DEFAULT")
	};
	let mut last_detection = Detection::default();

	// Make the marker we want at the origin:
	let mut aruco_marker = Image::gen_image_color(MARKER_IMAGE_SIZE, MARKER_IMAGE_SIZE, BLACK); // Image::empty()?
	let (marker_width, marker_bits) = detector.dictionary.make_binary_image(MARKER_ID);
	let square_size = MARKER_IMAGE_SIZE / marker_width as u16;
	for y in 0..marker_width {
		for x in 0..marker_width {
			let x_offset = x as u16 * square_size;
			let y_offset = y as u16 * square_size;
			if marker_bits[(x + y*marker_width) as usize] {
				for dy in 0..square_size {
					for dx in 0..square_size {
						aruco_marker.set_pixel((x_offset + dx) as u32, (y_offset + dy) as u32, WHITE);
					}
				}
			}
		}
	}
	let aruco_texture = Texture2D::from_image(&aruco_marker);

	// Prep a render texture that we can pass to the detector every other frame.
	let render_target = render_target(screen_width() as u32, screen_height() as u32);
	render_target.texture.set_filter(FilterMode::Nearest);
	let mut camera_image: image_rs::RgbaImage = image_rs::RgbaImage::default();

	// Then keep track of the current frame and time so we can spin around.
	let mut time: f32 = 0.0f32;
	let mut frames_to_next_estimate: u8 = FRAMES_BETWEEN_ESTIMATES;

	loop {
		clear_background(LIGHTGRAY);

		let camera: Camera3D = Camera3D {
			position: vec3(20f32 * time.cos(), 15., 20f32 * time.sin()),
			up: vec3(0., 1., 0.),
			target: vec3(0., 0., 0.),
			//render_target: Some(render_target.clone()), // NOTE: We may be capturing this upside down!
			..Default::default()
		};
		set_camera(&camera);

		draw_grid(20, 1., BLACK, GRAY);
		draw_orientation_marker(20f32, 0f32, 20f32);

		//draw_cube_wires(vec3(0., 1., -6.), vec3(2., 2., 2.), DARKGREEN);
		draw_plane(vec3(0., 0., 0.1), vec2(5., 5.), Some(&aruco_texture), WHITE);
		//draw_cube(vec3(-5., 1., -2.), vec3(2., 2., 2.), Some(&rust_logo), WHITE,);

		frames_to_next_estimate = frames_to_next_estimate.saturating_sub(1);
		if frames_to_next_estimate == 0 {
			frames_to_next_estimate = FRAMES_BETWEEN_ESTIMATES;
			unsafe {
				get_internal_gl().flush();
			}
			render_target.texture.grab_screen();
			let texture_data = render_target.texture.get_texture_data();
			camera_image = image_rs::RgbaImage::from_raw(texture_data.width as u32, texture_data.height as u32, texture_data.bytes).expect("Failed to decode image from GL capture.");
			flip_image(&mut camera_image); // The image is flipped.
			//camera_image.save("output.png").expect("Failed to save image.");
			//return;
			last_detection = detector.detect(camera_image.into());
		}

		// Back to screen space, render some text
		set_default_camera();
		let cam_pos = camera.position;
		let text = format!("Ground Truth: Marker ID {MARKER_ID}, Camera Position: {cam_pos}");
		draw_text(&text, 10.0, 20.0, 12.0, BLACK);
		for m in last_detection.markers.iter() {
			draw_text(format!("ID: {}", m.id).as_str(), m.corners[0].0 as f32, m.corners[0].1 as f32, 12.0, RED);
			for (a, b) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
				draw_line(m.corners[a].0 as f32, m.corners[a].1 as f32, m.corners[b].0 as f32, m.corners[b].1 as f32, 1.0f32, RED);
			}
		}

		time += get_frame_time().min(0.05f32);
		next_frame().await
	}
}

fn flip_image(im: &mut image_rs::RgbaImage) {
	for y in 0..im.height()/2 {
		for x in 0..im.width() {
			let top = im.get_pixel(x, y).clone();
			let bottom = im.get_pixel(x, im.height()-1-y).clone();
			im.put_pixel(x, im.height()-1-y, top);
			im.put_pixel(x, y, bottom);
		}
	}
}

fn draw_orientation_marker(x: f32, y: f32, z: f32) {
	// Draw four points for our coodinate system.
	// +Z is blue, +Y is green, +X is red.
	draw_sphere(vec3(x, y, z), 0.2f32, None, BLACK);
	draw_sphere(vec3(x+1f32, y, z), 0.2f32, None, RED);
	draw_sphere(vec3(x, y+1f32, z), 0.2f32, None, GREEN);
	draw_sphere(vec3(x, y, z+1f32), 0.2f32, None, BLUE);
}
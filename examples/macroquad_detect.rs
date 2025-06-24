use aruco3::{ARDictionary, Detection, Detector, DetectorConfig, estimate_pose};
use macroquad::prelude::*;
use image as image_rs;
use imageproc;

const MOUSE_SENSITIVITY: f32 = 0.1;
const MOVE_SPEED: f32 = 0.1;
const MARKER_IMAGE_SIZE: u16 = 512;
const MARKER_ID: usize = 69;
const MARKER_SIZE: f32 = 0.10f32;
const CAMERA_FOV: f32 = 45.0f32.to_radians();

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

	// Track the mouse so we can move about:
	let WORLD_UP = vec3(0.0, 1.0, 0.0);
	let mut mouse_grabbed = false;
	let mut last_mouse_position: Vec2 = mouse_position().into();
	let mut camera_position: Vec3 = vec3(0.0f32, 0.0f32, 0.0f32);
	let mut camera_rot_x = 0.0f32;
	let mut camera_rot_z = 0.0f32;

	loop {
		let delta = get_frame_time().min(0.2f32); // Prevent big jumps by saying the max frame time is 200ms.
		clear_background(LIGHTGRAY);

		// Input handling for mouse move.
		let forward = vec3(
			camera_rot_z.cos() * camera_rot_x.cos(),
			camera_rot_x.sin(),
			camera_rot_z.sin() * camera_rot_x.cos(),
		).normalize();
		let right = forward.cross(WORLD_UP).normalize();
		let up = right.cross(forward).normalize();

		if is_key_released(KeyCode::Escape) {
			mouse_grabbed = !mouse_grabbed;
			set_cursor_grab(mouse_grabbed);
			show_mouse(!mouse_grabbed);
		}
		if mouse_grabbed {
			let mouse_position: Vec2 = mouse_position().into();
			let mouse_delta = mouse_position - last_mouse_position;

			last_mouse_position = mouse_position;

			camera_rot_z += mouse_delta.x * delta * MOUSE_SENSITIVITY;
			camera_rot_x += mouse_delta.y * delta * -MOUSE_SENSITIVITY;
			
			camera_rot_x = camera_rot_x.clamp(-1.5, 1.5);

			if is_key_down(KeyCode::Up) || is_key_down(KeyCode::W) {
				camera_position += forward * MOVE_SPEED * delta;
			}
			if is_key_down(KeyCode::Down) || is_key_down(KeyCode::S) {
				camera_position -= forward * MOVE_SPEED * delta;
			}
			if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
				camera_position -= right * MOVE_SPEED * delta;
			}
			if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
				camera_position += right * MOVE_SPEED * delta;
			}
			if is_key_down(KeyCode::E) {
				camera_position += WORLD_UP * MOVE_SPEED * delta;
			}
			if is_key_down(KeyCode::Q) {
				camera_position -= WORLD_UP * MOVE_SPEED * delta;
			}
		}

		let camera: Camera3D = Camera3D {
			//position: vec3(2f32 * time.cos(), 1. + 0.5 * (3.0 * time).cos(), 2f32 * time.sin()),
			position: camera_position,
			up: up,
			target: forward,
			fovy: CAMERA_FOV * (screen_height()/screen_width()),
			//render_target: Some(render_target.clone()), // NOTE: We may be capturing this upside down!
			..Default::default()
		};
		set_camera(&camera);

		draw_grid(20, 1., DARKGRAY, SKYBLUE);
		draw_orientation_marker(20f32, 0f32, 20f32);

		//draw_cube_wires(vec3(0., 1., -6.), vec3(2., 2., 2.), DARKGREEN);
		draw_plane(vec3(0., 0.01, 0.), vec2(MARKER_SIZE, MARKER_SIZE), Some(&aruco_texture), WHITE);
		//draw_cube(vec3(-5., 1., -2.), vec3(2., 2., 2.), Some(&rust_logo), WHITE,);

		if is_mouse_button_pressed(MouseButton::Left) {
			unsafe {
				get_internal_gl().flush();
			}
			// This is the slowest operation of all of them.
			// Grabbing a screen capture from the GPU, flipping it, and converting to a CPU image is even slower than detection.
			render_target.texture.grab_screen();
			let texture_data = render_target.texture.get_texture_data();
			camera_image = image_rs::RgbaImage::from_raw(texture_data.width as u32, texture_data.height as u32, texture_data.bytes).expect("Failed to decode image from GL capture.");
			flip_image(&mut camera_image); // The image is flipped.
			//camera_image.save("output.png").expect("Failed to save image.");
			last_detection = detector.detect(camera_image.into());
		}

		// Back to screen space, render some text
		// Note that we solve every frame.
		set_default_camera();
		let cam_pos = camera.position;
		let mut text = format!("Ground Truth: Camera Position: {cam_pos}");
		draw_text(&text, 10.0, 20.0, 12.0, BLACK);
		for m in last_detection.markers.iter() {
			draw_text(format!("ID: {}", m.id).as_str(), m.corners[0].0 as f32, m.corners[0].1 as f32, 12.0, RED);
			for (a, b) in [(0usize, 1usize), (1, 2), (2, 3), (3, 0)] {
				draw_line(m.corners[a].0 as f32, m.corners[a].1 as f32, m.corners[b].0 as f32, m.corners[b].1 as f32, 2.0f32, RED);
			}
			let (pose_best, pose_alt) = estimate_pose(last_detection.grey.clone().expect("Missing image in marker cap").dimensions(), &m.corners, MARKER_SIZE);
			// Swap Y and Z because our coordinate systems are different.
			let estimated_position = vec3(pose_best.translation.x, pose_best.translation.z, pose_best.translation.y);
			text = format!("Estimated: Camera Position: {}", estimated_position);
			draw_text(&text, 10.0, 30.0, 12.0, BLACK);

			text = format!("Estimated: Alt Camera Position: {}, {}, {}", pose_alt.translation.x, pose_alt.translation.z, pose_alt.translation.y);
			draw_text(&text, 10.0, 40.0, 12.0, BLACK);

			text = format!("Error: Position: {}", estimated_position.distance(cam_pos));
			draw_text(&text, 10.0, 50.0, 12.0, BLACK);

			draw_sphere_wires(estimated_position, 1.0f32, None, PINK);
		}

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
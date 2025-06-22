use aruco3::{ARDictionary, Detector, DetectorConfig, PoseEstimator};
use macroquad::prelude::*;
use image as image_rs;

const MARKER_IMAGE_SIZE: u16 = 512;

#[macroquad::main("3D")]
async fn main() {
	//let rust_logo = load_texture("examples/rust.png").await.unwrap();
	let detector = Detector {
		config: DetectorConfig {
			..Default::default()
		},
		dictionary: ARDictionary::new_from_named_dict("ARUCO_DEFAULT")
	};

	// Make the marker we want at the origin:
	let mut aruco_marker = Image::gen_image_color(MARKER_IMAGE_SIZE, MARKER_IMAGE_SIZE, BLACK); // Image::empty()?
	let (marker_width, marker_bits) = detector.dictionary.make_binary_image(69);
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
	let render_target = render_target(1920, 1080);
	render_target.texture.set_filter(FilterMode::Nearest);

	let mut time: f32 = 0.0f32;
	let mut run_pose_estimate: bool = false;  // Every other frame we want to draw our estimated marker pose.

	loop {
		clear_background(LIGHTGRAY);

		let camera: Camera3D = Camera3D {
			position: vec3(20f32 * time.cos(), 15., 20f32 * time.sin()),
			up: vec3(0., 1., 0.),
			target: vec3(0., 0., 0.),
			render_target: Some(render_target.clone()), // NOTE: We may be capturing this upside down!
			..Default::default()
		};
		set_camera(&camera);

		draw_grid(20, 1., BLACK, GRAY);
		
		// Draw four points for our coodinate system.
		// +Z is blue, +Y is green, +X is red.
		draw_sphere(vec3(20f32, 0f32, 20f32), 0.5f32, None, BLACK);
		draw_sphere(vec3(21f32, 0f32, 20f32), 0.5f32, None, RED);
		draw_sphere(vec3(20f32, 1f32, 20f32), 0.5f32, None, GREEN);
		draw_sphere(vec3(20f32, 0f32, 21f32), 0.5f32, None, BLUE);
		
		//draw_cube_wires(vec3(0., 1., -6.), vec3(2., 2., 2.), DARKGREEN);
		draw_plane(vec3(0., 0., 0.1), vec2(5., 5.), Some(&aruco_texture), WHITE);
		//draw_cube(vec3(-5., 1., -2.), vec3(2., 2., 2.), Some(&rust_logo), WHITE,);

		if run_pose_estimate {
			// We visualized our output last time, so this frame we need to read our image.
			//let camera_image = render_target.texture.get_texture_data();
			//let rgba = image_rs::RgbaImage::from_raw(camera_image.width as u32, camera_image.height as u32, camera_image.bytes).expect("Failed to unwrap camera image.");
			//rgba.save("output.png").expect("Failed to save image.");
			//return;
		} else {

		}
		run_pose_estimate = !run_pose_estimate;

		// Back to screen space, render some text

		set_default_camera();
		clear_background(WHITE);
		draw_texture_ex(
			&render_target.texture,
			0.,
			0.,
			WHITE,
			DrawTextureParams {
				dest_size: Some(vec2(screen_width(), screen_height())),
				flip_y: false,
				..Default::default()
			},
		);
		draw_text("Camera Position: ", 10.0, 20.0, 30.0, BLACK);
		
		time += get_frame_time();
		next_frame().await
	}
}
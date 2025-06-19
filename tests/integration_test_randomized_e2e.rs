mod common;

use common::render_image;

#[test]
fn sanity_test() {
	let mut img = render_image(10f32, 10f32, 0f32, 0f32, 0f32, 0f32);
	img.save("output_10_10.png").unwrap();
	let mut img = render_image(10f32, 10f32, 0f32, 5f32.to_radians(), 0f32, 0f32);
	img.save("output_10_10_0_5deg_0deg_0deg.png").unwrap();
}
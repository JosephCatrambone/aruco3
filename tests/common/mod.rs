
mod software_renderer;

// For testing the test:
use software_renderer::{Transform, Triangle, Float3};

// For export for the actual integration test:
pub use software_renderer::render_image;

#[test]
fn test_sub_impl_order() {
	assert_eq!(Float3::new(1.0, 2.0, 3.0) - Float3::new_const(3.0), Float3::new(-2.0, -1.0, 0.0));
}

#[test]
fn test_cross() {
	assert_eq!(Float3 { x: 1.0, y: 0.0, z: 0.0 }.cross(&Float3 {x: 0.0, y: 1.0, z: 0.0 }), Float3 { x: 0.0, y: 0.0, z: 1.0 });
}

#[test]
fn test_triangle() {
	// Test area fn:
	// CLOCKWISE!
	let t = Triangle::new(
		Float3::new(0.0, 0.0, 0.0),
		Float3::new(2.0, 0.0, 0.0),
		Float3::new(0.0, 2.0, 0.0),
	);
	assert_eq!(t.area(), 2.0);
	assert!(Triangle::point_right_of_line(&Float3::new(10f32, 0f32, 0f32), &Float3::new(10f32, 10f32, 0f32), &Float3::new(5f32, 0.1f32, 0f32)));
	// Point in triangle:
	assert!(t.point_in_triangle_2d(&Float3::new(0.5f32, 0.1f32, 0.0f32)));
}

use nalgebra as na;

struct CameraIntrinsics {
	optical_center_x: f32,
	optical_center_y: f32,
	focal_length: f32,
}

struct CameraExtrinsicsBasis {
	right: na::Vector3<f32>,
	up: na::Vector3<f32>,
	forward: na::Vector3<f32>,
}

struct PoseEstimator {
	pub intrinsics: CameraIntrinsics
}

impl PoseEstimator {
	pub fn estimate_pose(&self, marker_size: f32) {
		
	}
}


use nalgebra as na;

#[derive(Clone, Debug)]
pub struct CameraExtrinsics {
	pub basis: na::Rotation3<f32>,
	pub origin: na::Vector3<f32>,
}

#[derive(Clone, Debug)]
pub struct CameraIntrinsics {
	pub image_width: u32,
	pub image_height: u32,
	pub focal_x: f32,
	pub focal_y: f32,
	pub principal_x: f32,
	pub principal_y: f32,
}

pub struct CameraModel {
	pub intrinsics: CameraIntrinsics,
	pub extrinsics: CameraExtrinsics,
}

impl CameraIntrinsics {
	pub fn new(image_width: u32, image_height: u32, focal_x: f32, focal_y: f32, principal_x: Option<f32>, principal_y: Option<f32>) -> Self {
		Self {
			image_width,
			image_height,
			focal_x,
			focal_y,
			principal_x: principal_x.unwrap_or(image_width as f32/2.0f32),
			principal_y: principal_y.unwrap_or(image_height as f32/2.0f32),
		}
	}

	pub fn new_from_fov_horizontal(horizontal_fov_radians: f32, sensor_width_mm: f32, resolution_x: u32, resolution_y: u32) -> Self {
		// Horizontal FOV=Vertical FOV×Aspect Ratio
		// vfov = hfov / aspect_ratio
		let aspect_ratio = resolution_x as f32/resolution_y as f32;
		let vertical_fov_radians = horizontal_fov_radians / aspect_ratio;
		let sensor_height_mm = sensor_width_mm / aspect_ratio;

		// Focal length = sensor width / (2*tan(FOV/2))
		// Field angle of view = 2 x arctan ((sensor dimension (mm) / 2) / focal length (mm))
		// hfov = 2*arctan((sensor_width/2) / focal_len_mm)
		// hfov / 2 = arctan((sensor_width/2) / focal_len_mm)
		// tan(hfov/2) = (sensor_width/2) / focal_len_mm
		// tan(hfov/2) / (sensor_width/2) = 1/focal_len_mm
		let horizontal_focal_length_mm = (sensor_width_mm*0.5f32) / (horizontal_fov_radians * 0.5f32).tan();
		let vertical_focal_length_mm = (sensor_height_mm*0.5f32) / (vertical_fov_radians * 0.5f32).tan();
		Self {
			image_width: resolution_x,
			image_height: resolution_y,
			focal_x: horizontal_focal_length_mm,
			focal_y: vertical_focal_length_mm,
			principal_x: resolution_x as f32 * 0.5f32,
			principal_y: resolution_y as f32 * 0.5f32,
		}
	}

	/// Transform an object from camera coordinates (camera space) to image/homogeneous coordinates (view space).
	/// In OpenGL this would be the projection matrix, which takes points in the frustum (-1.0 to +1.0) and maps them to (0 - image size).
	/// A point at (0, 0, z) in the camera space would be at the image center after this.
	pub fn project(&self, x: f32, y:f32, z:f32) -> (f32, f32, f32) {
		(
			(x * self.focal_x) + (z * self.principal_x),
			(y * self.focal_y) + (z * self.principal_y),
			z,
		)
	}

	/// Transforms an object from camera coordinates/camera space/frustum to image/homogeneous coordinates (view space).
	/// The input is a point from -1 to 1.0 on each axis (or, really, outside of that if you're wonky) and the output is from 0-image size.
	/// Unlike project, this will clip the point if it's outside the normal viewing space (i.e., z <= 0.0).
	pub fn project_culled(&self, x: f32, y:f32, z:f32) -> Option<(f32, f32)> {
		if z <= 0.0f32 { // x < -1.0f32 || y < -1.0f32 || z <= 0.0f32 || x > 1.0f32 || y > 1.0f32 {
			return None;
		}
		Some((
			(x * self.focal_x)/z + self.principal_x,
			(y * self.focal_y)/z + self.principal_y,
		))
	}

	/// From image plane coordinates to camera space coordinates.
	/// Assumes z=1.
	pub fn unproject(&self, x: f32, y:f32) -> (f32, f32) {
		(
			(x - self.principal_x) / self.focal_x,
			(y - self.principal_y) / self.focal_y,
		)
	}
}

/// This is the K matrix in the projection equation [u,v,1].T = K * [R | t] * [X, Y, Z, 1].T
impl From<CameraIntrinsics> for na::Matrix3<f32> {
	fn from(intrinsics: CameraIntrinsics) -> Self {
		na::Matrix3::new(
			intrinsics.focal_x, 0.0f32, intrinsics.principal_x,
			0f32, intrinsics.focal_y, intrinsics.principal_y,
			0f32, 0f32, 1f32,
		)
	}
}

impl From<CameraIntrinsics> for na::Matrix3x4<f32> {
	fn from(intrinsics: CameraIntrinsics) -> Self {
		na::Matrix3x4::new(
			intrinsics.focal_x, 0.0f32, intrinsics.principal_x, 0f32,
			0f32, intrinsics.focal_y, intrinsics.principal_y, 0f32,
			0f32, 0f32, 1f32, 0f32,
		)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use image::RgbImage;

	#[test]
	fn test_project_point_intrinsic() {
		let intr = CameraIntrinsics::new(640, 480, 1.0f32, 1.0f32, None, None);

	}

	#[test]
	fn test_unproject_intrinsics() {
		let intr = CameraIntrinsics::new(640, 480, 1.0f32, 1.0f32, None, None);

	}
}


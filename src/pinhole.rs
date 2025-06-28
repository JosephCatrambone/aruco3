
use nalgebra as na;

#[derive(Clone, Debug)]
pub struct CameraExtrinsics {
	pub left: na::Vector3<f32>,
	pub up: na::Vector3<f32>,
	pub forward: na::Vector3<f32>,
	pub position: na::Vector3<f32>,
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

impl CameraExtrinsics {
	/// Given a coordinate system where -z is forward, +y is up, and +x is right, returns a set of camera extrinsics.
	/// +z is backward.  -y is down.  -x is left.
	pub fn new_from_lookat(eye_position: (f32, f32, f32), looking_at: (f32, f32, f32), world_up: (f32, f32, f32)) -> Self {
		let mut camera_up = na::Vector3::new(world_up.0, world_up.1, world_up.2).normalize();
		let eye = na::Vector3::new(eye_position.0, eye_position.1, eye_position.2);
		let lookat = na::Vector3::new(looking_at.0, looking_at.1, looking_at.2);
		let forward = (lookat - eye).normalize();
		let left = forward.cross(&camera_up);
		camera_up = left.cross(&forward);  // Refine camera up.
		CameraExtrinsics {
			left,
			up: camera_up,
			forward,
			position: eye,
		}
	}
	
	pub fn new_from_matrices(rotation: na::Matrix3<f32>, translation: na::Vector3<f32>) -> Self {
		CameraExtrinsics {
			left: na::Vector3::new(rotation.m11, rotation.m12, rotation.m13),
			up: na::Vector3::new(rotation.m21, rotation.m22, rotation.m23),
			forward: na::Vector3::new(rotation.m31, rotation.m32, rotation.m33),
			position: translation,
		}
	}

	pub fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
		// rx, ux, dx, tx | x
		// ry, uy, dy, ty | y
		// rz, uz, dz, tz | z
		let xp = (self.left.x * x + self.up.x * y + self.forward.x*z) + self.position.x;
		let yp = (self.left.y * x + self.up.y * y + self.forward.y*z) + self.position.y;
		let zp = (self.left.z * x + self.up.z * y + self.forward.z*z) + self.position.z;
		(xp, yp, zp)
	}
}

impl Default for CameraExtrinsics {
	fn default() -> Self {
		CameraExtrinsics {
			left: na::Vector3::new(1.0f32, 0.0f32, 0.0f32),
			up: na::Vector3::new(0.0f32, 1.0f32, 0.0f32),
			forward: na::Vector3::new(0.0f32, 0.0f32, 1.0f32),
			position: na::Vector3::new(0f32, 0f32, 0f32)
		}
	}
}

impl From<CameraExtrinsics> for na::Matrix4<f32> {
	fn from(extrinsics: CameraExtrinsics) -> Self {
		na::Matrix4::new(
			extrinsics.left.x, extrinsics.left.y, extrinsics.left.z, extrinsics.position.x,
			extrinsics.up.x, extrinsics.up.y, extrinsics.up.z, extrinsics.position.y,
			extrinsics.forward.x, extrinsics.forward.y, extrinsics.forward.z, extrinsics.position.z,
			0f32, 0f32, 0f32, 1f32,
		)
	}
}

impl CameraModel {
	/// Given a point in 3D space, project to a fractional screen coordinate IF IN FRONT.
	pub fn project_point(&self, x: f32, y: f32, z: f32) -> Option<(f32, f32)> {
		let (x2, y2, z2) = self.extrinsics.project(x, y, z);
		self.intrinsics.project_culled(x2, y2, z2)
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
	fn test_project_point_extrinsic() {
		let intr = CameraIntrinsics::new(640, 480, 1.0f32, 1.0f32, None, None);

	}

	#[test]
	fn test_unproject_intrinsics() {
		let intr = CameraIntrinsics::new(640, 480, 1.0f32, 1.0f32, None, None);

	}

	#[test]
	fn test_project() {
		let intr = CameraIntrinsics::new(640, 480, 1.0f32, 1.0f32, None, None);
		let mut extr = CameraExtrinsics::new_from_lookat((0f32, 0f32, -10f32), (0f32, 0f32, 0f32), (0f32, 1f32, 0f32));
		let mut c = CameraModel { extrinsics: extr, intrinsics: intr };
		let image_point = c.project_point(0f32, 0f32, 0f32); // We expect if the camera is moved back by 10 units it should still be centered.
		assert!(image_point.is_some());
		assert_eq!(image_point, Some((320f32, 240f32))); // Center of the focal plane.
	}
}


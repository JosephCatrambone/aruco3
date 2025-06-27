
use nalgebra as na;
use nalgebra::Matrix4;

pub struct CameraExtrinsics {
	pub left: na::Vector3<f32>,
	pub up: na::Vector3<f32>,
	pub forward: na::Vector3<f32>,
	pub position: na::Vector3<f32>,
}

pub struct CameraIntrinsics {
	pub image_width: u32,
	pub image_height: u32,
	pub focal_x: f32,
	pub focal_y: f32,
	pub principal_x: f32,
	pub principal_y: f32,
}

pub struct CameraModel {
	intrinsics: CameraIntrinsics,
	extrinsics: CameraExtrinsics,
}

impl CameraIntrinsics {
	/*
	@classmethod
    def from_fov(cls, fov: float, resolution_x: int, resolution_y: int):
        aspect_ratio = resolution_x/resolution_y
        return cls(focal_x=aspect_ratio*math.tan(fov))

    def set_sensor_width(self, width_mm: float):
        self.focal_x = width_mm / self.image_width

    def set_sensor_height(self, height_mm: float):
        self.focal_y = height_mm / self.image_height

    def to_matrix(self):
        return numpy.array([[self.focal_x, 0, self.image_width/2.0], [0.0, self.focal_y, self.image_height/2.0], [0.0, 0.0, 1.0]])
	*/
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

	pub fn new_from_fov(horizontal_fov_radians: f32, sensor_width_mm: f32, resolution_x: u32, resolution_y: u32) -> Self {
		// Horizontal FOV=Vertical FOV×Aspect Ratio
		// vfov = hfov / aspect_ratio
		let aspect_ratio = resolution_x as f32/resolution_y as f32;

		//Field angle of view = 2 x arctan ((sensor dimension (mm) / 2) / focal length (mm))
		// hfov = 2*arctan((sensor_width/2) / focal_len_mm)
		// hfov / 2 = arctan((sensor_width/2) / focal_len_mm)
		// tan(hfov/2) = (sensor_width/2) / focal_len_mm
		// tan(hfov/2) / (sensor_width/2) = 1/focal_len_mm
		let horizontal_focal_length_mm = (sensor_width_mm*0.5f32) / (horizontal_fov_radians * 0.5f32).tan();
		let vertical_focal_length_mm = ((sensor_width_mm / aspect_ratio)*0.5f32) / (horizontal_fov_radians * 0.5f32).tan();
		Self {
			image_width: resolution_x,
			image_height: resolution_y,
			focal_x: horizontal_focal_length_mm,
			focal_y: vertical_focal_length_mm,
			principal_x: resolution_x as f32 * 0.5f32,
			principal_y: resolution_y as f32 * 0.5f32,
		}
	}

	/// If we have world coordinates, image plane coordinates, and camera coordinates, this goes from camera_coordinates (0-1) to image plane coordinates.
	pub fn project(&self, x: f32, y:f32, z:f32) -> (f32, f32, f32) {
		(
			(x * self.focal_x) + (z * self.principal_x),
			(y * self.focal_y) + (z * self.principal_y),
			z,
		)
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
		/*
        # Eye is the camera position.  Lookat is the point upon which the camera is focusing.  Up is the global world-up.
        # -z is forward.  +y is up.  +x is right.
        # +z is backward.  -y is down.  -x is left.
        rev_camera_direction = numpy.array(eye) - numpy.array(lookat)
        norm_rev_camera_direction = normalize(rev_camera_direction)
        camera_right = normalize(numpy.cross(up, norm_rev_camera_direction))  # Represents +x in camera space.
        camera_up = numpy.cross(norm_rev_camera_direction, camera_right)
        return cls(left=-camera_right, up=camera_up, forward=norm_rev_camera_direction, camera_position=rev_camera_direction)
        """
        eye = numpy.array(eye)
        forward = normalize(numpy.array(lookat) - eye)
        left = numpy.cross(forward, normalize(up))
        camera_up = numpy.cross(left, forward)
        return cls(left=left, up=camera_up, forward=forward, camera_position=eye)
		*/
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

	pub fn project(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
		// rx, ux, dx, tx | x
		// ry, uy, dy, ty | y
		// rz, uz, dz, tz | z
		// P(3x4) = K(3x3) [ R -Rx_0 ]
		let xp = (self.left.x * x + self.up.x * y - self.forward.x*z) - self.position.x;
		let yp = (self.left.y * x + self.up.y * y - self.forward.y*z) - self.position.y;
		let zp = (self.left.z * x + self.up.z * y - self.forward.z*z) - self.position.z;
		(xp, yp, zp)
	}

	/*
def to_matrix(self):
"""
right = -self.left
direction = self.forward
basis_partial = numpy.eye(4)
basis_partial[0,0:3] = right
basis_partial[1,0:3] = self.up
basis_partial[2,0:3] = direction
translation_partial = numpy.eye(4)
translation_partial[0:3,-1] = -self.camera_position
return basis_partial @ translation_partial
"""
partial = numpy.eye(4)
partial[0, 0:3] = self.left
partial[1, 0:3] = self.up
partial[2, 0:3] = -self.forward
partial[0:3, -1] = -self.camera_position
return partial
*/
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
		/*
		class CameraModel:
		intrinsics: CameraIntrinsics
		extrinsics: CameraExtrinsics

		def transform_points(self, points):
			# Given a matrix of nx3, transform the points in accordance with the intrinsics and extrinsics.
			assert points.shape[1] == 3
			points = numpy.hstack((points, numpy.ones((points.shape[0], 1), dtype=points.dtype)))
			points = points @ self.extrinsics.to_matrix().T  # nx4
			points = points[:, 0:3] / points[:, 3:]  # nx3
			points = points @ self.intrinsics.to_matrix().T # nx3
			points = points[:, 0:2] / points[:, 2:] # nx2
			return points
		*/
		let (x2, y2, z2) = self.extrinsics.project(x, y, z);
		let (x3, y3, z3) = self.intrinsics.project(x2, y2, z2);
		Some((x3/z3, y3/z3))
		// sensor sys = image plane to sensor mat * camera to image mat * obj to camera mat * obj system homo
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


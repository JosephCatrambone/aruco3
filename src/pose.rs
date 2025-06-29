use std::ops::{Add, Sub, Mul, Div};
use nalgebra as na;
use crate::CameraIntrinsics;

/// MarkerPose contains a translation and rotation that moves a fiducial directly in front of the camera away and into some place+orientation in the scene.
/// It uses OpenCV's chirality: right-handed with +Z forward, +Y Down, and +X right.
#[derive(Clone, Debug)]
pub struct MarkerPose {
	pub error: f32,
	pub rotation: na::Matrix3<f32>,
	pub translation: na::Vector3<f32>,
}

impl MarkerPose {
	/// Translate an array of points such that if they were immediately in front of the camera, they would now be somewhere in the scene.
	/// Uses a right-hand (+Z forward, +Y down, +X right) coordinate system.
	pub fn apply_transform_to_points(&self, points: &Vec<(f32, f32, f32)>) -> Vec<(f32, f32, f32)> {
		let as_vec3 = points.iter().map(|p| { na::Vector3::new(p.0, p.1, p.2) }).collect();
		self.apply_transform_to_vectors(&as_vec3).into_iter().map(|p| {(p.x, p.y, p.z)}).collect()
	}

	/// Translate an array of vectors such that if they were immediately in front of the camera, they would now be somewhere in the scene.
	/// Uses a right-hand (+Z forward, +Y down, +X right) coordinate system.
	pub fn apply_transform_to_vectors(&self, points: &Vec<na::Vector3<f32>>) -> Vec<na::Vector3<f32>> {
		points.iter().map(|p| {
			(self.rotation * p) + self.translation
		}).collect()
	}

	pub fn apply_inverse_transform_to_points(&self, points: &Vec<(f32, f32, f32)>) -> Vec<(f32, f32, f32)> {
		let as_vec3 = points.iter().map(|p| { na::Vector3::new(p.0, p.1, p.2) }).collect();
		self.apply_inverse_transform_to_vectors(&as_vec3).into_iter().map(|p| {(p.x, p.y, p.z)}).collect()
	}

	pub fn apply_inverse_transform_to_vectors(&self, points: &Vec<na::Vector3<f32>>) -> Vec<na::Vector3<f32>> {
		points.iter().map(|p| {
			self.rotation.transpose() * (p - self.translation)
		}).collect()
	}
}

impl Default for MarkerPose {
	fn default() -> Self {
		Self {
			error: 1e31,
			rotation: na::Matrix3::new(1.0, 0.0, 0.0,  0.0, 1.0, 0.0,  0.0, 0.0, 1.0),
			translation: na::Vector3::new(0.0, 0.0, 0.0),
		}
	}
}

pub fn solve_with_intrinsics(image_points: &Vec<(u32, u32)>, marker_size_mm: f32, camera_intrinsics: &CameraIntrinsics) -> (MarkerPose, MarkerPose) {
	let pts = image_points.iter().map(|&(x, y)| { camera_intrinsics.unproject(x as f32, y as f32) }).collect();
	solve_with_normalized_points(&pts, marker_size_mm)
}

// As a matter of usability, should we pack the intrinsics/image size/marker size into an object?
/// This method assumes that points have been undistorted (or that you don't care) and are on the image plane from 0-width,0-height.
pub fn solve_with_undistorted_points(image_points: &Vec<(u32, u32)>, marker_size_mm: f32, image_size: (u32, u32)) -> (MarkerPose, MarkerPose) {
	let pts = image_points.iter().map(|&(x, y)| { ((x as f32) / image_size.0 as f32, (y as f32) / image_size.1 as f32) }).collect();
	solve_with_normalized_points(&pts, marker_size_mm)
}

pub fn solve_with_normalized_points(normalized_image_points: &Vec<(f32, f32)>, marker_size_mm: f32) -> (MarkerPose, MarkerPose) {
	let object_points_2d = make_marker_square(marker_size_mm);
	let normalized_points: Vec<na::Vector2<f32>> = normalized_image_points.iter().map(|&(x, y)| { na::Vector2::new(x, y) }).collect();

	// Compute the homography from the marker square to the image points.
	// Homography Matrix: [[x1], [y1], [1]] = H * [[x2], [y2], [1]]
	// Both our world coordinates and our undistorted coordinates have zeros for their last row
	// canonical_object_points_to_normalized_points_homography == homography
	let homography = compute_homography_from_marker_square(marker_size_mm, &normalized_points);

	let (pose1, pose2) = solve_canonical_form(&object_points_2d, &normalized_points, &homography);

	if pose1.error < pose2.error {
		(pose1, pose2)
	} else {
		(pose2, pose1)
	}
}

/// Generate four points, clockwise, starting from the top left, around the center with z=0.
/// 'Up' is +Y, 'Right' is +X.
fn make_marker_square(marker_size_mm: f32) -> Vec<na::Vector3<f32>> {
	let hw = 0.5f32 * marker_size_mm;
	vec![
		na::Vector3::new(-hw, hw, 0.0),
		na::Vector3::new(hw, hw, 0.0),
		na::Vector3::new(hw, -hw, 0.0),
		na::Vector3::new(-hw, -hw, 0.0),
	]
}

/// Given an image size, computes the homography from an idealized 1x1 cube to the image points.
fn compute_homography_from_marker_square(marker_size_mm: f32, target_points: &Vec<na::Vector2<f32>>) -> na::Matrix3<f32> {
	// In the C++ implementation these are all inverted.
	let p1x = -target_points[0].x;
	let p1y = -target_points[0].y;
	let p2x = -target_points[1].x;
	let p2y = -target_points[1].y;
	let p3x = -target_points[2].x;
	let p3y = -target_points[2].y;
	let p4x = -target_points[3].x;
	let p4y = -target_points[3].y;

	let half_width = marker_size_mm / 2.0;
	let det_inv: f32 = -1.0f32 / (half_width * (p1x * p2y - p2x * p1y - p1x * p4y + p2x * p3y - p3x * p2y + p4x * p1y + p3x * p4y - p4x * p3y));

	let homography = na::Matrix3::new(
		det_inv * (p1x * p3x * p2y - p2x * p3x * p1y - p1x * p4x * p2y + p2x * p4x * p1y - p1x * p3x * p4y + p1x * p4x * p3y + p2x * p3x * p4y - p2x * p4x * p3y), // 0,0
		det_inv * (p1x * p2x * p3y - p1x * p3x * p2y - p1x * p2x * p4y + p2x * p4x * p1y + p1x * p3x * p4y - p3x * p4x * p1y - p2x * p4x * p3y + p3x * p4x * p2y), // 0,1
		det_inv * half_width * (p1x * p2x * p3y - p2x * p3x * p1y - p1x * p2x * p4y + p1x * p4x * p2y - p1x * p4x * p3y + p3x * p4x * p1y + p2x * p3x * p4y - p3x * p4x * p2y), // 0,2
		det_inv * (p1x * p2y * p3y - p2x * p1y * p3y - p1x * p2y * p4y + p2x * p1y * p4y - p3x * p1y * p4y + p4x * p1y * p3y + p3x * p2y * p4y - p4x * p2y * p3y), // 1,0
		det_inv * (p2x * p1y * p3y - p3x * p1y * p2y - p1x * p2y * p4y + p4x * p1y * p2y + p1x * p3y * p4y - p4x * p1y * p3y - p2x * p3y * p4y + p3x * p2y * p4y), // 1,1
		det_inv * half_width * (p1x * p2y * p3y - p3x * p1y * p2y - p2x * p1y * p4y + p4x * p1y * p2y - p1x * p3y * p4y + p3x * p1y * p4y + p2x * p3y * p4y - p4x * p2y * p3y), // 1,2
		-det_inv * (p1x * p3y - p3x * p1y - p1x * p4y - p2x * p3y + p3x * p2y + p4x * p1y + p2x * p4y - p4x * p2y), // 2,0
		det_inv * (p1x * p2y - p2x * p1y - p1x * p3y + p3x * p1y + p2x * p4y - p4x * p2y - p3x * p4y + p4x * p3y), // 2,1
		1.0f32,
	);

	homography
}

/// Compute the jacobian of the homography (which maps an idealized real marker to image points),
/// then use that jacobian to refine said homography into real pose data.
/// Input: the idealized corners of a marker at z=0, clockwise from top-left.
/// Output: two possible poses.
/// 'canonical points' are just points centered at zero with z=0. If we generate the marker points this satisfies the constraint.
fn solve_canonical_form(object_points_2d: &Vec<na::Vector3<f32>>, normalized_image_points: &Vec<na::Vector2<f32>>, h: &na::Matrix3<f32>) -> (MarkerPose, MarkerPose) {
	let jacobian = na::Matrix2::<f32>::new(
		h.m11 - h.m31 * h.m13, h.m12 - h.m32 * h.m13,
		h.m21 - h.m31 * h.m23, h.m22 - h.m32 * h.m23,
	);

	// The homography's third column has a transform of 0,0 into the image plane.
	let translation = na::Vector2::new(h.m13, h.m23);

	let mut pose1 = MarkerPose::default();
	let mut pose2 = MarkerPose::default();

	let (rot1, rot2) = compute_rotations(&jacobian, translation.x, translation.y);

	let tx1 = compute_translation(object_points_2d, normalized_image_points, &rot1);
	let tx2 = compute_translation(object_points_2d, normalized_image_points, &rot2);

	pose1.rotation = rot1;
	pose1.translation = tx1;
	pose2.rotation = rot2;
	pose2.translation = tx2;

	pose1.error = compute_reprojection_error(&pose1, object_points_2d, normalized_image_points);
	pose2.error = compute_reprojection_error(&pose2, object_points_2d, normalized_image_points);

	(pose1, pose2)
}

fn compute_rotations(jacobian: &na::Matrix2<f32>, tx: f32, ty: f32) -> (na::Matrix3<f32>, na::Matrix3<f32>) {
	// Begin weeping openly.
	let translation = na::Vector3::new(
		tx,
		ty,
		1.0
	);

	let rv = find_rotation_to_z(&translation).transpose();

	// Closed-form 3x3 SVD?:
	let b00 = rv.m11 - tx * rv.m31;
	let b01 = rv.m12 - tx * rv.m32;
	let b10 = rv.m21 - ty * rv.m31;
	let b11 = rv.m22 - ty * rv.m32;

	let inv_det = 1.0 / (b00 * b11 - b01 * b10); // dtinv = 1.0 / ((b00 * b11 - b01 * b10));
	let binv00 = inv_det * b11;
	let binv01 = -inv_det * b01;
	let binv10 = -inv_det * b10;
	let binv11 = inv_det * b00;

	let a00 = binv00 * jacobian.m11 + binv01 * jacobian.m21;
	let a01 = binv00 * jacobian.m12 + binv01 * jacobian.m22;
	let a10 = binv10 * jacobian.m11 + binv11 * jacobian.m21;
	let a11 = binv10 * jacobian.m12 + binv11 * jacobian.m22;

	// Largest singular value of A:
	let ata00 = a00 * a00 + a01 * a01;
	let ata01 = a00 * a10 + a01 * a11;
	let ata11 = a10 * a10 + a11 * a11;

	let gamma = (0.5 * (ata00 + ata11 + ((ata00 - ata11) * (ata00 - ata11) + 4.0 * ata01 * ata01).sqrt())).sqrt();

	// Reconstruct rot matrices:
	let rtilde00 = a00 / gamma;
	let rtilde01 = a01 / gamma;
	let rtilde10 = a10 / gamma;
	let rtilde11 = a11 / gamma;

	let rtilde00_2 = rtilde00 * rtilde00;
	let rtilde01_2 = rtilde01 * rtilde01;
	let rtilde10_2 = rtilde10 * rtilde10;
	let rtilde11_2 = rtilde11 * rtilde11;

	let b0 = (-rtilde00_2 - rtilde10_2 + 1.0).sqrt();
	let mut b1 = (-rtilde01_2 - rtilde11_2 + 1.0).sqrt();
	let sp = -rtilde00 * rtilde01 - rtilde10 * rtilde11;

	if sp < 0.0 {
		b1 = -b1;
	}

	let r1 = na::Matrix3::new(
		(rtilde00)*rv.m11 + (rtilde10)*rv.m12 + (b0)*rv.m13,
		(rtilde01)*rv.m11 + (rtilde11)*rv.m12 + (b1)*rv.m13,
		(b1 * rtilde10 - b0 * rtilde11) * rv.m11 + (b0 * rtilde01 - b1 * rtilde00) * rv.m12 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv.m13,
		(rtilde00)*rv.m21 + (rtilde10)*rv.m22 + (b0)*rv.m23,
		(rtilde01)*rv.m21 + (rtilde11)*rv.m22 + (b1)*rv.m23,
		(b1 * rtilde10 - b0 * rtilde11) * rv.m21 + (b0 * rtilde01 - b1 * rtilde00) * rv.m22 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv.m23,
		(rtilde00)*rv.m31 + (rtilde10)*rv.m32 + (b0)*rv.m33,
		(rtilde01)*rv.m31 + (rtilde11)*rv.m32 + (b1)*rv.m33,
		(b1 * rtilde10 - b0 * rtilde11) * rv.m31 + (b0 * rtilde01 - b1 * rtilde00) * rv.m32 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv.m33,
	);
	let r2 = na::Matrix3::new(
		(rtilde00)*rv.m11 + (rtilde10)*rv.m12 + (-b0) * rv.m13,
		(rtilde01)*rv.m11 + (rtilde11)*rv.m12 + (-b1) * rv.m13,
		(b0 * rtilde11 - b1 * rtilde10) * rv.m11 + (b1 * rtilde00 - b0 * rtilde01) * rv.m12 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv.m13,
		(rtilde00)*rv.m21 + (rtilde10)*rv.m22 + (-b0) * rv.m23,
		(rtilde01)*rv.m21 + (rtilde11)*rv.m22 + (-b1) * rv.m23,
		(b0 * rtilde11 - b1 * rtilde10) * rv.m21 + (b1 * rtilde00 - b0 * rtilde01) * rv.m22 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv.m23,
		(rtilde00)*rv.m31 + (rtilde10)*rv.m32 + (-b0) * rv.m33,
		(rtilde01)*rv.m31 + (rtilde11)*rv.m32 + (-b1) * rv.m33,
		(b0 * rtilde11 - b1 * rtilde10) * rv.m31 + (b1 * rtilde00 - b0 * rtilde01) * rv.m32 + (rtilde00 * rtilde11 - rtilde01 * rtilde10) * rv.m33,
	);

	(r1, r2)
}

/// Find the rotation that, when applied to vec, will align it with <0, 0, 1>.
fn find_rotation_to_z(vec: &na::Vector3<f32>) -> na::Matrix3<f32> {
	let mut rot = na::Matrix3::<f32>::zeros();
	let vec = vec.normalize();
	let ax = vec.x;
	let ay = vec.y;
	let az = vec.z;

	if (1.0+az).abs() < 1e-6 {
		// We have a potential numerical stability issue.
		rot.m11 = 1.0;
		rot.m22 = 1.0;
		rot.m33 = -1.0;
	} else {
		let d = 1.0 / (1.0+az);
		let ax2 = ax*ax;
		let ay2 = ay*ay;
		let axay = ax*ay;
		rot.m11 = -ax2*d + 1.0;
		rot.m12 = -axay*d;
		rot.m13 = -ax;
		rot.m21 = -axay*d;
		rot.m22 = -ay2*d + 1.0;
		rot.m23 = -ay;
		rot.m31 = ax;
		rot.m32 = ay;
		rot.m33 = 1.0 - (ax2+ay2)*d;
	}

	rot
}

fn compute_translation(object_points_2d: &Vec<na::Vector3<f32>>, normalized_image_points: &Vec<na::Vector2<f32>>, rot: &na::Matrix3<f32>) -> na::Vector3<f32> {
	// This is solved by building the linear system At = b, where t corresponds to the (unknown) translation.
	// This is then inverted with the associated normal equations to give t = inv(transpose(A)*A)*transpose(A)*b
	// For efficiency we only store the coefficients of (transpose(A)*A) and (transpose(A)*b)

	// Coefficients of A.T * A 
	let mut ata = na::Matrix3::new(
		4f32, 0.0, 0.0,
		0f32, 4f32, 0f32,
		0f32, 0f32, 0f32
	);

	let mut atb0 = 0f32;
	let mut atb1 = 0f32;
	let mut atb2 = 0f32;

	//now loop through each point and increment the coefficients:
	for i in 0..4 {
		let rx = rot.m11 * object_points_2d[i].x + rot.m12 * object_points_2d[i].y;
		let ry = rot.m21 * object_points_2d[i].x + rot.m22 * object_points_2d[i].y;
		let rz = rot.m31 * object_points_2d[i].x + rot.m32 * object_points_2d[i].y;

		let a2 = -normalized_image_points[i].x;
		let b2 = -normalized_image_points[i].y;

		ata.m13 += a2;
		ata.m23 += b2;
		ata.m31 += a2;
		ata.m32 += b2;
		ata.m33 += a2*a2 + b2*b2;

		let bx = -a2 * rz - rx;
		let by = -b2 * rz - ry;

		atb0 += bx;
		atb1 += by;
		atb2 += a2 * bx + b2 * by;
	}

	// println!("atb0, 1, 2: {} {} {}", &atb0, &atb1, &atb2);

	let det_a_inv = 1.0 / (ata.m11 * ata.m22 * ata.m33 - ata.m11 * ata.m23 * ata.m32 - ata.m13 * ata.m22 * ata.m31);
	//let det_a_inv = 1.0 / ata.determinant();

	// (A.T * A).inv() / A.det().pow(2)
	// Do we want to do this with matmuls or long form?
	let s = na::Matrix3::new(
		ata.m22 * ata.m33 - ata.m23 * ata.m32,
		ata.m13 * ata.m32,
		-ata.m13 * ata.m22,
		ata.m23 * ata.m31,
		ata.m11 * ata.m33 - ata.m13 * ata.m31,
		-ata.m11 * ata.m23,
		-ata.m22 * ata.m31,
		-ata.m11 * ata.m32,
		ata.m11 * ata.m22,
	);

	//println!("S00, 01, ...:\n{}, {}, {}\n{}, {}, {}\n{}, {}, {}", s.m11, s.m12, s.m13, s.m21, s.m22, s.m23, s.m31, s.m32, s.m33);

	// Solve t:
	na::Vector3::new(
		det_a_inv * (s.m11 * atb0 + s.m12 * atb1 + s.m13 * atb2),
		det_a_inv * (s.m21 * atb0 + s.m22 * atb1 + s.m23 * atb2),
		det_a_inv * (s.m31 * atb0 + s.m32 * atb1 + s.m33 * atb2),
	)
}

fn compute_reprojection_error(pose: &MarkerPose, object_points_2d: &Vec<na::Vector3<f32>>, normalized_image_points: &Vec<na::Vector2<f32>>) -> f32 {
	let projected = pose.apply_transform_to_vectors(object_points_2d);
	let mut error = 0f32;

	for i in 0..normalized_image_points.len() {
		let dx = (projected[i].x / projected[i].z.max(1e-5)) - normalized_image_points[i].x;
		let dy = (projected[i].y / projected[i].z.max(1e-5)) - normalized_image_points[i].y;
		error += (dx*dx + dy*dy).sqrt();
	}

	error
}

#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra as na;
	use rand::{rng, Rng};

	fn assert_approx_equal<'a, R, C, S>(
		a: &'a na::Matrix<f32, R, C, S>,
		b: &'a na::Matrix<f32, R, C, S>,
		epsilon: f32,
	) -> bool where
		R: na::Dim,
		C: na::Dim,
		S: na::RawStorage<f32, R, C>,
	{
		let mut success = true;
		let (rows, cols) = a.shape();
		for r in 0..rows {
			for c in 0..cols {
				if (a[(r, c)] - b[(r, c)]).abs() > epsilon {
					success = false;
					eprintln!("|a-b| > epsilon at {},{}: {} {}", &r, &c, &a[(r,c)], &b[(r,c)]);
				}
			}
		}
		assert!(success);
		success
	}

	#[test]
	fn test_marker_transforms() {
		let test_points = vec![(0f32, 0f32, 0f32), (7f32, 11f32, 13f32)];
		// Our pose translates by 1,2,3 and rotates 90 degrees around the +Y axis, swapping X and Z.
		let mut pose = MarkerPose::default();
		pose.translation.x = 1.0f32;
		pose.translation.y = 2.0f32;
		pose.translation.z = 3.0f32;
		pose.rotation.m11 = 0.0;
		pose.rotation.m13 = 1.0;
		pose.rotation.m31 = 1.0;
		pose.rotation.m33 = 0.0;
		assert_eq!(pose.apply_transform_to_points(&test_points), vec![(1f32, 2f32, 3f32), (14f32, 13f32, 10f32)]);
	}

	#[test]
	fn test_marker_identity_random() {
		// Randomly generate 100 different marker configurations, then verify that they're invertable.
		// They should be, given we create proper rotations and translations.
		let mut rng = rng();
		let mut failures = 0;
		let mut tests = 0;
		let mut total_error = 0f64;
		let mut max_error = 0f32;

		for _ in 0..100 {
			let mut pose = MarkerPose::default();

			// Create a random translation:
			pose.translation.x = rng.random();
			pose.translation.y = rng.random();
			pose.translation.z = rng.random();
			// Pick something on the XY plane and something on the YZ plane that we know are non-orthogonal, normalize, and take the crossproduct to make random rotation basis.
			let mut row1 = na::Vector3::new(1.0 + rng.random::<f32>(), 1.0 + rng.random::<f32>(), 0.0f32).normalize(); // We offset by 1.0 to avoid a degenerate point at the origin.
			let mut row2 = na::Vector3::new(0.0, 1.1 + rng.random::<f32>(), 1.0 + rng.random::<f32>()).normalize(); // We offset by 1.0 to avoid a degenerate point at the origin.
			let mut row3 = row1.cross(&row2).normalize();
			// Ensure that the rotation is valid by verifying all bases are orthogonal.
			for _ in 0..10 {
				row2 = row1.cross(&row3);
				row1 = row3.cross(&row2);
			}
			pose.rotation.set_column(0, &row1);
			pose.rotation.set_column(1, &row2);
			pose.rotation.set_column(2, &row3);

			for _ in 0..100 {
				tests += 1;
				let points = vec![rng.random::<(f32, f32, f32)>()];
				let transformed = pose.apply_transform_to_points(&points);
				let untransformed = pose.apply_inverse_transform_to_points(&transformed);
				let distance: f32 = points.iter().zip(&untransformed).map(|(a, b)|{ (a.0-b.0).abs() + (a.1-b.1).abs() + (a.2-b.2).abs() }).sum();
				if distance > 1e-5 {
					failures += 1;
				}
				max_error = max_error.max(distance);
				total_error += distance as f64;
			}
		}
		println!("Total Tests: {tests} \n Total Error: {total_error} \n Biggest Error: {max_error} \n Failures: {failures}");
		assert_eq!(failures, 0);
	}

	#[test]
	fn test_gen_marker_square() {
		let square = make_marker_square(11.0f32);
		assert_eq!(square[0].x, -5.5);
		assert_eq!(square[0].y, 5.5);

		assert_eq!(square[1].x, 5.5);
		assert_eq!(square[1].y, 5.5);

		assert_eq!(square[2].x, 5.5);
		assert_eq!(square[2].y, -5.5);

		assert_eq!(square[3].x, -5.5);
		assert_eq!(square[3].y, -5.5);
	}

	#[test]
	fn test_homography_solve() {
		let marker_size = 11.0f32;
		let square = make_marker_square(marker_size);
		let target_points = vec![
			na::Vector2::<f32>::new(0.1, 0.1),
			na::Vector2::<f32>::new(0.3, 0.1),
			na::Vector2::<f32>::new(0.3, 0.3),
			na::Vector2::<f32>::new(0.1, 0.3),
		];
		let expected = na::Matrix3::new(
			0.01818181818181819, 0.0, 0.2,
			9.856383386231859e-19, -0.01818181818181819, 0.2000000000000001,
			1.577021341797097e-17, -1.577021341797097e-17, 1.0
		);
		let computed_homography = compute_homography_from_marker_square(marker_size, &target_points);
		assert!((computed_homography - expected).abs().sum() < 1e-5);
	}

	#[test]
	fn test_canonical_solve() {
		let marker_size = 11.0f32;
		let square = make_marker_square(marker_size);
		let target_points = vec![
			na::Vector2::<f32>::new(0.1, 0.1),
			na::Vector2::<f32>::new(0.3, 0.1),
			na::Vector2::<f32>::new(0.3, 0.3),
			na::Vector2::<f32>::new(0.1, 0.3),
		];
		let homography = compute_homography_from_marker_square(marker_size, &target_points);
		let (pa, pb) = solve_canonical_form(&square, &target_points, &homography);
		let expected_pose_a = na::Matrix4::new(
			1.0, -2.775557561562891e-17, 1.02695629777827e-15, 10.99999999999999,
			7.632783294297951e-17, -1.0, 1.02695629777827e-15, 11.0,
			1.02695629777827e-15, -9.992007221626409e-16, -1.0, 54.99999999999996,
			0.0, 0.0, 0.0, 1.0,
		);
		let expected_pose_b = na::Matrix4::new(
			0.9259259259259256, 0.07407407407407443, -0.3703703703703712, 10.79629629629629,
			-0.0740740740740744, -0.9259259259259256, -0.3703703703703713, 10.79629629629629,
			-0.3703703703703712, 0.3703703703703713, -0.8518518518518512, 54.99999999999999,
			0.0, 0.0, 0.0, 1.0,
		);
		let rot_a = expected_pose_a.view((0, 0), (3, 3));
		let rot_b = expected_pose_b.view((0, 0), (3, 3));
		let translation_a = na::Vector3::new(expected_pose_a[(0, 3)], expected_pose_a[(1, 3)], expected_pose_a[(2, 3)]);
		let translation_b = na::Vector3::new(expected_pose_b[(0, 3)], expected_pose_b[(1, 3)], expected_pose_b[(2, 3)]);

		println!("{}", &pa.translation);
		println!("{}", &pb.translation);

		assert!((pa.rotation - rot_a).abs().sum() < 1e-5);
		assert!((pb.rotation - rot_b).abs().sum() < 1e-5);
		assert!((pa.translation - translation_a).abs().sum() < 1e-4);
		assert!((pb.translation - translation_b).abs().sum() < 1e-4);
	}

	#[test]
	fn test_e2e_pose() {
		let target_points = vec![
			(90, 89),
			(95, 150),
			(80, 170),
			(75, 90),
		];
		let (pa, pb) = solve_with_undistorted_points(&target_points, 17.0f32, (1000, 1000));
		let pa_expected = MarkerPose {
			translation: na::Vector3::new(20.32196265994096f32, 29.69316666108512f32, 238.3658341694123f32),
			rotation: na::Matrix3::new(
				0.07313995850727262, 0.2953796077825095, 0.9525762089070907,
				0.9973210134149258, -0.02055233410014844, -0.07020254813082821,
				-0.001158736630905738, 0.9551588814795613, -0.2960914866390682,
			),
			error: 0.0f32,
		};
		let pb_expected = MarkerPose {
			translation: na::Vector3::new(19.85146615649354f32, 29.20013946746331f32, 234.3277337340188f32),
			rotation: na::Matrix3::new(
				0.05174977302896467, 0.1311239186581316, -0.9900143832021767,
				0.9667844474723887, -0.2550432732960733, 0.01675592050389792,
				-0.2502994069448807, -0.957997623536802, -0.1399669967559523,
			),
			error: 0.0f32,
		};

		let rot_mat_err_a = (pa.rotation - pa_expected.rotation).abs().sum();
		let rot_mat_err_b = (pb.rotation - pb_expected.rotation).abs().sum();

		let translation_err_a = (pa.translation - pa_expected.translation).abs().sum();
		let translation_err_b = (pb.translation - pb_expected.translation).abs().sum();

		assert!(rot_mat_err_a < 2e-5);
		assert!(rot_mat_err_b < 2e-5);
		assert!(translation_err_a < 0.0005f32);
		assert!(translation_err_b < 0.0005f32);
	}

	#[test]
	fn test_e2e_pose2() {
		let marker_size = 19.0f32;
		let points = vec![
			(-0.090f32, -0.089f32),
			(-0.095f32, -0.150f32),
			(-0.080f32, -0.170f32),
			(-0.075f32, -0.090f32),
		];
		let points_f32 = points.iter().map(|&(x,y)| { na::Vector2::new(x, y) }).collect();

		let h = compute_homography_from_marker_square(marker_size, &points_f32);
		let expected_homography = na::Matrix3::new(
			0.0001197249881460392, -0.00193812233285917, -0.08585585585585585,
			-0.003084400189663352, -0.00115457562825984, -0.1225675675675677,
			-0.004504504504504568, 0.01351351351351346, 1.0,
		);

		assert_approx_equal(&h, &expected_homography, 1e-5); //(h - expected_homography).abs().sum() < 1e-5);

		let (pa, pb) = solve_with_normalized_points(&points, marker_size);
		let pa_expected = MarkerPose {
			translation: na::Vector3::new( -22.712781796404,  -33.18648038591866, 266.408873483460),
			rotation: na::Matrix3::new(
				-0.07313995850727262, -0.2953796077825095, -0.9525762089070907,
				-0.9973210134149258, 0.02055233410014844, 0.07020254813082821,
				-0.001158736630905738, 0.9551588814795613, -0.2960914866390682,
			),
			error: 0.0f32,
		};
		let pb_expected = MarkerPose {
			translation: na::Vector3::new( -22.18693276313984, -32.6354499930472,  261.8957024086092),
			rotation: na::Matrix3::new(
			-0.05174977302896467, -0.1311239186581316, 0.9900143832021767,
			-0.9667844474723887, 0.2550432732960733, -0.01675592050389792,
			-0.2502994069448807, -0.957997623536802, -0.1399669967559523,
			),
			error: 0.0f32,
		};

		assert_approx_equal(&pa.rotation, &pa_expected.rotation, 1e-5);
		assert_approx_equal(&pb.rotation, &pb_expected.rotation, 1e-5);
		assert_approx_equal(&pa.translation, &pa_expected.translation, 1e-3);
		assert_approx_equal(&pb.translation, &pb_expected.translation, 1e-3);
	}
}
use std::ops::{Add, Sub, Mul, Div};
use nalgebra as na;

const INITIAL_SVD_EPSILON: f32 = 1e-5;
const INITIAL_SVD_MAX_ITER: usize = 100;
const MAX_POSE_ITERATIONS: usize = 100;

// This is a separate structure because we need to compute the pseudoinverse of the points for a given marker size + focal dist.
// It's faster to update a pose than to recompute it from scratch.
#[derive(Debug)]
pub struct PoseEstimator {
	focal_length: f32,
	marker_size: f32,
	untransformed_marker_points: Vec<na::Vector3<f32>>,
	model_vectors: na::Matrix3<f32>,
	model_normal: na::Vector3<f32>,
	pseudoinverse: na::Matrix3<f32>,
}

#[derive(Clone, Debug)]
pub struct MarkerPose {
	pub error: f32,
	pub rotation: na::Matrix3<f32>,
	pub translation: na::Vector3<f32>,
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

impl PoseEstimator {
	fn new(marker_size: f32, focal_length: f32) -> Self {
		let original_frame = make_marker_squares(marker_size);
		let mut model_vectors: na::Matrix3<f32> = na::Matrix3::new(
			// Why can't we just make this from a list of rows!?  Isn't that the purpose of from_rows!?
			// na::RowVector3::from((original_frame[1] - original_frame[0]).into()),
			original_frame[1].x - original_frame[0].x, original_frame[1].y - original_frame[0].y, original_frame[1].z - original_frame[0].z,
			original_frame[2].x - original_frame[0].x, original_frame[2].y - original_frame[0].y, original_frame[2].z - original_frame[0].z,
			original_frame[3].x - original_frame[0].x, original_frame[3].y - original_frame[0].y, original_frame[3].z - original_frame[0].z,
		);

		let pseudoinverse = model_vectors.pseudo_inverse(INITIAL_SVD_EPSILON).expect("Failed to compute initial pseudoinverse of model vectors. Is the focal distance real? Is the marker size nonzero?");
		assert!(!pseudoinverse.sum().is_nan());
		// Double check this:
		let svd: na::linalg::SVD<f32, na::U3, na::U3> = na::linalg::SVD::try_new(model_vectors, true, true, INITIAL_SVD_EPSILON, INITIAL_SVD_MAX_ITER).expect("Failed to compute initial decomposition for marker and vector. This can happen if marker size or focal length is zero.");
		let v_t = svd.v_t.unwrap();
		/*
		// Ideally since we're computing the SVD once we should just multiply it back out to get the pseudoinverse, but...
		let u = svd.u.unwrap();
		let s_inverse = na::Matrix3::from_fn_generic(na::U3, na::U3, |i, j| {
			if i != j || svd.singular_values[i].abs() < 1e-6 { 0.0 } else { 1.0 / svd.singular_values[i] }
		});
		let pseudoinverse: na::Matrix3<f32> = u * s_inverse * v_t;
		*/

		/*
		let svd: na::linalg::SVD<f32, na::U3, na::U3> = na::linalg::SVD::try_new(model_vectors, true, true, INITIAL_SVD_EPSILON, INITIAL_SVD_MAX_ITER).expect("Failed to compute initial decomposition for marker and vector. This can happen if marker size or focal length is zero.");
		let v_t = svd.v_t.unwrap();
		let pseudoinverse = svd.pseudo_inverse(INITIAL_SVD_EPSILON).expect("Failed to compute pseudoinverse.");
		*/

		// This was v.col(min_index) in the original, so it assumes that singular values are sorted.
		// We're sorting greatest to smallest, so we need to take the _max_ column, which is the last one.
		// Also since it's v_t we're grabbing a row.
		// And since columns won't coerce to Vec3's, we do a transpose afterwards.
		let model_normal: na::Vector3<f32> = v_t.row(2).transpose();

		Self {
			focal_length,
			marker_size,
			untransformed_marker_points: original_frame,
			model_vectors,
			model_normal,
			pseudoinverse,
		}
	}

	pub fn estimate_marker_pose(&self, points: &Vec<(u32, u32)>) -> (MarkerPose, MarkerPose) {
		// I hate keeping this as Vector3 instead of na::Matrix1x3<f32> because the xyz aspect is confusing, but the ergonomics of Matrix1x3 are wrong.
		let mut epsilon_step = na::Vector3::new(1.0f32, 1.0f32, 1.0f32);

		// x_i(1+epsilon_i) is the coordinate x'_i of the point p_i, the scaled orthographic projection of point M_i.
		// M_i is the point somewhere in space.  p_i is the projection of the point

		let mut candidate1 = MarkerPose::default();
		let mut candidate2 = MarkerPose::default();

		self.pose_from_orthography_and_scaling(points, &mut epsilon_step, &mut candidate1, &mut candidate2);
		self.refine_pose(points, &mut candidate1);
		self.refine_pose(points, &mut candidate2);

		if candidate1.error < candidate2.error {
			(candidate1, candidate2)
		} else {
			(candidate2, candidate1)
		}
	}

	fn pose_from_orthography_and_scaling(&self, camera_points: &Vec<(u32, u32)>, epsilon: &mut na::Vector3<f32>, cand1: &mut MarkerPose, cand2: &mut MarkerPose) {
		// "Iterative Pose Estimation Using Coplanar Feature Points" by D OBERKAMPF et al.
		// A lot of these magic variables don't have direct translations into something that will be conceptually helpful in understanding the code.
		// My comments are, at best, a description of my limited understanding of the various transformations, and even with visual aids assigning real names to the variables is like trying to describe 'the color blue'.
		// Given this fact, I'll use variable names that stick to the paper and leave comments describing the algorithm.
		let xi = na::Vector3::new(camera_points[1].0 as f32, camera_points[2].0 as f32,camera_points[3].0 as f32);
		let yi = na::Vector3::new(camera_points[1].1 as f32, camera_points[2].1 as f32,camera_points[3].1 as f32);
		let xs: na::Vector3<f32> = xi.component_mul(epsilon).add_scalar(-(camera_points[0].0 as f32)).into();
		let ys: na::Vector3<f32> = yi.component_mul(epsilon).add_scalar(-(camera_points[0].1 as f32)).into();
		let i0 = matrix_vector_dot(&self.pseudoinverse, &xs);
		let j0 = matrix_vector_dot(&self.pseudoinverse, &ys);
		let s = j0.dot(&j0) - i0.dot(&i0);
		let ij = i0.dot(&j0);  // How close are these to orthogonal?

		let mut r = 0.0;
		let mut theta = 0.0;

		if s.abs() == 0.0 { // Less than epsilon, maybe?  I don't like the abs comparison with float.
			r = 2.0f32*ij.abs().sqrt();
			if ij < 0.0 {
				theta = std::f32::consts::PI / 2.0;
			} else if ij > 0.0 {
				theta = -std::f32::consts::PI / 2.0;
			} else {
				theta = 0.0;
			}
		} else {
			// See page 12 of "Iterative Pose Estimation Using Coplanar Feature Points" by D OBERKAMPF et al.
			r = ((s*s) + (4.0*ij*ij)).sqrt().sqrt();
			theta = (2.0 * ij / s).atan();
			if s < 0.0 {
				theta += std::f32::consts::PI;
			}
			theta /= 2.0;
		}

		// In the paper these are the real and imaginary parts of C, coming from the system of equations for solutions of coplanar points.
		let lambda = r*theta.cos();
		let mu = r*theta.sin();

		// In one candidate pose we add and in the other we subtract.
		for (cand, vecadd) in [cand1, cand2].iter_mut().zip([true, false]) {
			let (mut i, mut j) = if vecadd {
				(na::Vector3::from((self.model_normal*lambda) + i0),
				na::Vector3::from((self.model_normal*mu) + j0))
			} else {
				(na::Vector3::from((self.model_normal*lambda) - i0),
				na::Vector3::from((self.model_normal*mu) - j0))
			};
			let i_magnitude = i.magnitude();
			let j_magnitude = j.magnitude();
			i.normalize_mut();
			j.normalize_mut();
			let mut k = i.cross(&j);
			// From Rows:
			cand.rotation = na::Matrix3::new(
				i.x, i.y, i.z,
				j.x, j.y, j.z,
				k.x, k.y, k.z
			);
			let mean_candidate1_scale = (i_magnitude + j_magnitude) / 2.0;
			let temp = cand.rotation * self.model_vectors[0]; // Mat3.multVector(rotation1, this.model[0]);
			cand.translation.x = (camera_points[0].0 as f32 / mean_candidate1_scale) - temp[0];
			cand.translation.y = (camera_points[0].1 as f32 / mean_candidate1_scale) - temp[1];
			cand.translation.z = self.focal_length / mean_candidate1_scale;
			cand.error = self.compute_pose_error(camera_points, &cand);
		}
	}

	// Refines a pose and returns the delta error.
	// If you need the error of the new pose, it's a member of the struct.
	fn refine_pose(&self, points: &Vec<(u32, u32)>, pose: &mut MarkerPose) {
		let mut previous_error = pose.error;
		for _ in 0..MAX_POSE_ITERATIONS {
			//assert!(!pose.error.is_nan());
			// eps = Vec3.addScalar( Vec3.multScalar( Mat3.multVector( this.modelVectors, rotation.row(2) ), 1.0 / translation.v[2]), 1.0);
			let rot_row_2 = pose.rotation.row(2);
			let rot_vec_2 = na::Vector3::new(rot_row_2[0], rot_row_2[1], rot_row_2[2]); // It is infuriating we can't just do row(i).into()
			//let mut new_epsilon: na::Vector3<f32> = matrix_vector_dot(&self.model_vectors, &(pose.rotation.row(2).into())).mul(1.0 / pose.translation.z).add_scalar(1.0f32);
			let mut new_epsilon: na::Vector3<f32> = matrix_vector_dot(&self.model_vectors, &rot_vec_2).mul(1.0 / pose.translation.z).add_scalar(1.0f32);
			let mut new_c1 = pose.clone();
			let mut new_c2 = pose.clone();

			self.pose_from_orthography_and_scaling(points, &mut new_epsilon, &mut new_c1, &mut new_c2);

			if new_c1.error < new_c2.error {
				*pose = new_c1;
			} else {
				*pose = new_c2;
			}
			if pose.error <= 2.0 || pose.error > previous_error {
				return;
			}
			previous_error = pose.error;
		}
	}

	// Computes the error for the given pose and points but DOES NOT assign it to the marker.
	// We shouldn't assign the error as this is called with the same marker a bunch to try and find the min error.
	fn compute_pose_error(&self, points: &Vec<(u32, u32)>, pose: &MarkerPose) -> f32 {
		let reprojected_model = (0..4).map(|i| {
			// Mulvector: eps = Vec3.addScalar( Vec3.multScalar( Mat3.multVector( this.modelVectors, rotation.row(2) ), 1.0 / translation.v[2]), 1.0);
			let mut v: na::Vector3<f32> = matrix_vector_dot(&pose.rotation, &self.untransformed_marker_points[i]).add(&pose.translation);
			//let mut v: na::Vector3<f32> = matrix_vector_dot(&pose.rotation, &self.untransformed_marker_points[i]).scale(1.0f32 / pose.translation.z).add_scalar(1.0);
			//let mut v: na::Vector3<f32> = pose.rotation.mul(&self.untransformed_marker_points[i]).add(pose.translation);
			v *= self.focal_length / v.z;
			v
		}).collect::<Vec<na::Vector3<f32>>>();

		let mut errors = 0.0;
		for (idx_a, idx_b, idx_c) in [(0, 1, 3), (1, 2, 0), (2, 1, 3), (3, 0, 2)] {
			let interior_angle = angle_points(&points[idx_a], &points[idx_b], &points[idx_c]);
			let projected_angle = angle2d(&reprojected_model[idx_a], &reprojected_model[idx_b], &reprojected_model[idx_c]);
			let angle_error = (interior_angle - projected_angle).abs() / 4.0;
			errors += angle_error;
		}
		errors
	}
}

fn make_marker_squares(size: f32) -> Vec<na::Vector3<f32>> {
	let hs = size/2.0f32;
	// In screen space, -y is up and +x is right.
	// In world space we're taking Z to be forward from the camera and +x to be right.

	// (This one?) If +y is down and -x is left: lower left, lower right, upper right, upper left, counter-clockwise
	// If +y is up and -x is left: upper left, upper right, lower right, lower left, clockwise
	vec![
		na::Vector3::<f32>::new(-hs, hs, 0.0),
		na::Vector3::<f32>::new(hs, hs, 0.0),
		na::Vector3::<f32>::new(hs, -hs, 0.0),
		na::Vector3::<f32>::new(-hs, -hs, 0.0),
	]
	/*
	vec![
		na::Vector3::<f32>::new(-hs, -hs, 0.0),
		na::Vector3::<f32>::new(-hs, hs, 0.0),
		na::Vector3::<f32>::new(hs, hs, 0.0),
		na::Vector3::<f32>::new(hs, -hs, 0.0),
	]
	*/
}

/// Given ABC, compute the angle between AB and AC.  A is the corner.
fn angle_points(a: &(u32, u32), b: &(u32, u32), c: &(u32, u32)) -> f32 {
	let p = na::Vector3::new(a.0 as f32, a.1 as f32, 0.0f32);
	let q = na::Vector3::new(b.0 as f32, b.1 as f32, 0.0f32);
	let r = na::Vector3::new(c.0 as f32, c.1 as f32, 0.0f32);
	angle(&p, &q, &r)
}

/// Drop the 'z' component from each vector and compute the 2D angle between AB and AC.
fn angle2d(a: &na::Vector3<f32>, b: &na::Vector3<f32>, c: &na::Vector3<f32>) -> f32 {
	let p = na::Vector3::new(a.x, a.y, 0.0);
	let q = na::Vector3::new(b.x, b.y, 0.0);
	let r = na::Vector3::new(c.x, c.y, 0.0);
	angle(&p, &q, &r)
}

/// Compute the angle between lines AB and AC.
fn angle(a: &na::Vector3<f32>, b: &na::Vector3<f32>, c: &na::Vector3<f32>) -> f32 {
	// a dot b = ||a|| * ||b|| * cos(theta)
	//fn angle<D, M>(a: &M, b: &M, c: &M) -> f32 where M: na::Matrix<f32, U1, D, na::ArrayStorage<f32, 1, D>> {
	// It would be really nice to make this generic over Vector.
	let mut p = b-a;
	let mut q = c-a;
	let pmag = p.magnitude();
	let qmag = q.magnitude();
	let dot = p.dot(&q);
	let normalized_dot = dot / (pmag * qmag);
	#[cfg(debug_assertions)]
	if normalized_dot.is_infinite() || normalized_dot.is_nan() || normalized_dot < -1.1f32 || normalized_dot > 1.1f32 {
		eprintln!("angle() called with bad numerical values: {}, {}, {}", &a, &b, &c);
		eprintln!("Normalized Dot: {}\nInfinite: {}\nNormalized is NaN: {}\nNormalized dot < -1.0: {}\nNormalized dot > 1.0: {}", normalized_dot, normalized_dot.is_infinite(), normalized_dot.is_nan(), normalized_dot < -1.0, normalized_dot > 1.0);
	}
	normalized_dot.max(-1.0).min(1.0).acos()
}

/// Computes a new vec3 from the dot product of each matrix row with the given vector.
/// Equivalent to v @ M^T, but returns a vector.
fn matrix_vector_dot(mat: &na::Matrix3<f32>, v: &na::Vector3<f32>) -> na::Vector3<f32> {
	let mt = mat.transpose();
	na::Vector3::new(
		mt.column(0).dot(v),
		mt.column(0).dot(v),
		mt.column(0).dot(v),
	)
}


#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra as na;
	use std::f32::consts::PI;

	#[test]
	fn test_cos_angle() {
		// TODO: Could use better tests, but...
		let mut a = na::Vector3::new(1.0, 0.0, 0.0);
		let mut b = na::Vector3::new(1.0, 0.0, 1.0);
		let mut c = na::Vector3::new(1.0, 1.0, 0.0);
		assert_eq!(angle(&a, &b, &c), std::f32::consts::PI/2.0);

		for i in 0..10 {
			a.x = i as f32;
			b.x = a.x;
			c.x = a.x;
			assert_eq!(angle(&a, &b, &c), std::f32::consts::PI/2.0);
		}
	}

	#[test]
	fn test_cos_angle_domain() {
		// From a bug where we got stuff out of range which caused all kinds of problems.
		let mut a = na::Vector3::new(813.0, 423.0, 0.0);
		let mut b = na::Vector3::new(1095.0, 453.0, 1.0);
		let mut c = na::Vector3::new(769.0, 693.0, 0.0);
		assert!(!angle(&a, &b, &c).is_nan());

		a = na::Vector3::new(-5896.0, -2020.6613, 0.0);
		b = na::Vector3::new(4.2192984, 24.331556, 0.0);
		c = na::Vector3::new(66.10354, 45.78034, 0.0);
		assert!(!angle(&a, &b, &c).is_nan());
	}

	#[test]
	fn test_init() {
		let pe = PoseEstimator::new(10.0, 1.0);
		dbg!(&pe);
		let marker_pts = vec![(0, 10), (10, 10), (10, 0), (0, 0)];
		let (c1, c2) = pe.estimate_marker_pose(&marker_pts);
		dbg!(&c1.translation);
		dbg!(&c1.rotation);
	}

	#[test]
	fn test_detection_from_known_pose() {
		let corners = vec![
			(
				813u32,
				423,
			),
			(
				1098,
				453,
			),
			(
				1071,
				726,
			),
			(
				769,
				693,
			),
		];
		let marker_size_cm = 40f32; // Is this in mm or cm?
		let focal_length_mm = 35f32;
		let sensor_width_mm = 35f32; // Yes, same as width.
		let image_width = 1920;
		let image_height = 1080;
		let camera_transform_m = (0.141521, -0.959589, 2.41665);
		let camera_rotation_deg = (22.76, -0.00019, 6.643);

		let pe = PoseEstimator::new(marker_size_cm, focal_length_mm);

		let (c1, c2) = pe.estimate_marker_pose(&corners);
		dbg!(&c1);
		dbg!(&c2);
	}

	#[test]
	fn test_planar_translations() {
		use nalgebra as na;
		use nalgebra_glm as glm;

		let FOV = PI / 2.0f32;
		let focal_length = 1.0f32;// /(FOV/2.0f32).tan(); // FOV = 2*arctan(sensor_size/2*focal_length) -> tan(FOV/2) = sensor_size/(2*focal_len) -> (2*focal_len)*tan(FOV/2) = sensor_size -> 2*focal_len = sensor_size/tan(FOV/2)
		let image_width = 640f32;
		let image_height = 480f32;
		let camera_position = glm::vec3(0f32, 0f32, -10f32);

		// Four points sitting upright with +y up and +x right.
		// Lower left is at the origin.
		let marker_ll = glm::vec3(0.0f32, 0.0f32, 0.0f32);
		let marker_lr = glm::vec3(0.1f32, 0.0f32, 0.0f32);
		let marker_ur = glm::vec3(0.1f32, 0.1f32, 0.0f32);
		let marker_ul = glm::vec3(0.0f32, 0.1f32, 0.0f32);
		//glm::project_no()

		// [Model Coords] -(model matrix)-> [World Coordinates] -(view matrix)-> [Camera Coordinates] -(projection matrix)-> [Homogeneous Coordinates]
		// Model-View matrix positions the camera in OpenGL (generally via look-at).

		// Stick with OpenGL +y up left-handed.
		let mvm = glm::look_at_lh(&camera_position, &[0f32, 0f32, 0f32].into(), &[0f32, 1f32, 0f32].into());
		let projm = glm::perspective_fov_lh_no(PI / 2.0f32, image_width, image_height, 1.0f32, 100f32);
		let viewport = glm::vec4(0.0f32, 0.0f32, image_width, image_height); // Lower left, upper right, XY, WH.
		// Ordering for projection is UL, UR, LR, LL
		let mut projected_a = glm::project_no(&marker_ul, &mvm, &projm, viewport.clone());
		let mut projected_b = glm::project_no(&marker_ur, &mvm, &projm, viewport.clone());
		let mut projected_c = glm::project_no(&marker_lr, &mvm, &projm, viewport.clone());
		let mut projected_d = glm::project_no(&marker_ll, &mvm, &projm, viewport.clone());
		projected_a /= projected_a.z;
		projected_b /= projected_b.z;
		projected_c /= projected_c.z;
		projected_d /= projected_d.z;

		println!("Projected points to {}, {}, {}, {}.", &projected_a, &projected_b, &projected_c, &projected_d);

		let pe = PoseEstimator::new(10.0, focal_length);
		let marker_pts = vec![
			(projected_a.x as u32, projected_a.y as u32),
			(projected_b.x as u32, projected_b.y as u32),
			(projected_c.x as u32, projected_c.y as u32),
			(projected_d.x as u32, projected_d.y as u32),
		];

		let (c1, c2) = pe.estimate_marker_pose(&marker_pts);
		dbg!(&c1);
		dbg!(&c2);

		//let camera = na::Perspective3::new(1.0f32, 1.0, 1.0, 100.0).as_matrix();
		//let centered_points = make_marker_squares(10.0f32);
		//let transformed_points = centered_points.iter().map(|p| { na::Matrix1x3::from(p.sub(&c1.translation).transpose()).mul(&c1.rotation.transpose()) }).collect::<Vec<na::Matrix1x3<f32>>>();
		//dbg!(transformed_points);
	}
}

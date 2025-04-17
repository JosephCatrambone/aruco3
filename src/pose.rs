use std::ops::{Add, Sub, Mul, Div};
use nalgebra as na;

const INITIAL_SVD_EPSILON: f32 = 1e-5;
const INITIAL_SVD_MAX_ITER: usize = 100;
const MAX_POSE_ITERATIONS: usize = 100;

// This is a separate structure because we need to compute the pseudoinverse of the points for a given marker size + focal dist.
// It's faster to update a pose than to recompute it from scratch.
#[derive(Debug)]
pub struct PoseEstimator {
	image_size: (u32, u32),
	focal_length: f32,
	marker_size: f32,
	untransformed_marker_points: Vec<na::Vector3<f32>>,
	model_vectors: na::Matrix4x3<f32>,
	model_normal: na::Vector3<f32>,
	pseudoinverse: na::Matrix3x4<f32>,
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
	fn new(image_size: (u32, u32), marker_size: f32, focal_length: f32) -> Self {
		let original_frame = make_marker_squares(marker_size);
		let mut model_vectors: na::Matrix4x3<f32> = na::Matrix4x3::new(
			// Why can't we just make this from a list of rows!?  Isn't that the purpose of from_rows!?
			// na::RowVector3::from((original_frame[1] - original_frame[0]).into()),
			0.0f32, 0.0f32, 0.0f32,
			original_frame[1].x - original_frame[0].x, original_frame[1].y - original_frame[0].y, original_frame[1].z - original_frame[0].z,
			original_frame[2].x - original_frame[0].x, original_frame[2].y - original_frame[0].y, original_frame[2].z - original_frame[0].z,
			original_frame[3].x - original_frame[0].x, original_frame[3].y - original_frame[0].y, original_frame[3].z - original_frame[0].z,
		);

		let pseudoinverse = model_vectors.pseudo_inverse(INITIAL_SVD_EPSILON).expect("Failed to compute initial pseudoinverse of model vectors. Is the focal distance real? Is the marker size nonzero?");

		let norm_from_svd = false;
		let model_normal = if norm_from_svd {
			// Double check this:
			let svd: na::linalg::SVD<f32, na::U4, na::U3> = na::linalg::SVD::try_new(model_vectors, true, true, INITIAL_SVD_EPSILON, INITIAL_SVD_MAX_ITER).expect("Failed to compute initial decomposition for marker and vector. This can happen if marker size or focal length is zero.");
			let v_t = svd.v_t.unwrap();
			/*
			// Since we need to compute the SVD anyway it's probably more efficient to use it to build the pseudoinverse, rather than compute it above.
			let u = svd.u.unwrap();
			let s_inverse = na::Matrix3::from_fn_generic(na::U3, na::U3, |i, j| {
				if i != j || svd.singular_values[i].abs() < 1e-6 { 0.0 } else { 1.0 / svd.singular_values[i] }
			});
			//let pseudoinverse: na::Matrix3x4<f32> = u * s_inverse * v_t;
			let pseudoinverse: na::Matrix3x4<f32> = v_t * s_inverse * u.transpose();
			*/

			// This was v.col(min_index) in the original, so it assumes that singular values are sorted.
			// We're sorting greatest to smallest, so we need to take the _max_ column, which is the last one.
			// Also since it's v_t we're grabbing a row.
			// And since columns won't coerce to Vec3's, we do a transpose afterwards.
			//let model_normal: na::Vector3<f32> = v_t.column(2).into();
			let mut model_normal: na::Vector3<f32> = v_t.row(2).transpose();
			model_normal
		} else {
			// I've seen this implemented in two ways: doing the cross and pulling from SVD:
			// The model normal needs to be corrected so make sure the length is nonzero.
			// We already use vec0 as an origin, so try crossing v1 with v2 and v1 with v3.
			let candidate_normal_1 = model_vectors.row(1).cross(&model_vectors.row(2));
			let candidate_normal_2 = model_vectors.row(1).cross(&model_vectors.row(3));
			let mut model_normal: na::Vector3<f32> = if candidate_normal_1.magnitude() > 0.0f32 {
				candidate_normal_1.transpose().into()
			} else if candidate_normal_2.magnitude() > 0.0f32 {
				candidate_normal_2.transpose().into()
			} else {
				panic!("Failed to get normal from marker: points may be degenerate. Is marker size nonzero? Are the points unique?");
			};
			model_normal.normalize_mut();
			model_normal
		};

		Self {
			image_size,
			focal_length,
			marker_size,
			untransformed_marker_points: original_frame,
			model_vectors,
			model_normal,
			pseudoinverse,
		}
	}

	pub fn estimate_marker_pose(&self, points: &Vec<(u32, u32)>) -> (MarkerPose, MarkerPose) {
		// Recenter based on image size AND flip +y to be up:
		let points: Vec<(f32, f32)> = points.iter().map(|p| { (p.0 as f32 - (self.image_size.0 as f32 * 0.5f32), (self.image_size.1 as f32 * 0.5f32) - p.1 as f32)  }).collect();

		let mut candidate1 = MarkerPose::default();
		let mut candidate2 = MarkerPose::default();

		self.make_initial_estimate(&points, &mut candidate1, &mut candidate2);
		//self.refine_pose(&points, &mut candidate1);
		//self.refine_pose(&points, &mut candidate2);

		if candidate1.error <= candidate2.error {
			(candidate1, candidate2)
		} else {
			(candidate2, candidate1)
		}
	}

	fn make_initial_estimate(&self, camera_points: &Vec<(f32, f32)>, cand1: &mut MarkerPose, cand2: &mut MarkerPose) {
		//"Iterative Pose Estimation using Coplanar Feature Points" by Denis Oberkampf, Daniel F. DeMenthon, Larry S. Davis
		// http://www.cfar.umd.edu/~daniel/daniel_papersfordownload/CoplanarPts.pdf

		// Compute the image vectors, a set of points WRT the 0-th item (top left).
		// We expect to have four, with the first being (0,0).
		let marker_origin_x = camera_points[0].0;
		let marker_origin_y = camera_points[0].1;
		let mut image_vectors = Vec::<(f32, f32)>::with_capacity(4);
		for i in 0usize..4 {
			image_vectors.push((camera_points[i].0 as f32 - marker_origin_x, camera_points[i].1 as f32 - marker_origin_y));
		}

		let mut i0 = na::Vector3::new(0f32, 0f32, 0f32);
		let mut j0 = na::Vector3::new(0f32, 0f32, 0f32);
		for j in 0..4 {
			i0.x += self.pseudoinverse.get((0,j)).unwrap() * image_vectors[j].0;
			i0.y += self.pseudoinverse.get((1,j)).unwrap() * image_vectors[j].0;
			i0.z += self.pseudoinverse.get((2,j)).unwrap() * image_vectors[j].0;
			j0.x += self.pseudoinverse.get((0,j)).unwrap() * image_vectors[j].1;
			j0.y += self.pseudoinverse.get((1,j)).unwrap() * image_vectors[j].1;
			j0.z += self.pseudoinverse.get((2,j)).unwrap() * image_vectors[j].1;
		}
		let i0i0 = i0.dot(&i0);
		let j0j0 = j0.dot(&j0);
		let i0j0 = i0.dot(&j0);

		let delta = (j0j0 - i0i0) * (j0j0 - i0i0) + 4.0 * (i0j0*i0j0);
		let q = if j0j0 - i0i0 > 0.0 {
			(j0j0 - i0i0 + delta.sqrt()) / 2.0
		} else {
			(j0j0 - i0i0 - delta.sqrt()) / 2.0
		};

		let mut lambda = 0.0;
		let mut mu = 0.0;
		if q >= 0.0 {
			lambda = q.sqrt();
			mu = if lambda.abs() < 1e-6 {
				0.0
			} else {
				-i0j0 / lambda
			};
		} else {
			lambda = (-(i0j0 * i0j0) / q).sqrt();
			mu = if lambda.abs() < 1e-6 {
				(i0i0 - j0j0).sqrt()
			} else {
				-i0j0 / lambda
			};
		}
		let compute_rot = |out: &mut MarkerPose, lm_sign: f32| {
			let ivec = i0 + self.model_normal.scale(lm_sign * lambda);
			let jvec = j0 + self.model_normal.scale(lm_sign * mu);
			let scale = ivec.magnitude(); // ivec.dot(&ivec).sqrt();
			let row1 = ivec.scale(1.0 / scale);
			let row2 = jvec.scale(1.0 / scale);
			let row3 = row1.cross(&row2);
			out.rotation.set_row(0, &row1.transpose());
			out.rotation.set_row(1, &row2.transpose());
			out.rotation.set_row(2, &row3.transpose());
			out.translation.x = camera_points[0].0 as f32 / scale;
			out.translation.y = camera_points[0].1 as f32 / scale;
			out.translation.z = self.focal_length / scale;
			out.error = self.compute_pose_error(camera_points, &out);
		};
		compute_rot(cand1, 1f32);
		compute_rot(cand2, -1f32);
	}

	// Refines a pose and returns the delta error.
	// If you need the error of the new pose, it's a member of the struct.
	fn refine_pose(&self, points: &Vec<(f32, f32)>, pose: &mut MarkerPose) {
		let mut previous_error = pose.error;
		for _ in 0..MAX_POSE_ITERATIONS {
			//assert!(!pose.error.is_nan());
			// eps = Vec3.addScalar( Vec3.multScalar( Mat3.multVector( this.modelVectors, rotation.row(2) ), 1.0 / translation.v[2]), 1.0);
			let rot_row_2 = pose.rotation.row(2);
			let rot_vec_2 = na::Vector3::new(rot_row_2[0], rot_row_2[1], rot_row_2[2]); // It is infuriating we can't just do row(i).into()
			//let mut new_epsilon: na::Vector3<f32> = matrix_vector_dot(&self.model_vectors, &(pose.rotation.row(2).into())).mul(1.0 / pose.translation.z).add_scalar(1.0f32);
			//let mut new_epsilon: na::Vector3<f32> = matrix_vector_dot(&self.model_vectors, &rot_vec_2).mul(1.0 / pose.translation.z).add_scalar(1.0f32);
			let mut new_c1 = pose.clone();
			let mut new_c2 = pose.clone();

			//self.pose_from_orthography_and_scaling(points, &mut new_epsilon, &mut new_c1, &mut new_c2);
			self.make_initial_estimate(points, &mut new_c1, &mut new_c2);

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
	fn compute_pose_error(&self, points: &Vec<(f32, f32)>, pose: &MarkerPose) -> f32 {
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
fn angle_points(a: &(f32, f32), b: &(f32, f32), c: &(f32, f32)) -> f32 {
	let p = na::Vector3::new(a.0, a.1, 0.0f32);
	let q = na::Vector3::new(b.0, b.1, 0.0f32);
	let r = na::Vector3::new(c.0, c.1, 0.0f32);
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
		let pe = PoseEstimator::new((320, 240), 35.0, 320.0);
		// Vectors internally should be (0,0), (35,0), (35,-35), (0,-35)
		// Obj normal should be 0,0,-1
		// Pseudoinverse should be 3x4
		/*
		0: (4) […]
			0: -4.186041222059461e-19
			1: 0.01904761904761905
			2: 0.009523809523809518
			3: -0.009523809523809528
		1: (4) […]
			0: -8.37208244411892e-19
			1: 0.009523809523809528
			2: -0.009523809523809526
			3: -0.019047619047619056
		2: (4) […]
			0: 0
			1: 0
			2: 0
			3: 0
		*/
		assert!(pe.model_normal.x.abs() < 1e-6f32 && pe.model_normal.y.abs() < 1e-6f32 && (pe.model_normal.z - -1f32).abs() < 1e-6);
		assert_eq!(&pe.pseudoinverse.shape(), &(3usize, 4usize));
		assert!((pe.pseudoinverse - na::Matrix3x4::new(
			0.0, 0.01904761904761905, 0.009523809523809518, -0.009523809523809528,
			0.0, 0.009523809523809528, -0.009523809523809526, -0.019047619047619056,
			0.0, 0.0, 0.0, 0.0
		)).abs().sum() < 1e-5f32);
	}

	#[test]
	fn test_known_pose() {
		let corners = vec![
			(116, 107),
			(142, 105),
			(143, 119),
			(119, 121),
		];
		let marker_size_cm = 35f32; // Is this in mm or cm?
		let focal_length_mm = 320f32;
		let image_width = 320;
		let image_height = 240;

		// Corners should map to (-44, 13), (-18, 15), (-17, 1), (-41, -1)
		// Image vectors: (0, 0), (26, 2), (27, -12), (3, -14)
		let pe = PoseEstimator::new((image_width, image_height), marker_size_cm, focal_length_mm);

		/*
		// Model normal should be (0, 0, -1)
		assert!((pe.pseudoinverse - na::Matrix3x4::new(
			0.0, 0.01904761904761905, 0.009523809523809518, -0.009523809523809528,
			0.0, 0.009523809523809528, -0.009523809523809526, -0.019047619047619056,
			0.0, 0.0, 0.0, 0.0
		)).abs().sum() < 1e-5f32);
		*/

		let (c1, c2) = pe.estimate_marker_pose(&corners);
		assert!(c1.error <= c2.error);
		// i0: 0.7238095238095237, -0.06666666666666665, 0
		// j0: 0.05714285714285727, 0.40000000000000013, 0
		let expected_rotation_1 = na::Matrix3::new(
			0.995229155609543, -0.09166584327982631, -0.03341109098060741,
			0.0785707228112799, 0.5499950596789582, 0.8314638151150369,
			-0.05784089679136317, -0.830122164205087, 0.5545733703973112,
		);
		let expected_rotation_2 = na::Matrix3::new(
			0.995229155609543, -0.09166584327982631, 0.03341109098060741,
			0.0785707228112799, 0.5499950596789582, -0.8314638151150369,
			0.05784089679136317,  0.830122164205087, 0.5545733703973112
		);
		let rotation_error_1 = (c1.rotation - expected_rotation_1).abs().sum();
		let rotation_error_2 = (c2.rotation - expected_rotation_2).abs().sum();
		assert!(rotation_error_1 < 1e-6);
		assert!(rotation_error_2 < 1e-6);
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

		let pe = PoseEstimator::new((image_width as u32, image_height as u32),10.0, focal_length);
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

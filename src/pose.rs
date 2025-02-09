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
			// Why can't we just make this from a list of rows!?  Isnt' that the purpose of from_rows!?
			// na::RowVector3::from((original_frame[1] - original_frame[0]).into()),
			original_frame[1].x - original_frame[0].x, original_frame[1].y - original_frame[0].y, original_frame[1].z - original_frame[0].z,
			original_frame[2].x - original_frame[0].x, original_frame[2].y - original_frame[0].y, original_frame[2].z - original_frame[0].z,
			original_frame[3].x - original_frame[0].x, original_frame[3].y - original_frame[0].y, original_frame[3].z - original_frame[0].z,
		);

		// Double check this:
		// Do we want to use try_new?  Should we use the built-in pseudoinverse?
		/*
		let svd: na::linalg::SVD<f32, na::U3, na::U3> = na::linalg::SVD::new(model_vectors, true, true);
		let v_t = svd.v_t.unwrap();
		let s_inverse = na::Matrix3::from_fn_generic(na::U3, na::U3, |i, j| {
			if i != j { 0.0 } else { 1.0 / svd.singular_values[i] }
		});
		let pseudoinverse: na::Matrix3<f32> = (v_t * s_inverse) * svd.u.unwrap().transpose();
		*/
		let svd: na::linalg::SVD<f32, na::U3, na::U3> = na::linalg::SVD::try_new(model_vectors, true, true, INITIAL_SVD_EPSILON, INITIAL_SVD_MAX_ITER).expect("Failed to compute initial decomposition for marker and vector. This can happen if marker size or focal length is zero.");
		let v_t = svd.v_t.unwrap();
		let pseudoinverse = svd.pseudo_inverse(INITIAL_SVD_EPSILON).expect("Failed to compute pseudoinverse.");

		// This was v.col(min_index) in the original, so it assumes that singular values are sorted.
		// Also since it's v_t we're grabbing a row.
		let model_normal: na::Vector3<f32> = v_t.row(0).transpose();

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
	fn compute_pose_error(&self, points: &Vec<(u32, u32)>, pose: &MarkerPose) -> f32 {
		let reprojected_model = (0..4).map(|i| {
			// Mulvector: eps = Vec3.addScalar( Vec3.multScalar( Mat3.multVector( this.modelVectors, rotation.row(2) ), 1.0 / translation.v[2]), 1.0);
			let mut v: na::Vector3<f32> = matrix_vector_dot(&pose.rotation, &self.untransformed_marker_points[i]).scale(1.0f32 / pose.translation.z).add_scalar(1.0);
			//let mut v: na::Vector3<f32> = pose.rotation.mul(self.planar_pose.row(i)).add(pose.translation);
			v *= self.focal_length / v.z;
			v
		}).collect::<Vec<na::Vector3<f32>>>();

		let errors = (0..4).map(|i| {
			// Angles 013, 120, 231, 302
			// a {0123}, b {1230}, c {3012}
			let idx_a = i;
			let idx_b = (idx_a+1)%4;
			let idx_c = (idx_a+3)%4;
			let interior_angle = angle_points(&points[idx_a], &points[idx_b], &points[idx_c]);
			let projected_angle = angle2d(&reprojected_model[idx_a], &reprojected_model[idx_b], &reprojected_model[idx_c]);
			(interior_angle - projected_angle).abs()
		});

		errors.sum::<f32>() / 4.0
	}
}

fn make_marker_squares(size: f32) -> Vec<na::Vector3<f32>> {
	let hs = size/2.0f32;
	// (This one?) If +y is down and -x is left: lower left, lower right, upper right, upper left, counter-clockwise
	// If += is up and -x is left: upper left, upper right, lower right, lower left, clockwise
	vec![
		na::Vector3::<f32>::new(-hs, hs, 0.0),
		na::Vector3::<f32>::new(hs, hs, 0.0),
		na::Vector3::<f32>::new(hs, -hs, 0.0),
		na::Vector3::<f32>::new(-hs, -hs, 0.0),
	]
}

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
	//fn angle<D, M>(a: &M, b: &M, c: &M) -> f32 where M: na::Matrix<f32, U1, D, na::ArrayStorage<f32, 1, D>> {
	// It would be really nice to make this generic over Vector.
	let p = b-a;
	let q = c-a;
	let dot = p.dot(&q);
	let pmag = p.magnitude();
	let qmag = q.magnitude();
	if pmag < 1e-6 || qmag < 1e-6 {
		return dot.acos();
	}
	dot.acos() / (pmag * qmag)
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
/*
Mat3.multVector = function(m, a){
  m = m.m; a = a.v;

  return new Vec3(
    m[0][0] * a[0] + m[0][1] * a[1] + m[0][2] * a[2],
    m[1][0] * a[0] + m[1][1] * a[1] + m[1][2] * a[2],
    m[2][0] * a[0] + m[2][1] * a[1] + m[2][2] * a[2]);
};

x = r0.x * ax + r0.y *ay ...
x = first mat row * vec
== a * vT
*/

#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra as na;

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
	fn test_init() {
		let pe = PoseEstimator::new(10.0, 1.0);
		dbg!(&pe);
		let marker_pts = vec![(0, 10), (10, 10), (10, 0), (0, 0)];
		let (c1, c2) = pe.estimate_marker_pose(&marker_pts);
		dbg!(&c1.translation);
		dbg!(&c1.rotation);
	}

	#[test]
	fn test_planar_translations() {
		let pe = PoseEstimator::new(10.0, 1.0);
		let marker_pts = vec![(0, 10), (10, 10), (10, 0), (0, 0)]; // Basically don't move from the origin.
		let (c1, c2) = pe.estimate_marker_pose(&marker_pts);
		dbg!(&c1.translation);
		dbg!(&c1.rotation);
		dbg!(&c1.error);

		dbg!(&c2.translation);
		dbg!(&c2.rotation);
		dbg!(&c2.error);
	}
}

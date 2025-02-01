use nalgebra as na;

// This is a separate structure because we need to compute the pseudoinverse of the points for a given marker size + focal dist.
// It's faster to update a pose than to recompute it from scratch.
#[derive(Debug)]
pub struct PoseEstimator {
	focal_length: f32,
	marker_size: f32,
	planar_pose: na::Matrix4x3<f32>,
	model_vectors: na::Matrix3<f32>,
	model_normal: na::Vector3<f32>,
	pseudoinverse: na::Matrix3<f32>,
}

impl PoseEstimator {
	fn new(marker_size: f32, focal_length: f32) -> Self {
		let original_frame = make_marker_squares(marker_size);
		let mut model_vectors: na::Matrix3<f32> = na::Matrix3::from_rows(&[
			//na::RowVector3::
			original_frame.row(1) - original_frame.row(0),
			original_frame.row(2) - original_frame.row(0),
			original_frame.row(3) - original_frame.row(0),
		]);

		// Double check this:
		// Do we want to use try_new?  Should we use the built-in pseudoinverse?
		let svd: na::linalg::SVD<f32, na::U3, na::U3> = na::linalg::SVD::new(model_vectors, true, true);
		let v_t = svd.v_t.unwrap();
		let s_inverse = na::Matrix3::from_fn_generic(na::U3, na::U3, |i, j| {
			if i != j { 0.0 } else { 1.0 / svd.singular_values[i] }
		});
		let pseudoinverse: na::Matrix3<f32> = (v_t * s_inverse) * svd.u.unwrap().transpose();

		// This was v.col(min_index) in the original, so it assumes that singular values are sorted.
		// Also since it's v_t we're grabbing a row.
		let model_normal: na::Vector3<f32> = v_t.row(0).transpose();

		Self {
			focal_length,
			marker_size,
			planar_pose: original_frame,
			model_vectors,
			model_normal,
			pseudoinverse,
		}
	}
}

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

pub fn estimate_marker_pose(marker_size: f32, focal_length: f32, points: Vec<(u32, u32)>) -> (MarkerPose, MarkerPose) {
	

	let mut candidate1 = MarkerPose::default();
	let mut candidate2 = MarkerPose::default();
	
	todo!()
}

fn make_marker_squares(size: f32) -> na::Matrix4x3<f32> {
	let hs = size/2.0f32;
	na::Matrix4x3::new(
		-hs, hs, 0.0,
		hs, hs, 0.0,
		hs, -hs, 0.0,
		-hs, -hs, 0.0,
	)
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
		return f32::NAN;
	}
	dot.acos() / (pmag * qmag)
}


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
		dbg!(pe);
	}
}

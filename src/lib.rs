mod aruco;
mod dictionaries;
mod pose;
mod pinhole;

pub use aruco::{Detector, DetectorConfig, Detection, Marker};
pub use dictionaries::ARDictionary;
pub use pinhole::{CameraIntrinsics};
pub use pose::{MarkerPose, solve_ippe_square as estimate_pose};

pub(crate) fn hamming_distance(a: u64, b: u64) -> u8 {
	let mut flipped_bits = a ^ b;
	let mut flip_count = 0u8;
	while flipped_bits > 0 {
		if flipped_bits % 2 == 1 {
			flip_count += 1;
		}
		flipped_bits = flipped_bits >> 1;
	}
	return flip_count;
}

#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra as na;

	#[test]
	fn test_hamming_distance() {
		// Zero distance for the same numbers.
		for i in 0..255 {
			assert_eq!(hamming_distance(i as u64, i as u64), 0);
		}

		assert_eq!(hamming_distance(0xFFFFFFFF as u64, 0x0 as u64), 32);
		assert_eq!(hamming_distance(0x0 as u64, 0xFFFFFFFF_FFFFFFFF as u64), 64);

		// Test the highest bits.
		assert_eq!(hamming_distance(0b10000000_00000000_00000000_00000000, 0b01000000_00000000_00000000_00000000), 2);
	}
	
	#[test]
	fn test_svd() {
		let m: na::DMatrix<f32> = na::dmatrix![1.0, 2., 3., 4.; 5.0, 10., 20., 30.; -1.0, 0., 1., 0.; 420.0, 69., 1337., 31337.];
		let usvt = na::linalg::SVD::new(m, true, true);
		dbg!(&usvt);
	}
}


use image::{DynamicImage, GrayImage};
use imageproc::contours::Contour;
use imageproc::point::Point;
use crate::dictionaries::ARDictionary;

pub struct Marker {
	pub id: usize,
	pub code: u64,
	pub corners: Vec<(u32, u32)>,
	pub hamming_distance: u8,
}

pub struct Detection {
	pub grey: GrayImage,
	pub candidates: Vec<Vec<Point<u32>>>,
	pub homographies: Vec<GrayImage>,
	pub markers: Vec<Marker>,
}

pub struct DetectorConfig {
	pub threshold_window: u32,
	pub contour_simplification_epsilon: f64,  // Higher gives a more simplified polygon.
	pub min_side_length_factor: f32, // As a function of image width, what's the minimum size length?  2%? 5%?
}

impl Default for DetectorConfig {
	fn default() -> Self {
		DetectorConfig {
			threshold_window: 7,
			contour_simplification_epsilon: 0.05,
			min_side_length_factor: 0.2,
		}
	}
}


struct Detector {
	config: DetectorConfig,
	dictionary: ARDictionary,
}

impl Detector {
	pub fn detect(&self, image: DynamicImage) -> Detection {
		let image_width = image.width();
		let min_edge_length = (image_width as f32 * self.config.min_side_length_factor) as u32;

		// Starting in image space, pull convert to greyscale and convert to contours after threshold.
		//let grey = DynamicImage::ImageRgb8(image).into_luma8();
		let grey = image.into_luma8();
		let thresholded = imageproc::contrast::adaptive_threshold(&grey, self.config.threshold_window);
		let contours = imageproc::contours::find_contours::<u32>(&thresholded);

		// Now that we're in 'point space', get the candidate edges.
		let mut candidate_polygons = self.contours_to_candidates(&contours, min_edge_length);
		enforce_clockwise_corners(&mut candidate_polygons);
		// TODO: Ensure they're not too near.


		Detection {
			grey: grey,
			candidates: candidate_polygons,
			homographies: vec![],
			markers: vec![],
		}
	}

	fn contours_to_candidates(&self, contours: &Vec<Contour<u32>>, min_edge_length: u32) -> Vec<Vec<Point<u32>>> {
		let mut candidate_polygons: Vec<Vec<Point<u32>>> = vec![];
		for c in contours.into_iter() {
			// Ramer–Douglas–Peucker algorithm to simplify the polygon.
			// We start marching down the line and collapse points that are within epsilon.
			let mut edges = imageproc::geometry::approximate_polygon_dp(&c.points, self.config.contour_simplification_epsilon, true);
			// We can't have a quad if it's not four points, convex, with a min edge length above threshold.
			// Point count:
			if edges.len() != 4 {
				continue;
			}
			// Convexity check:
			// TODO: This is a hack. We're finding the convex hull of four points, which will basically mean if there _aren't_ four points after this it was concave.
			// We can probably do a faster convexity check manually.
			edges = imageproc::geometry::convex_hull(edges);
			if edges.len() != 4 {
				continue;
			}
			// Length check.
			let mut total_length = 0u32;
			for i in 0..4 {
				let j = (i+1)%4;
				let dx = (edges[i].x as i32) - (edges[j].x as i32);
				let dy = (edges[i].y as i32) - (edges[j].y as i32);
				total_length += ((dx*dx)+(dy*dy)) as u32;
			}
			if total_length < min_edge_length {
				continue;
			}

			candidate_polygons.push(edges);
		}
		candidate_polygons
	}


}

fn enforce_clockwise_corners(candidate_polygons: &mut Vec<Vec<Point<u32>>>) {
	// Note: This may seem to be anticlockwise, but remember y is down.
	// Since these are guaranteed to be convex by the pervious step, we can take the simple approach.
	for i in 0..candidate_polygons.len() {
		// For each quad, compute the angle between edges AB and AC, then do a simple cosine product.
		// If that's less than zero then it's counter-clockwise and we can simply flip B and D to reverse the direction.
		let dx1: i32 = candidate_polygons[i][1].x as i32 - candidate_polygons[i][0].x as i32;
		let dy1: i32 = candidate_polygons[i][1].y as i32 - candidate_polygons[i][0].y as i32;
		let dx2: i32 = candidate_polygons[i][2].x as i32 - candidate_polygons[i][0].x as i32;
		let dy2: i32 = candidate_polygons[i][2].y as i32 - candidate_polygons[i][0].y as i32;

		if (dx1 * dy2 - dy1 * dx2) < 0 {
			let swap = candidate_polygons[i][1];
			candidate_polygons[i][1] = candidate_polygons[i][3];
			candidate_polygons[i][3] = swap;
		}
	}
}

#[cfg(test)]
mod tests {
	use std::path::Path;
	use std::fs::read_dir;

	use image;
	use crate::dictionaries::ARDictionary;
	use super::*;

	#[test]
	fn test_find_marker() {
		let detector = Detector {
			config: DetectorConfig::default(),
			dictionary: ARDictionary::new_from_named_dict("ARUCO_DEFAULT"),
		};

		let asset_path = Path::new("assets");
		for entry in read_dir(&asset_path).unwrap() {
			let entry = entry.unwrap();
			let path = entry.path();
			let test_image: DynamicImage = image::open(path).unwrap();
			let detection = detector.detect(test_image);
		}
		//dbg!(detection.candidates);
		//assert_eq!(dist, 1);
	}

	#[test]
	fn test_enforce_clockwise() {
		let clockwise = vec![
			Point::new(0, 0), Point::new(0, 1), Point::new(1, 1), Point::new(1, 0),
		];
		let counterclockwise = vec![
			Point::new(0, 0), Point::new(1, 0), Point::new(1, 1), Point::new(0, 1),
		];
		let mut corners = vec![clockwise, counterclockwise];
		enforce_clockwise_corners(&mut corners);
		assert_eq!(&corners[0], &corners[1]);
		dbg!(corners);
	}
}

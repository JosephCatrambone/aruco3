
use image::{DynamicImage, GenericImageView, GrayImage};
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
	pub grey: Option<GrayImage>,
	pub candidates: Vec<Vec<Point<u32>>>,
	pub homographies: Vec<GrayImage>,
	pub markers: Vec<Marker>,
}

pub struct DetectorConfig {
	pub threshold_window: u32,
	pub contour_simplification_epsilon: f64,  // Lower number gives a more simplified polygon.
	pub min_side_length_factor: f32, // As a function of image width, what's the minimum size length?  2%? 5%?
	pub homography_sample_size: usize, // We extract and map the homography to an image of this width and height.  Make sure this is big enough to include all the points in the binary code.
}

impl Default for DetectorConfig {
	fn default() -> Self {
		DetectorConfig {
			threshold_window: 7,
			contour_simplification_epsilon: 0.05,
			min_side_length_factor: 0.2,
			homography_sample_size: 49,
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
		let image_height = image.height();
		let min_edge_length = (image_width.min(image_height) as f32 * self.config.min_side_length_factor) as u32;

		// Starting in image space, pull convert to greyscale and convert to contours after threshold.
		//let grey = DynamicImage::ImageRgb8(image).into_luma8();
		let grey = image.clone().into_luma8();
		let thresholded = imageproc::contrast::adaptive_threshold(&grey, self.config.threshold_window);
		#[cfg(debug_assertions)]
		thresholded.save("DEBUG_thresholded.png");
		let contours = imageproc::contours::find_contours::<u32>(&thresholded);

		// Now that we're in 'point space', get the candidate edges.
		let mut candidate_polygons = contours_to_candidates(&contours, min_edge_length, self.config.contour_simplification_epsilon);
		enforce_clockwise_corners(&mut candidate_polygons);
		// TODO: Ensure they're not too near.

		// Debug: Draw contours on image.
		#[cfg(debug_assertions)]
		{
			let mut debug_image = image.clone();
			for (idx, poly) in candidate_polygons.iter().enumerate() {
				let mut new_poly = vec![];
				let color = [((idx*29)%255usize) as u8, ((idx*23)%255usize) as u8, ((idx*19)%255usize) as u8, 128];
				for p in poly.iter() {
					new_poly.push(Point::new(p.x as i32, p.y as i32));
				}
				debug_image = imageproc::drawing::draw_polygon(&image, new_poly.as_slice(), color.into()).into();
				debug_image.save(format!("DEBUG_detected_polygons_{}.png", idx));
			}
		}

		// Use the polygons to extract chunks from the image.
		let mut homographies = extract_homographies(&grey, &candidate_polygons, self.config.homography_sample_size as u32);


		Detection {
			grey: Some(grey),
			candidates: candidate_polygons,
			homographies: homographies,
			markers: candidate_markers,
		}
	}

}

fn contours_to_candidates(contours: &Vec<Contour<u32>>, min_edge_length: u32, contour_simplification_epsilon: f64) -> Vec<Vec<Point<u32>>> {
	let mut stat_reject_point_count = 0;
	let mut stat_reject_convexity = 0;
	let mut stat_reject_edge_length = 0;
	let mut candidate_polygons: Vec<Vec<Point<u32>>> = vec![];
	for c in contours.into_iter() {
		// Ramer–Douglas–Peucker algorithm to simplify the polygon.
		// We start marching down the line and collapse points that are within epsilon.
		// TODO: Should epsilon be a function of the number of points in the contour or an absolute?
		let mut edges = imageproc::geometry::approximate_polygon_dp(&c.points, c.points.len() as f64 * contour_simplification_epsilon, true);
		// We can't have a quad if it's not four points, convex, with a min edge length above threshold.
		// Point count:
		if edges.len() != 4 {
			stat_reject_point_count += 1;
			continue;
		}
		// Convexity check:
		// TODO: This is a hack. We're finding the convex hull of four points, which will basically mean if there _aren't_ four points after this it was concave.
		// We can probably do a faster convexity check manually.
		edges = imageproc::geometry::convex_hull(edges);
		if edges.len() != 4 {
			stat_reject_convexity += 1;
			continue;
		}
		// Length check.
		let mut candidate_min_edge_length = min_edge_length + 1;
		for i in 0..4 {
			let j = (i+1)%4;
			let dx = (edges[i].x as i32) - (edges[j].x as i32);
			let dy = (edges[i].y as i32) - (edges[j].y as i32);
			candidate_min_edge_length = (((dx*dx)+(dy*dy)) as u32).min(candidate_min_edge_length);
		}
		if candidate_min_edge_length < min_edge_length {
			stat_reject_edge_length += 1;
			continue;
		}

		candidate_polygons.push(edges);
	}
	#[cfg(debug_assertions)]
	println!("DEBUG:\nRejections point count: {}\nReject convexity: {}\nReject edge length: {}\nFound: {}", stat_reject_point_count, stat_reject_convexity, stat_reject_edge_length, &candidate_polygons.len());
	candidate_polygons
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

fn extract_homographies(grey_image: &GrayImage, polygons: &Vec<Vec<Point<u32>>>, homography_size: u32) -> Vec<GrayImage> {
	// tl;dr: Pull out homographies from the pile of polygons, but don't call them markers yet.
	// For each polygon, compute the perspective transform that would be used to make it and then pull it into an image.
	// The reprojected homographies can have a size larger or smaller than they appear in the image, which can help with decoding.

	// Note: We tried pre-cropping around the image, but it didn't actually save compute and made things more complicated, so we crop afterward.
	let mut candidates = vec![];

	polygons.iter().enumerate().for_each(|(polygon_idx, poly)|{
		// Compute the perspective projection for these corners.
		let h = homography_size as f32; // Convenience
		let projection = imageproc::geometric_transformations::Projection::from_control_points(
			[(poly[0].x as f32, poly[0].y as f32), (poly[1].x as f32, poly[1].y as f32), (poly[2].x as f32, poly[2].y as f32), (poly[3].x as f32, poly[3].y as f32)],
			[(0f32, 0f32), (h, 0f32), (h, h), (0f32, h)]
		);

		// The conversion may fail, and if it does add an empty homography.  We could optimize a bit with result types...
		if let Some(projection) = projection {
			// Extract homography:
			let mut homography = imageproc::geometric_transformations::warp(
				grey_image,
				&projection,
				imageproc::geometric_transformations::Interpolation::Bicubic,
				[0u8].into()
			);
			let cropped_homography = image::imageops::crop(&mut homography, 0, 0, homography_size, homography_size).to_image();
			candidates.push(cropped_homography);
		} else {
			candidates.push(GrayImage::new(1, 1));
		}
	});

	candidates
}

fn homography_to_code_permutations(homography: &GrayImage, mark_size: u8) -> Option<(u64, u64, u64, u64)> {
	let otsu_threshold = imageproc::contrast::otsu_level(&homography);
	let binarized = imageproc::contrast::threshold(&homography, otsu_threshold, imageproc::contrast::ThresholdType::Binary);

	// Our homography is larger than the mark size by a decent amount, so we need to sample whole squares.
	//let binary_pixel_width = homography.width() / mark_size as u32;
	//let nonzero_threshold = (binary_pixel_width * binary_pixel_width) / 2;
	//...

	// JC Note: Couldn't we just resize the whole thing to the mark size and threshold?
	let reduced = image::imageops::resize(&binarized, mark_size as u32, mark_size as u32, image::imageops::FilterType::Triangle);

	// Iterate around the perimeter.  If there are any pixels above 127 then it's not a real marker.
	for i in 0..mark_size {
		if (
			reduced.get_pixel(i as u32, 0)[0] > 127 ||
			reduced.get_pixel(i as u32, (mark_size-1) as u32)[0] > 127 ||
			reduced.get_pixel(0, i as u32)[0] > 127 ||
			reduced.get_pixel( (mark_size-1) as u32, i as u32)[0] > 127
		) {
			return None;
		}
	}
	todo!()
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

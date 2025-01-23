
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
	pub grey: Option<GrayImage>,
	pub thresholded: Option<GrayImage>,
	pub homography: Option<GrayImage>,
	pub binary: Option<GrayImage>,
	pub contours: Vec<Vec<Point<u32>>>,
	pub polys: Vec<Vec<Point<u32>>>,
	pub candidates: Vec<Vec<Point<u32>>>,
}

#[derive(Default)]
pub struct DetectorConfig {
	threshold_window: u32,
}

struct Detector {
	config: DetectorConfig,
	dictionary: ARDictionary,
}

impl Detector {
	fn detect(&self, image: DynamicImage) -> Detection {
		//let grey = DynamicImage::ImageRgb8(image).into_luma8();
		let grey = image.into_luma8();
		let thresholded = imageproc::contrast::adaptive_threshold(&grey, self.config.threshold_window);
		let contours = imageproc::contours::find_contours::<u32>(&thresholded);

		todo!()
	}

	fn approximate_polygon(contours: Vec<Contour<f32>>) {

	}
}

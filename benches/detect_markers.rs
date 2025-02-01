use aruco3::{ARDictionary, Detector, DetectorConfig};
use divan::{black_box, Bencher};
use image::{self, Pixel};
use rand::prelude::*;
//use std::path::Path;
//use std::fs::read_dir;


fn main() {
	divan::main();
}

#[divan::bench]
fn bench_detect_markers_real(bencher: Bencher) {
	bencher
		.with_inputs(|| {
			let detector = Detector {
				config: DetectorConfig::default(),
				dictionary: ARDictionary::new_from_named_dict("ARUCO"),
			};
			let image = image::open("assets/benchmark.jpg").expect("Failed to upwrap benchmark image.");
			(detector, image)
		})
		.bench_values(|(detector, img)| {
			detector.detect(img.into())
		});
}

#[divan::bench(args = [(1920, 1080), (1280, 720), (960, 540), (512, 512)])]
fn bench_detect_markers(bencher: Bencher, resolution: (u32, u32)) {
	bencher
		.with_inputs(|| {
			let detector = Detector {
				config: DetectorConfig::default(),
				dictionary: ARDictionary::new_from_named_dict("ARUCO"),
			};
			let mut rng = rand::rng();
			let image = image::RgbImage::from_fn(resolution.0, resolution.1, |x, y|{
				let pixel = [
					rng.random::<u8>(),
					rng.random::<u8>(),
					rng.random::<u8>(),
				];
				pixel.into()
			});
			(detector, image)
		})
		.bench_values(|(detector, img)| {
			detector.detect(img.into())
		});
}


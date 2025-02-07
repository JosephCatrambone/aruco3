/*
 * Copyright 2022 l1npengtul <l1npengtul@protonmail.com> / The Nokhwa Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use nokhwa::{
	nokhwa_initialize,
	pixel_format::{RgbAFormat, RgbFormat, LumaFormat},
	query,
	utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType},
	CallbackCamera,
	Camera,
};
use minifb::{Key, Window, WindowOptions};
use std::env;

fn main() {
	let args: Vec<String> = env::args().collect();

	// only needs to be run on OSX
	nokhwa_initialize(|granted| {
		println!("User said {}", granted);
	});
	let cameras = query(ApiBackend::Auto).unwrap();
	cameras.iter().for_each(|cam| println!("{:?}", cam));

	// We could prompt for cameras here?
	// Parse args instead.
	let mut camera_index:i32 = -1;
	for arg in args.iter() {
		if arg.starts_with("--camera-index=") {
			camera_index = arg.get("--camera-index=".len()..).expect("Failed to get index after '--camera-index='").parse::<i32>().expect("Got invalid index.");
		}
	}
	if camera_index == -1 {
		println!("No camera selected.");
		return;
	}
	let format = RequestedFormat::new::<LumaFormat>(RequestedFormatType::AbsoluteHighestFrameRate);
	let index = CameraIndex::Index(camera_index as u32);
	let mut camera = Camera::new(index, format).expect("Unable to open camera.");
	//camera.open_stream().expect("Failed to open camera stream.");

	// If we didn't have the index and just wanted to grab the first:
	//let first_camera = cameras.first().unwrap();
	let mut threaded = CallbackCamera::new(camera.index().clone(), format, |buffer| {
	 	let image = buffer.decode_image::<LumaFormat>().unwrap();
	}).unwrap();
	threaded.open_stream().unwrap();

	#[allow(clippy::empty_loop)] // keep it running
	loop {
		//let frame = threaded.poll_frame().unwrap();
		//let image = frame.decode_image::<RgbAFormat>().unwrap();

		//if let Ok(frame) = camera.frame() {
		if let Ok(frame) = threaded.poll_frame() {
			let image = frame.decode_image::<LumaFormat>().unwrap();
			println!(
				"{}x{} {} naripoggers",
				image.width(),
				image.height(),
				image.len()
			);
		} else {
			println!("No frame. Waiting.");
		}
	}
}


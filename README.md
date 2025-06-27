# RS-Aruco3 
A pure Rust implementation of Aruco and AprilTag detection/pose estimation.
Ported from the excellent https://github.com/damianofalcioni/js-aruco2 library and using a lightly modified version of the IPPE algorithm as implemented in https://github.com/tobycollins/IPPE.

Detection and quad estimation work well in most cases.  Pose estimations are consistent with what the OpenCV solution reports, though we still need to handle lens distortion.

See webcam_kamera or webcam_nokhwa for usage.

Examples:

```rust
/// Detect markers.
/// Assumes that we have an image from image-rs like this: let img = image::RgbImage::new();
let detector = Detector {
    config: Default::default(),
    dictionary: ARDictionary::new_from_named_dict("ARUCO"),
};

// Detect Markers:
let detections = detector.detect(img.into());
for d in detections.markers.iter() {
    for i in 0..4 {
        // d.corners[i]  The 0-th item is always the top-left corner of a marker and they're wound clockwise.
    }
}
```

```rust
/// Compute a pose.
/// Takes in the marker points detected from a detection 'd' above.
for d in detections.markers.iter() {
    // Each detection has two physically plausible poses. They're sorted by expected accuracy.
    // Also note: this does not undistort the corners of the image. That's not _strictly_ necessary, but...
    let (pose_best, pose_alt) = estimate_pose((1920, 1080), &d.corners, MARKER_SIZE_IN_MM, None /* camera intrinsice can be provided here */); 
    
    // Do something with the pose:
    let marker_points = vec![(0.0, 0.0, 0.0f32), (1.0f32, 0.0, 0.0), (0.0, 1.0f32, 0.0), (0.0, 0.0, 1.0f32)];
    let unproj_pts = pose1.apply_transform_to_points(&marker_points);
        draw_axes(&unproj_pts, &mut window_buffer, w, h);
    }
}
```

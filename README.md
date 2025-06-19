# RS-Aruco3 
A pure Rust implementation of Aruco and AprilTag detection/pose estimation.
Ported from the excellent https://github.com/damianofalcioni/js-aruco2 library.

Detection and quad estimation work well.  Pose estimation is broken, but I will continue to pour my rapidly dwindling sanity into it to find a solution.

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
let mut pose_estimator = PoseEstimator::new((640, 480), 0.04f32, 0.01f32); // 40mm markers on a 640x480 image.
pose_estimator.max_refinement_iterations = 100; // Lower = more noise but faster.

for d in detections.markers.iter() {
    // Each detection has two physically plausible poses. They're sorted by expected accuracy.
    let (pose1, _) = pose_estimator.estimate_marker_pose(&d.corners); 
    
    // Do something with the pose:
    let marker_points = vec![(0.0, 0.0, 0.0f32), (1.0f32, 0.0, 0.0), (0.0, 1.0f32, 0.0), (0.0, 0.0, 1.0f32)];
    let unproj_pts = pose1.apply_transform_to_points(&marker_points);
        draw_axes(&unproj_pts, &mut window_buffer, w, h);
    }
}
```

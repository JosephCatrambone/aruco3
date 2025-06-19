use std::ops::{Add, Sub, Mul, Div};
// Gigantic idiotic monolithic software renderer.
// This will let us render Aruco markers to image files to do a full pipeline test.
use derive_more::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};
use image::{RgbImage, Rgb};

// <editor-fold desc="Float3f32">
#[derive(Clone, Debug, Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign, PartialEq)]
pub struct Float3 {
	pub x: f32,
	pub y: f32,
	pub z: f32,
}

impl Add for &Float3 {
	type Output = Float3;
	fn add(self, other: &Float3) -> Float3 {
		Float3 {
			x: self.x + other.x,
			y: self.y + other.y,
			z: self.z + other.z,
		}
	}
}

impl Sub for &Float3 {
	type Output = Float3;
	fn sub(self, other: &Float3) -> Float3 {
		Float3 {
			x: self.x - other.x,
			y: self.y - other.y,
			z: self.z - other.z,
		}
	}
}

impl Float3 {
	#[inline]
	pub fn element_binary_op_mut(&mut self, other: &Float3, op: fn(f32, f32) -> f32) {
		self.x = op(self.x, other.x);
		self.y = op(self.y, other.y);
		self.z = op(self.z, other.z);
	}

	#[inline]
	pub fn element_binary_op(&self, other: &Float3, op: fn(f32, f32) -> f32) -> Float3 {
		let mut res = self.clone();
		res.element_binary_op_mut(other, op);
		res
	}

	#[inline]
	pub fn element_unary_op_mut(&mut self, op: fn(f32) -> f32) {
		self.x = op(self.x);
		self.y = op(self.y);
		self.z = op(self.z);
	}

	#[inline]
	pub fn element_unary_op(&self, op: fn(f32) -> f32) -> Float3 {
		let mut res = self.clone();
		res.element_unary_op_mut(op);
		res
	}

	pub fn mul_broadcast(&self, v: f32) -> Float3 {
		Float3 { x: self.x * v, y: self.y * v, z: self.z * v }
	}

	pub fn add_broadcast(&self, v: f32) -> Float3 {
		Float3 { x: self.x + v, y: self.y + v, z: self.z + v }
	}

	pub fn dot(&self, other: &Self) -> f32 {
		(self.x * other.x) + (self.y * other.y) + (self.z * other.z)
	}

	pub fn cross(&self, other: &Self) -> Self {
		Float3 {
			x: self.y*other.z - self.z*other.y,
			y: self.z*other.x - self.x*other.z,
			z: self.x*other.y - self.y*other.x,
		}
	}

	pub fn negate(&self) -> Self {
		self.element_unary_op(|v| -v)
	}

	pub fn invert(&self) -> Self {
		self.element_unary_op(|v| 1.0f32 / v)
	}

	pub fn magnitude_squared(&self) -> f32 {
		self.dot(self)
	}

	pub fn magnitude(&self) -> f32 {
		self.magnitude_squared().sqrt()
	}

	pub fn normalized(&self) -> Self {
		let magnitude = self.magnitude();
		Self {
			x: self.x / magnitude,
			y: self.y / magnitude,
			z: self.z / magnitude,
		}
	}

	pub fn new(x: f32, y: f32, z: f32) -> Self {
		Float3 { x, y, z }
	}

	pub fn new_const(val: f32) -> Self {
		Float3 { x: val, y: val, z: val }
	}

	pub fn new_zero() -> Self {
		Float3 { x: 0.0, y: 0.0, z: 0.0 }
	}
}
// </editor-fold>

// <editor-fold desc="Triangle">
#[derive(Clone)]
pub struct Triangle {
	a: Float3,
	b: Float3,
	c: Float3,

	uva: Float3,
	uvb: Float3,
	uvc: Float3,
}

impl Triangle {
	pub fn new(a: Float3, b: Float3, c: Float3) -> Self {
		Triangle { 
			a, b, c, 
			
			uva: Float3::new_zero(), 
			uvb: Float3::new_zero(),
			uvc: Float3::new_zero(),
		}
	}

	pub fn new_from_slice(data: impl AsRef<[f32; 9]>) -> Self {
		let data: &[f32; 9] = data.as_ref();
		Triangle::new(
			Float3::new(
				data[0],
				data[1],
				data[2],
			),
			Float3::new(
				data[3],
				data[4],
				data[5],
			),
			Float3::new(
				data[6],
				data[7],
				data[8],
			)
		)
	}
	
	pub fn point_weights(&self, p: &Float3) -> (f32, f32, f32) {
		let this_area = self.area();
		let abp_area = triangle_area(&self.a, &self.b, p);
		let bcp_area = triangle_area(&self.b, &self.c, p);
		let cap_area = triangle_area(&self.c, &self.a, p);
		(bcp_area / this_area, cap_area / this_area, abp_area / this_area)
	}

	pub fn area(&self) -> f32 {
		triangle_area(&self.a, &self.b, &self.c)
	}

	pub fn point_in_triangle_2d(&self, p: &Float3) -> bool {
		self.point_in_triangle_fast(p)
	}

	fn point_in_triangle_fast(&self, p: &Float3) -> bool {
		if !Triangle::point_right_of_line(&self.a, &self.b, &p) { return false; }
		if !Triangle::point_right_of_line(&self.b, &self.c, &p) { return false; }
		if !Triangle::point_right_of_line(&self.c, &self.a, &p) { return false; }
		true
	}
	
	fn point_in_triangle_slow(&self, p: &Float3) -> bool {
		let (a_weight, b_weight, c_weight) = self.point_weights(p);
		if a_weight > 1.0 { return false; }
		if b_weight > 1.0 { return false; }
		if c_weight > 1.0 { return false; }
		if (1.0 - (a_weight + b_weight + c_weight)).abs() > 1e-5 {
			return false;
		}
		true
	}

	/// Returns true if the point is right of the line from A to B.
	/// DON'T FORGET: +y is DOWN in screen space, so (2, 5) is RIGHT of (5, -10) to (5, 10)
	pub fn point_right_of_line(a: &Float3, b: &Float3, p: &Float3) -> bool {
		let mut ba = b - a;
		let mut pa = p - a;
		ba.z = 0f32;
		pa.z = 0f32;

		// Make pa perpendicular so it's rotated 90 degrees.
		let tmp = pa.y;
		pa.y = -pa.x;
		pa.x = tmp;

		pa.dot(&ba) > 0.0f32
	}
}

fn triangle_area(a: &Float3, b: &Float3, c: &Float3) -> f32 {
	// This is rolled into a separate external method because I don't want to have to allocate a new triangle to do area calcs with a point inside.
	let ac = c - a;
	let ab = b - a;
	0.5f32 * ab.cross(&ac).magnitude()
}
// </editor-fold>

// <editor-fold desc="Transform">
#[derive(Clone)]
pub struct Transform {
	pub origin: Float3,
	pub euler: Float3,
	pub scale: f32,
}

impl Transform {
	pub fn to_world_point(&self, p: &Float3) -> Float3 {
		let (ihat, jhat, khat) = self.get_basis_vectors();
		&self.origin + &Self::transform_vector(&ihat, &jhat, &khat, &p.mul_broadcast(self.scale))
	}

	pub fn to_local_point(&self, p: &Float3) -> Float3 {
		// Invert the transformation from to_world_point.  Takes a point that is rotated and transformed and brings it into local space, undoing transforms, scale, and rotation.
		// If we had done this _THE RIGHT WAY_ with matrices we'd be able to just transpose the matrix, but noooo.
		let (ihat, jhat, khat) = self.get_inverse_basis_vectors();
		Self::transform_vector(&ihat, &jhat, &khat, &(p - &self.origin)).mul_broadcast(1.0 / self.scale)
	}

	pub fn get_basis_vectors(&self) -> (Float3, Float3, Float3) {
		let x = self.euler.x;
		let y = self.euler.y;
		let z = self.euler.z;

		// Yaw (rotation around Y)
		let ihat_yrot = Float3::new(y.cos(), 0f32, y.sin());
		let jhat_yrot = Float3::new(0f32, 1f32, 0f32);
		let khat_yrot = Float3::new((-y).sin(), 0f32, y.cos());
		// Pitch (rotation around x)
		let ihat_xrot = Float3::new(1f32, 0f32, 0f32);
		let jhat_xrot = Float3::new(0f32, x.cos(), (-x).sin());
		let khat_xrot = Float3::new(0f32, x.sin(), x.cos());
		// Roll (rotation around z)
		let ihat_zrot = Float3::new(z.cos(), z.sin(), 0f32);
		let jhat_zrot = Float3::new((-z).sin(), z.cos(), 0f32);
		let khat_zrot = Float3::new(0f32, 0f32, 1f32);

		// Rotate by Z, then X, then Y.

		let ihat_zx = Self::transform_vector(&ihat_xrot, &jhat_xrot, &ihat_xrot, &ihat_zrot);
		let jhat_zx = Self::transform_vector(&ihat_xrot, &jhat_xrot, &ihat_xrot, &jhat_zrot);
		let khat_zx = Self::transform_vector(&ihat_xrot, &jhat_xrot, &ihat_xrot, &khat_zrot);

		let ihat_zxy = Self::transform_vector(&ihat_yrot, &jhat_yrot, &khat_yrot, &ihat_zx);
		let jhat_zxy = Self::transform_vector(&ihat_yrot, &jhat_yrot, &khat_yrot, &jhat_zx);
		let khat_zxy = Self::transform_vector(&ihat_yrot, &jhat_yrot, &khat_yrot, &khat_zx);

		(ihat_zxy.normalized(), jhat_zxy.normalized(), khat_zxy.normalized())
	}
	
	pub fn get_inverse_basis_vectors(&self) -> (Float3, Float3, Float3) {
		self.get_inverse_basis_vectors_fast()
	}

	fn get_inverse_basis_vectors_fast(&self) -> (Float3, Float3, Float3) {
		// The transpose of the rotation matrix is equivalent to the inverse.
		let (ihat, jhat, khat) = self.get_basis_vectors();
		let ihat_inv = Float3::new(ihat.x, jhat.x, khat.x);
		let jhat_inv = Float3::new(ihat.y, jhat.y, khat.y);
		let khat_inv = Float3::new(ihat.z, jhat.z, khat.z);
		(ihat_inv, jhat_inv, khat_inv)
	}

	fn get_inverse_basis_vectors_slow(&self) -> (Float3, Float3, Float3) {
		let x = -self.euler.x;
		let y = -self.euler.y;
		let z = -self.euler.z;
		// Yaw (rotation around Y)
		let ihat_yrot = Float3::new(y.cos(), 0f32, y.sin());
		let jhat_yrot = Float3::new(0f32, 1f32, 0f32);
		let khat_yrot = Float3::new((-y).sin(), 0f32, y.cos());
		// Pitch (rotation around x)
		let ihat_xrot = Float3::new(1f32, 0f32, 0f32);
		let jhat_xrot = Float3::new(0f32, x.cos(), (-x).sin());
		let khat_xrot = Float3::new(0f32, x.sin(), x.cos());
		// Roll (rotation around z)
		// Also let this be the mutable start point.
		let ihat_zrot = Float3::new(z.cos(), z.sin(), 0f32);
		let jhat_zrot = Float3::new((-z).sin(), z.cos(), 0f32);
		let khat_zrot = Float3::new(0f32, 0f32, 1f32);

		// Transform: ZXY.  Reverse: YXZ.

		let ihat_yx = Self::transform_vector(&ihat_xrot, &jhat_xrot, &ihat_xrot, &ihat_yrot);
		let jhat_yx = Self::transform_vector(&ihat_xrot, &jhat_xrot, &ihat_xrot, &jhat_yrot);
		let khat_yx = Self::transform_vector(&ihat_xrot, &jhat_xrot, &ihat_xrot, &khat_yrot);

		let ihat_yxz = Self::transform_vector(&ihat_zrot, &jhat_zrot, &khat_zrot, &ihat_yx);
		let jhat_yxz = Self::transform_vector(&ihat_zrot, &jhat_zrot, &khat_zrot, &jhat_yx);
		let khat_yxz = Self::transform_vector(&ihat_zrot, &jhat_zrot, &khat_zrot, &khat_yx);

		(ihat_yxz.normalized(), jhat_yxz.normalized(), khat_yxz.normalized())
	}

	fn transform_vector(ihat: &Float3, jhat: &Float3, khat: &Float3, v: &Float3) -> Float3 {
		ihat.mul_broadcast(v.x) + jhat.mul_broadcast(v.y) + khat.mul_broadcast(v.z)
	}
}
// </editor-fold>

pub fn render_image(x: f32, y: f32, z: f32, rx: f32, ry: f32, rz: f32) -> RgbImage {
	let tf = Transform {
		origin: Float3::new(x, y, z),
		euler: Float3::new(rx, ry, rz),
		scale: 100.0,
	};
	let quad = [
		Float3::new_zero(),
		Float3::new(1.0, 0.0, 0.0),
		Float3::new(1.0, 1.0, 0.0),
		Float3::new(0.0, 1.0, 0.0),
	];

	let tris = [
		Triangle::new(quad[0].clone(), quad[1].clone(), quad[2].clone()),
		Triangle::new(quad[0].clone(), quad[2].clone(), quad[3].clone()),
	];

	let img = RgbImage::from_fn(320, 240, |x, y| {
		let mut r = 0u8;
		let mut g = 0u8;
		let mut b = 0u8;

		for t in tris.iter() {
			let p_image = Float3::new(x as f32, y as f32, 0f32);
			let p_image_prime = tf.to_local_point(&p_image);
			if t.point_in_triangle_2d(&p_image_prime) {
				// Convert the current triangle to screen coordinates
				let t_screen = Triangle::new(
					Float3::new(t.a.x, t.a.y, 0.0f32),
					Float3::new(t.b.x, t.b.y, 0.0f32),
					Float3::new(t.c.x, t.c.y, 0.0f32),
				);
				let (t_weight_a, t_weight_b, t_weight_c) = t_screen.point_weights(&p_image_prime);
				let depth: f32 = 1.0 / (((1.0 / t.a.z) * t_weight_a) + ((1.0 / t.b.z) * t_weight_b) + ((1.0 / t.c.z) * t_weight_c));
				
				let mut pixel_uv = Float3::new_zero();
				pixel_uv += t.uva.mul_broadcast(1.0 / t.a.z).mul_broadcast(t_weight_a);
				pixel_uv += t.uvb.mul_broadcast(1.0 / t.b.z).mul_broadcast(t_weight_b);
				pixel_uv += t.uvc.mul_broadcast(1.0 / t.c.z).mul_broadcast(t_weight_c);
				pixel_uv = pixel_uv.mul_broadcast(depth);
				
				r = (255.0f32 * t_weight_a) as u8;
				g = (255.0f32 * t_weight_b) as u8;
				b = (255.0f32 * t_weight_c) as u8;
			}
		}

		Rgb::from([r, g, b])
	});

	img
}

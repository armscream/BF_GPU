// BF_GPU/Math.odin
//
// Renderer-local matrix helpers.
//
// BF_Math owns the vector/matrix *types*; the renderer owns the handful of
// operations it needs to turn ECS transforms and camera components into the
// GPU-side matrices in Gpu_Types.odin. Keeping them here avoids growing a
// general math library before the engine needs one, and keeps the conventions
// documented in exactly one place:
//
//   * Mat4 is [16]f32, column-major: element (row r, column c) is m[c * 4 + r].
//   * Mat3 is [9]f32, column-major: element (row r, column c) is n[c * 3 + r].
//   * Projections are right-handed, looking down -Z, with clip depth in [0, 1]
//     (Vulkan convention). The `_y_flip` variant additionally negates Y so the
//     result matches Vulkan's framebuffer orientation.

package BF_GPU

import mth "../../Core/BF_Math"
import "core:math"

MAT4_IDENTITY :: mth.Mat4{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
MAT3_IDENTITY :: [9]f32{1, 0, 0, 0, 1, 0, 0, 0, 1}

// mat4_mul returns a * b (apply b first, then a).
mat4_mul :: proc(a, b: mth.Mat4) -> mth.Mat4 {
	out: mth.Mat4
	for c in 0 ..< 4 {
		for r in 0 ..< 4 {
			sum: f32 = 0
			for k in 0 ..< 4 {
				sum += a[k * 4 + r] * b[c * 4 + k]
			}
			out[c * 4 + r] = sum
		}
	}
	return out
}

// mat4_transform_point applies the full affine transform to a point.
mat4_transform_point :: proc(m: mth.Mat4, p: mth.Vec3) -> mth.Vec3 {
	return mth.Vec3 {
		m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
		m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
		m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14],
	}
}

// mat3_from_mat4 extracts the upper-left 3x3 block.
mat3_from_mat4 :: proc(m: mth.Mat4) -> [9]f32 {
	return [9]f32{m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]}
}

// mat4_from_mat3 widens a 3x3 into an affine 4x4 with no translation.
mat4_from_mat3 :: proc(n: [9]f32) -> mth.Mat4 {
	return mth.Mat4 {
		n[0], n[1], n[2], 0,
		n[3], n[4], n[5], 0,
		n[6], n[7], n[8], 0,
		0, 0, 0, 1,
	}
}

// mat3_inverse inverts the 3x3 block. Returns ok=false (and the identity)
// when the matrix is singular, which happens for zero-scaled transforms.
mat3_inverse :: proc(n: [9]f32) -> (out: [9]f32, ok: bool) {
	a, b, c := n[0], n[3], n[6] // row 0
	d, e, f := n[1], n[4], n[7] // row 1
	g, h, i := n[2], n[5], n[8] // row 2

	det := a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
	if abs(det) < 1e-12 {
		return MAT3_IDENTITY, false
	}
	inv_det := 1.0 / det

	// Row-major adjugate / det.
	iv: [3][3]f32
	iv[0][0] = (e * i - f * h) * inv_det
	iv[0][1] = (c * h - b * i) * inv_det
	iv[0][2] = (b * f - c * e) * inv_det
	iv[1][0] = (f * g - d * i) * inv_det
	iv[1][1] = (a * i - c * g) * inv_det
	iv[1][2] = (c * d - a * f) * inv_det
	iv[2][0] = (d * h - e * g) * inv_det
	iv[2][1] = (b * g - a * h) * inv_det
	iv[2][2] = (a * e - b * d) * inv_det

	for col in 0 ..< 3 {
		for row in 0 ..< 3 {
			out[col * 3 + row] = iv[row][col]
		}
	}
	return out, true
}

// mat3_normal_from_mat4 produces the inverse-transpose of the upper-left 3x3,
// i.e. the matrix that transforms normals correctly under non-uniform scale.
// Falls back to the plain 3x3 block when the transform is singular.
mat3_normal_from_mat4 :: proc(m: mth.Mat4) -> [9]f32 {
	basis := mat3_from_mat4(m)
	inv, ok := mat3_inverse(basis)
	if !ok {
		return basis
	}
	// transpose(inverse)
	out: [9]f32
	for col in 0 ..< 3 {
		for row in 0 ..< 3 {
			out[col * 3 + row] = inv[row * 3 + col]
		}
	}
	return out
}

// mat4_affine_inverse inverts a transform matrix whose last row is (0,0,0,1).
// Handles rotation, scale and shear; returns ok=false for singular bases.
mat4_affine_inverse :: proc(m: mth.Mat4) -> (out: mth.Mat4, ok: bool) {
	basis := mat3_from_mat4(m)
	inv3, inv_ok := mat3_inverse(basis)
	if !inv_ok {
		return MAT4_IDENTITY, false
	}

	out[0], out[1], out[2] = inv3[0], inv3[1], inv3[2]
	out[4], out[5], out[6] = inv3[3], inv3[4], inv3[5]
	out[8], out[9], out[10] = inv3[6], inv3[7], inv3[8]
	out[3], out[7], out[11] = 0, 0, 0

	tx, ty, tz := m[12], m[13], m[14]
	// -inv3 * t
	out[12] = -(inv3[0] * tx + inv3[3] * ty + inv3[6] * tz)
	out[13] = -(inv3[1] * tx + inv3[4] * ty + inv3[7] * tz)
	out[14] = -(inv3[2] * tx + inv3[5] * ty + inv3[8] * tz)
	out[15] = 1
	return out, true
}

// mat4_perspective builds a right-handed perspective projection with clip
// depth in [0, 1]. fov_y is in radians.
mat4_perspective :: proc(fov_y, aspect, near, far: f32) -> mth.Mat4 {
	out: mth.Mat4
	if fov_y <= 0 || aspect <= 0 || near <= 0 || far <= near {
		return MAT4_IDENTITY
	}
	f := 1.0 / math.tan(fov_y * 0.5)
	out[0] = f / aspect
	out[5] = f
	out[10] = far / (near - far)
	out[11] = -1
	out[14] = (far * near) / (near - far)
	return out
}

// mat4_orthographic builds a right-handed orthographic projection with clip
// depth in [0, 1]. `height` is the full vertical extent in world units.
mat4_orthographic :: proc(height, aspect, near, far: f32) -> mth.Mat4 {
	out: mth.Mat4
	if height <= 0 || aspect <= 0 || far <= near {
		return MAT4_IDENTITY
	}
	out[0] = 2.0 / (height * aspect)
	out[5] = 2.0 / height
	out[10] = 1.0 / (near - far)
	out[14] = near / (near - far)
	out[15] = 1
	return out
}

// mat4_y_flip negates the Y row, converting a Y-up projection into Vulkan's
// framebuffer orientation.
mat4_y_flip :: proc(m: mth.Mat4) -> mth.Mat4 {
	out := m
	out[1] = -out[1]
	out[5] = -out[5]
	out[9] = -out[9]
	out[13] = -out[13]
	return out
}

// frustum_planes_from_view_proj extracts the six clip planes in world space,
// ordered left, right, bottom, top, near, far. Each plane is (nx, ny, nz, d)
// normalised so that dot(plane.xyz, p) + plane.w > 0 means "inside".
frustum_planes_from_view_proj :: proc(vp: mth.Mat4) -> [6]mth.Vec4 {
	row :: #force_inline proc(m: mth.Mat4, r: int) -> mth.Vec4 {
		return mth.Vec4{m[r], m[4 + r], m[8 + r], m[12 + r]}
	}
	r0 := row(vp, 0)
	r1 := row(vp, 1)
	r2 := row(vp, 2)
	r3 := row(vp, 3)

	planes: [6]mth.Vec4
	planes[0] = r3 + r0 // left
	planes[1] = r3 - r0 // right
	planes[2] = r3 + r1 // bottom
	planes[3] = r3 - r1 // top
	planes[4] = r2 // near (clip depth 0..1)
	planes[5] = r3 - r2 // far

	for &p in planes {
		len := math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
		if len > 1e-12 {
			p /= len
		}
	}
	return planes
}

// aabb_to_culling_bounds converts a world-space AABB into the sphere + box
// pair the culling pipeline consumes.
aabb_to_culling_bounds :: proc(box: mth.AABB) -> Culling_Bounds {
	centre := mth.Vec3 {
		(box.min.x + box.max.x) * 0.5,
		(box.min.y + box.max.y) * 0.5,
		(box.min.z + box.max.z) * 0.5,
	}
	ex := box.max.x - centre.x
	ey := box.max.y - centre.y
	ez := box.max.z - centre.z
	return Culling_Bounds {
		centre = centre,
		radius = math.sqrt(ex * ex + ey * ey + ez * ez),
		min = box.min,
		max = box.max,
	}
}

// aabb_transform returns the world-space AABB of `box` after `m`, using the
// standard "transform the 8 corners" expansion.
aabb_transform :: proc(box: mth.AABB, m: mth.Mat4) -> mth.AABB {
	corners := [8]mth.Vec3 {
		{box.min.x, box.min.y, box.min.z},
		{box.max.x, box.min.y, box.min.z},
		{box.min.x, box.max.y, box.min.z},
		{box.max.x, box.max.y, box.min.z},
		{box.min.x, box.min.y, box.max.z},
		{box.max.x, box.min.y, box.max.z},
		{box.min.x, box.max.y, box.max.z},
		{box.max.x, box.max.y, box.max.z},
	}
	out: mth.AABB
	for corner, i in corners {
		p := mat4_transform_point(m, corner)
		if i == 0 {
			out.min = p
			out.max = p
			continue
		}
		out.min.x = min(out.min.x, p.x)
		out.min.y = min(out.min.y, p.y)
		out.min.z = min(out.min.z, p.z)
		out.max.x = max(out.max.x, p.x)
		out.max.y = max(out.max.y, p.y)
		out.max.z = max(out.max.z, p.z)
	}
	return out
}

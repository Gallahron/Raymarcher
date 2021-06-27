#pragma once
#include <stdio.h>
#include <SDL.h>

#define VZERO Vector3 { 0,0,0 }
#define VONE Vector3 { 1,1,1 }

class Vector3 {
public:
	float x;
	float y;
	float z;
	__device__ __host__
		Vector3() {
		x = 0;
		y = 0;
		z = 0;
	}
	__device__ __host__
		Vector3(float nx, float ny, float nz) {
		x = nx;
		y = ny;
		z = nz;
	}

	__device__ __host__
		Vector3 Add(Vector3 b) {
		Vector3 result;
		result.x = x + b.x;
		result.y = y + b.y;
		result.z = z + b.z;
		return result;
	}

	__device__ __host__
		Vector3 Sub(Vector3 b) {
		Vector3 result;
		result.x = x - b.x;
		result.y = y - b.y;
		result.z = z - b.z;
		return result;
	}

	__device__ __host__
		Vector3 Mul(float b) {
		Vector3 result;
		result.x = x * b;
		result.y = y * b;
		result.z = z * b;
		return result;
	}

	__device__ __host__
		float mag() {
#ifdef __CUDA_ARCH__
		return __fsqrt_rd(__fmaf_rd(x, x, __fmaf_rd(y, y, __fmul_rd(z, z))));
#else
		return sqrt(x * x + y * y + z * z);
#endif
	}

	__device__ __host__
		float Dist(Vector3 b) {
		Vector3 result;
		result = Sub(b);
		return result.mag();
	}

	__device__ __host__
		Vector3 RotX(float angle) {

		struct Vector3 result;
#ifdef __CUDA_ARCH__
		float cosThet;
		float sinThet;
		__sincosf(angle, &sinThet, &cosThet);
#else
		float cosThet = cos(angle);
		float sinThet = sin(angle);
#endif
		result.x = x;
		result.y = y * cosThet - z * sinThet;
		result.z = y * sinThet + z * cosThet;

		return result;
	}
	__device__ __host__
		Vector3 RotY(float angle) {
		struct Vector3 result;
#ifdef __CUDA_ARCH__
		float cosThet;
		float sinThet;
		__sincosf(angle, &sinThet, &cosThet);
#else
		float cosThet = cos(angle);
		float sinThet = sin(angle);
#endif
		result.x = x * cosThet + z * sinThet;
		result.y = y;
		result.z = -x * sinThet + z * cosThet;
		return result;
	}
	__device__ __host__
		Vector3 RotZ(float angle) {
		struct Vector3 result;

#ifdef __CUDA_ARCH__
		float cosThet;
		float sinThet;
		__sincosf(angle, &sinThet, &cosThet);
#else
		float cosThet = cos(angle);
		float sinThet = sin(angle);
#endif

		result.x = x * cosThet - y * sinThet;
		result.y = x * sinThet + y * cosThet;
		result.z = z;
		return result;
	}
	__device__ __host__
		float Dot(Vector3 b) {
#ifdef __CUDA_ARCH__
		return __fsqrt_rd(__fmaf_rd(x, b.x, __fmaf_rd(y, b.y, __fmul_rd(z, b.z))));
#else
		return x * b.x + y * b.y + z * b.z;
#endif
	}
	__device__ __host__
		Vector3 normalised() {
		Vector3 result;
		float magnitude = mag();
		result.x = x / magnitude;
		result.y = y / magnitude;
		result.z = z / magnitude;
		return result;
	}
	__device__ __host__
		Vector3 abs() {
		Vector3 result;
		result.x = fabsf(x);
		result.y = fabsf(y);
		result.z = fabsf(z);
		return result;
	}
	__device__ __host__
		Vector3 ApplyRot(Vector3 trans) {
		Vector3 out = Vector3(x, y, z);
		out = out.RotX(trans.x);
		out = out.RotY(trans.y);
		out = out.RotZ(trans.z);
		return out;
	}
	__device__ __host__
		Vector3 Negative() {
		Vector3 out;
		out.x = -x;
		out.y = -y;
		out.z = -z;
		return out;
	}
	__device__ __host__
		Vector3 Lerp(Vector3 b, float s) {
		Vector3 result;
		result.x = x + (b.x - x) * s;
		result.y = y + (b.y - y) * s;
		result.z = z + (b.z - z) * s;
		return result;
	}
	__device__ __host__
		Vector3 Max(Vector3 oth) {
		Vector3 result;
		result.x = fmaxf(x, oth.x);
		result.y = fmaxf(y, oth.y);
		result.z = fmaxf(z, oth.z);
		return result;
	}
	__device__ __host__
		Vector3 Min(Vector3 oth) {
		Vector3 result;
		result.x = fminf(x, oth.x);
		result.y = fminf(y, oth.y);
		result.z = fminf(z, oth.z);
		return result;
	}
};

class Transform {
public:
	Vector3 pos;
	Vector3 rot;
	Vector3 sca;

	__device__ __host__
		void Translate(Vector3 translation) {
		pos = pos.Add(translation);
	}
	__device__ __host__
	void Rotate(Vector3 rotation) {
		rot = rot.Add(rotation);
	}

	__device__ __host__
	void Scale(Vector3 mod) {
		sca.x *= mod.x;
		sca.y *= mod.y;
		sca.z *= mod.z;
	}
};
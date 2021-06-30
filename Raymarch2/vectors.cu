#pragma once
#include <stdio.h>
#include <SDL.h>
#include <cuda_fp16.h>

#define VZERO Vector3 { 0,0,0 }
#define VONE Vector3 { 1,1,1 }

__device__
__half hmax(__half a, __half b) {
	return __hgt(a, b) ? a : b;
}
__device__
__half hmin(__half a, __half b) {
	return __hlt(a, b) ? a : b;
}



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
		return __fmaf_rd(x, b.x, __fmaf_rd(y, b.y, __fmul_rd(z, b.z)));
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



class devVector3 {
public:
	__half x;
	__half y;
	__half z;
	__device__ __host__
	devVector3() {
		x = __int2half_rn(0);
		y = __int2half_rn(0);
		z = __int2half_rn(0);
	}
	__device__ __host__
	devVector3(float nx, float ny, float nz) {
		x = __float2half(nx);
		y = __float2half(ny);
		z = __float2half(nz);
	}
	__device__ __host__
	devVector3(__half nx, __half ny, __half nz) {
		x = nx;
		y = ny;
		z = nz;
	}
	__device__ __host__
	devVector3(Vector3 vector) : devVector3(vector.x, vector.y, vector.z) {}

	__device__
	devVector3 Add(devVector3 b) {
		devVector3 result;
		result.x = __hadd(x, b.x);
		result.y = __hadd(y, b.y);
		result.z = __hadd(z, b.z);
		return result;
	}

	__device__
	devVector3 Sub(devVector3 b) {
		devVector3 result;
		result.x = __hsub(x, b.x);
		result.y = __hsub(y, b.y);
		result.z = __hsub(z, b.z);
		return result;
	}

	__device__
	devVector3 Mul(__half b) {
		devVector3 result;
		result.x = __hmul(x, b);
		result.y = __hmul(y, b);
		result.z = __hmul(z, b);
		return result;
	}

	__device__
	__half mag() {
		return hsqrt(__hfma(x, x, __hfma(y, y, __hmul(z, z))));
	}

	__device__
	__half Dist(devVector3 b) {
		devVector3 result;
		result = Sub(b);
		return result.mag();
	}

	__device__
	devVector3 RotX(__half angle) {
		struct devVector3 result;

		__half sinThet = hsin(angle);
		__half cosThet = hcos(angle);

		result.x = x;
		result.y = __hsub(__hmul(y, cosThet), __hmul(z, sinThet));
		result.z = __hadd(__hmul(y, sinThet), __hmul(z, cosThet));

		return result;
	}
	__device__
	devVector3 RotY(__half angle) {
		struct devVector3 result;

		__half sinThet = hsin(angle);
		__half cosThet = hcos(angle);

		result.x = __hadd(__hmul(x, cosThet), __hmul(z, sinThet));
		result.y = y;
		result.z = __hsub(__hmul(z, cosThet), __hmul(x, sinThet));
		return result;
	}
	__device__
	devVector3 RotZ(__half angle) {
		struct devVector3 result;

		__half sinThet = hsin(angle);
		__half cosThet = hcos(angle);

		result.x = __hsub(__hmul(x, cosThet), __hmul(y, sinThet));
		result.y = __hadd(__hmul(x, sinThet), __hmul(y, cosThet));
		result.z = z;
		return result;
	}
	__device__
	__half Dot(devVector3 b) {
		return __hfma(x, b.x, __hfma(y, b.y, __hmul(z, b.z)));
	}
	__device__
	devVector3 normalised() {
		devVector3 result;
		float magnitude = mag();
		result.x = __hdiv(x, magnitude);
		result.y = __hdiv(y, magnitude);
		result.z = __hdiv(z, magnitude);
		return result;
	}
	__device__
	devVector3 abs() {
		devVector3 result;
		result.x = __habs(x);
		result.y = __habs(y);
		result.z = __habs(z);
		return result;
	}
	__device__
	devVector3 ApplyRot(devVector3 trans) {
		devVector3 out = devVector3(x, y, z);
		out = out.RotX(trans.x);
		out = out.RotY(trans.y);
		out = out.RotZ(trans.z);
		return out;
	}
	__device__
	devVector3 Negative() {
		devVector3 out;
		out.x = __hneg(x);
		out.y = __hneg(y);
		out.z = __hneg(z);
		return out;
	}
	__device__
	devVector3 Lerp(devVector3 b, __half s) {
		devVector3 result;
		result.x = __hfma(__hsub(b.x, x), s, x);
		result.y = __hfma(__hsub(b.y, y), s, y);
		result.z = __hfma(__hsub(b.z, z), s, z);
		return result;
	}
	__device__
	devVector3 Max(devVector3 oth) {
		devVector3 result;
		result.x = hmax(x, oth.x);
		result.y = hmax(y, oth.y);
		result.z = hmax(z, oth.z);
		return result;
	}
	__device__
	devVector3 Min(devVector3 oth) {
		devVector3 result;
		result.x = hmax(x, oth.x);
		result.y = hmax(y, oth.y);
		result.z = hmax(z, oth.z);
		return result;
	}

	__device__ __host__
		Vector3 ToVector3() {
		return Vector3{
			__half2float(x),
			__half2float(y),
			__half2float(z)
		};
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

class devTransform {
public:
	devVector3 pos;
	devVector3 rot;
	devVector3 sca;

	devTransform() = default;
	__device__ __host__
	devTransform(Transform trans) {
		pos = devVector3(trans.pos);
		rot = devVector3(trans.rot);
		sca = devVector3(trans.sca);
	}

	__device__
		void Translate(devVector3 translation) {
		pos = pos.Add(translation);
	}
	__device__
		void Rotate(Vector3 rotation) {
		rot = rot.Add(rotation);
	}

	__device__
		void Scale(Vector3 mod) {
		sca.x = __hmul(sca.x, mod.x);
		sca.y = __hmul(sca.y, mod.y);
		sca.z = __hmul(sca.z, mod.z);
	}
	
};
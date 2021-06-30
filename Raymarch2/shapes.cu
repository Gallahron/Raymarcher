
#include "vectors.cu"
#include <stdio.h>
#include <SDL.h>
#include <cuda_fp16.h>

struct DistReturn {
	Vector3 col;
	float dist;
};
struct devDistReturn {
	devVector3 col;
	__half dist;
};

class Light {
public:
	Vector3 pos;
	float intensity;
	Light(Vector3 newPos) {
		pos = newPos;
	}
};
class devLight {
public:
	devVector3 pos;
	__half intensity;
	devLight (devVector3 newPos) {
		pos = newPos;
	}
	devLight(Light light) {
		pos = devVector3(light.pos);
		intensity = __float2half(light.intensity);
	}
};

class Shape {
public:
	char type;
	Transform trans;
	Vector3 col;
	int blend;

	__device__ __host__
		Shape(float x, float y, float z, float rx, float ry, float rz, char typeof, int blended) {
		trans.pos.x = x;
		trans.pos.y = y;
		trans.pos.z = z;

		trans.rot.x = rx;
		trans.rot.y = ry;
		trans.rot.z = rz;

		trans.sca = VONE;

		type = typeof;
		blend = blended;
	}
	__device__ __host__
		Shape(Vector3 newpos, Vector3 newrot, char typeof, int blended) {
		trans.pos = newpos;
		trans.rot = newrot;
		trans.sca = VONE;

		type = typeof;
		blend = blended;
	}
	__device__
		DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.col = col;
		result.dist = 1;
		return result;
	};
	__device__
		DistReturn EstimatedDistance(Vector3 currPos) {
		return DistanceTo(currPos);
	}

	__device__
		Vector3 TransformPoint(Vector3 currpos) {
		return currpos.Sub(trans.pos).ApplyRot(trans.rot.Negative());
	}

	__device__
		virtual Vector3 GetNormal(Vector3 surfacePos) { return VZERO; };
};

class Sphere : public Shape {
public:
	__device__ __host__
		Sphere(float x, float y, float z, float rad, int blended) : Shape(x, y, z, 0, 0, 0, 's', blended) {
		trans.sca = Vector3{ rad, rad, rad };
	}
	__device__ __host__
		Sphere(Vector3 newpos, float rad, int blended) : Shape(newpos, Vector3(0, 0, 0), 's', blended) {
		trans.sca = Vector3{ rad, rad, rad };
	}
	__device__
		DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = trans.pos.Dist(currPos) - trans.sca.x;
		result.col = col;
		return result;
	}
	__device__
		Vector3 GetNormal(Vector3 surfacePos) {
		return surfacePos.Sub(trans.pos).normalised();
	}
};

class Cube : public Shape {
public:
	__device__ __host__
		Cube(float x, float y, float z, float rx, float ry, float rz, float bx, float by, float bz, int blended) : Shape(x, y, z, rx, ry, rz, 'c', blended) {
		trans.sca = Vector3(bx, by, bz);
	}
	__device__ __host__
		Cube(Vector3 newpos, Vector3 newrot, Vector3 bound, int blended) : Shape(newpos, newrot, 'c', blended) {
		trans.sca = bound;
	}
	__device__
		DistReturn DistanceTo(Vector3 currPos) {
		Vector3 delta = TransformPoint(currPos);
		Vector3 q = delta.abs().Sub(trans.sca);
		DistReturn result;
		delta.x = fmaxf(q.x, 0);
		delta.y = fmaxf(q.y, 0);
		delta.z = fmaxf(q.z, 0);
		result.dist = delta.mag() + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0);
		result.col = col;
		return result;
	}
	__device__
		Vector3 GetNormal(Vector3 surfacePos) {
		return surfacePos.Sub(trans.pos).normalised();
	}
	__device__
		DistReturn EstimatedDistance(Vector3 currPos) {
		DistReturn result;
		float rad = trans.sca.mag();
		result.dist = trans.pos.Dist(currPos) - rad;
		return result;
	}
};
class HollowCube : public Shape {
public:
	float thickness;
	__device__ __host__
		HollowCube(float x, float y, float z, float rx, float ry, float rz, float bx, float by, float bz, float e, int blended) : Shape(x, y, z, rx, ry, rz, 'h', blended) {
		trans.sca = Vector3(bx, by, bz);
		thickness = e;
	}
	__device__ __host__
		HollowCube(Vector3 newpos, Vector3 newrot, Vector3 bound, float e, int blended) : Shape(newpos, newrot, 'h', blended) {
		trans.sca = bound;
		thickness = e;
	}
	__device__
		DistReturn DistanceTo(Vector3 currPos) {
		Vector3 delta = TransformPoint(currPos);
		Vector3 p = delta.abs().Sub(trans.sca);
		Vector3 q = p;
		DistReturn result;
		q.x += thickness;
		q.y += thickness;
		q.z += thickness;
		q = q.abs();
		q.x -= thickness;
		q.y -= thickness;
		q.z -= thickness;

		float a = Vector3(p.x, q.y, q.z).Max(Vector3(0, 0, 0)).mag() + fminf(fmaxf(p.x, fmaxf(q.y, q.z)), 0);
		float b = Vector3(q.x, p.y, q.z).Max(Vector3(0, 0, 0)).mag() + fminf(fmaxf(q.x, fmaxf(p.y, q.z)), 0);
		float c = Vector3(q.x, q.y, p.z).Max(Vector3(0, 0, 0)).mag() + fminf(fmaxf(q.x, fmaxf(q.y, p.z)), 0);

		result.dist = fminf(fminf(a, b), c);
		//result.dist = fminf(fminf(a,b), c);
		result.col = col;
		/*
		delta.x = fmaxf(q.x, 0);
		delta.y = fmaxf(q.y, 0);
		delta.z = fmaxf(q.z, 0);
		result.dist = delta.mag() + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0);
		result.col = col;*/
		return result;
	}
	__device__
		Vector3 GetNormal(Vector3 surfacePos) {
		return surfacePos.Sub(trans.pos).normalised();
	}
	__device__
		DistReturn EstimatedDistance(Vector3 currPos) {
		DistReturn result;
		float rad = trans.sca.mag();
		result.dist = trans.pos.Dist(currPos) - rad;
		return result;
	}
};

class Plane : public Shape {
public:
	__device__ __host__
		Plane(Vector3 newpos, int blended) : Shape(newpos, VZERO, 'p', blended) { }
	__device__ __host__
		Plane(float height, int blended) : Shape(Vector3(0, height, 0), VZERO, 'p', blended) {}
	__device__
		DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = currPos.y - trans.pos.y;
		result.col = col;
		return result;
	}
	__device__
		Vector3 GetNormal(Vector3 surfacePos) {
		return Vector3(0, 1, 0);
	}
};



class devShape {
public:
	char type;
	devTransform trans;
	devVector3 col;
	int blend;

	__host__ __device__
	devShape(Shape shape) {
		type = shape.type;
		trans = devTransform(shape.trans);
		col = devVector3(shape.col);
		blend = shape.blend;
	}
	__device__
	devDistReturn DistanceTo(devVector3 currPos) {
		devDistReturn result;
		result.col = col;
		result.dist = currPos.x;
		return result;
	};
	__device__
	devDistReturn EstimatedDistance(Vector3 currPos) {
		return DistanceTo(currPos);
	}

	__device__
	devVector3 TransformPoint(devVector3 currpos) {
		return currpos.Sub(trans.pos).ApplyRot(trans.rot.Negative());
	}

	__device__
		virtual devVector3 GetNormal(devVector3 surfacePos) { return VZERO; };
};

class devSphere : public devShape {
public:
	__device__ __host__
	devSphere(Sphere orig) : devShape(orig) {};
	__device__
	devDistReturn DistanceTo(devVector3 currPos) {
		devDistReturn result;
		result.dist = __hsub(trans.pos.Dist(currPos), trans.sca.x);
		result.col = col;
		return result;
	}
	__device__
	devVector3 GetNormal(devVector3 surfacePos) {
		return surfacePos.Sub(trans.pos).normalised();
	}
};

class devCube : public devShape {
public:
	__device__ __host__
	devCube(Cube orig) : devShape(orig) {};
	__device__
	devDistReturn DistanceTo(devVector3 currPos) {
		devVector3 delta = TransformPoint(currPos);
		devVector3 q = delta.abs().Sub(trans.sca);
		devDistReturn result;
		delta.x = hmax(q.x, __int2half_rd(0));
		delta.y = hmax(q.y, __int2half_rd(0));
		delta.z = hmax(q.z, __int2half_rd(0));
		result.dist = __hadd(delta.mag(), hmin(hmax(q.x, hmax(q.y, q.z)), __int2half_rd(0)));
		result.col = col;
		return result;
	}
	__device__
	devVector3 GetNormal(devVector3 surfacePos) {
		
		return surfacePos.Sub(trans.pos).normalised();

	}
	__device__
		devDistReturn EstimatedDistance(devVector3 currPos) {
		devDistReturn result;

		__half rad = trans.sca.mag();
		result.dist = __hsub(trans.pos.Dist(currPos), rad);
		return result;
	}
};
class devHollowCube : public devShape {
public:
	__half thickness;
	__device__ __host__
	devHollowCube(HollowCube orig) : devShape(orig) {
		thickness = orig.thickness;
	}
	__device__
		devDistReturn DistanceTo(devVector3 currPos) {
		devVector3 delta = TransformPoint(currPos);
		devVector3 p = delta.abs().Sub(trans.sca);
		devVector3 q = p;
		devDistReturn result;
		q.x = __hadd(q.x, thickness);
		q.y = __hadd(q.y, thickness);
		q.z = __hadd(q.z, thickness);
		q = q.abs();
		q.x = __hsub(q.x, thickness);
		q.y = __hsub(q.y, thickness);
		q.z = __hsub(q.z, thickness);

		float a = devVector3(p.x, q.y, q.z).Max(devVector3(0, 0, 0)).mag() + hmin(hmax(p.x, hmax(q.y, q.z)), 0);
		float b = devVector3(q.x, p.y, q.z).Max(devVector3(0, 0, 0)).mag() + hmin(hmax(q.x, hmax(p.y, q.z)), 0);
		float c = devVector3(q.x, q.y, p.z).Max(devVector3(0, 0, 0)).mag() + hmin(hmax(q.x, hmax(q.y, p.z)), 0);

		result.dist = hmin(hmin(a, b), c);
		result.col = col;
		return result;
	}
	__device__
		devVector3 GetNormal(devVector3 surfacePos) {
		return surfacePos.Sub(trans.pos).normalised();
	}
	__device__
		devDistReturn EstimatedDistance(devVector3 currPos) {
		devDistReturn result;
		__half rad = trans.sca.mag();
		result.dist = __hsub(trans.pos.Dist(currPos), rad);
		return result;
	}
};

class devPlane : public devShape {
public:
	__device__ __host__
		devPlane(Plane orig) : devShape(orig) {}
	__device__
		devDistReturn DistanceTo(devVector3 currPos) {
		devDistReturn result;
		result.dist = __hsub(currPos.y, trans.pos.y);
		result.col = col;
		return result;
	}
	__device__
		devVector3 GetNormal(devVector3 surfacePos) {
		return devVector3(0, 1, 0);
	}
};
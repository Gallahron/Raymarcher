
#include "vectors.cu"
#include "holder.cu"
#include <stdio.h>
#include <SDL.h>

#pragma once

struct DistReturn {
	Vector3 col;
	float dist;
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
		Shape(Vector3 newPos, Vector3 newRot, char typeof, int blended) {
		trans.pos = newPos;
		trans.rot = newRot;
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

	__device__ __host__
		Vector3 TransformPoint(Vector3 currpos) {
		return currpos.Sub(trans.pos).ApplyRot(trans.rot.Negative());
	}
};

class Sphere : public Shape {
public:
	__device__ __host__
		Sphere(float x, float y, float z, float rad, int blended) : Shape(x, y, z, 0, 0, 0, 's', blended) {
		trans.sca = Vector3{ rad, rad, rad };
	}
	__device__ __host__
		Sphere(Vector3 newPos, float rad, int blended) : Shape(newPos, Vector3(0, 0, 0), 's', blended) {
		trans.sca = Vector3{ rad, rad, rad };
	}
	__device__ __host__
		DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = trans.pos.Dist(currPos) - trans.sca.x;
		result.col = col;
		return result;
	}
	__host__
	static Shape* Add(ShapeHolder* holder, Vector3 pos, float rad, int blended);
};

class Cube : public Shape {
public:
	__device__ __host__
		Cube(float x, float y, float z, float rx, float ry, float rz, float bx, float by, float bz, int blended) : Shape(x, y, z, rx, ry, rz, 'c', blended) {
		trans.sca = Vector3(bx, by, bz);
	}
	__device__ __host__
		Cube(Vector3 newPos, Vector3 newRot, Vector3 bound, int blended) : Shape(newPos, newRot, 'c', blended) {
		trans.sca = bound;
	}
	__device__ __host__
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
	
	__host__
		static Shape* Add(ShapeHolder* holder, Vector3 pos, Vector3 rot, Vector3 bounds, int blended);
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
		HollowCube(Vector3 newPos, Vector3 newRot, Vector3 bound, float e, int blended) : Shape(newPos, newRot, 'h', blended) {
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
	__host__
	static Shape* Add(ShapeHolder* holder,Vector3 pos, Vector3 rot, Vector3 bounds, float thickness, int blended);
	
};

class Plane : public Shape {
public:
	__device__ __host__
		Plane(Vector3 newPos, int blended) : Shape(newPos, VZERO, 'p', blended) { }
	__device__ __host__
		Plane(float height, int blended) : Shape(Vector3(0, height, 0), VZERO, 'p', blended) {}
	__device__
		DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = currPos.y - trans.pos.y;
		result.col = col;
		return result;
	}
	__host__
	static Shape* Add(ShapeHolder* holder, float height, int blended);
};
class Octahedron : public Shape {
public:
	__device__ __host__
		Octahedron(float x, float y, float z, float rx, float ry, float rz, float s, int blended) : Shape(x, y, z, rx, ry, rz, 'o', blended) {
		trans.sca = Vector3(s, s, s);
	}
	__device__ __host__
		Octahedron(Vector3 newPos, Vector3 newRot, float s, int blended) : Shape(newPos, newRot, 'o', blended) {
		trans.sca = Vector3(s, s, s);
	}
	__device__ __host__
	DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = 0;

		Vector3 delta = TransformPoint(currPos);
		Vector3 p = delta.abs();
		float m = p.x + p.y + p.z - trans.sca.x;
		Vector3 q;
		if (3.0f * p.x < m) q = p;
		else if (3.0f * p.y < m) q = Vector3(p.y, p.z, p.x);
		else if (3.0f * p.z < m) q = Vector3(p.z, p.x, p.y);
		else {
			result.dist = m * 0.57735027f;
		}
		if (result.dist == 0) {
			float k = fmaxf(0.0, fminf(trans.sca.x, 0.5f * (q.z - q.y + trans.sca.x)));
			result.dist = Vector3(q.x, q.y - trans.sca.x + k, q.z - k).mag();
		}
		result.col = col;
		return result;
	}

	__host__
		static Shape* Add(ShapeHolder* holder, Vector3 pos, Vector3 rot, float s, int blended);
};


__host__
Shape* Sphere::Add(ShapeHolder* holder, Vector3 pos, float rad, int blended) {
	Sphere* ptr;
	cudaMallocManaged(&ptr, sizeof(Sphere));
	*ptr = Sphere(pos, rad, blended);
	holder->AddElement(ptr);
	return ptr;
}
__host__
Shape* Cube::Add(ShapeHolder* holder, Vector3 pos, Vector3 rot, Vector3 bounds, int blended) {
	Cube* ptr;
	cudaMallocManaged(&ptr, sizeof(Cube));
	*ptr = Cube(pos, rot, bounds, blended);
	holder->AddElement(ptr);
	return ptr;
}
__host__
Shape* HollowCube::Add(ShapeHolder* holder,Vector3 pos, Vector3 rot, Vector3 bounds, float thickness, int blended) {
	HollowCube* ptr;
	cudaMallocManaged(&ptr, sizeof(HollowCube));
	*ptr = HollowCube(pos, rot, bounds, thickness, blended);
	holder->AddElement(ptr);
	return ptr;
}
__host__
Shape* Plane::Add(ShapeHolder* holder, float height, int blended) {
	Plane* ptr;
	cudaMallocManaged(&ptr, sizeof(Plane));
	*ptr = Plane(height, blended);
	holder->AddElement(ptr);
	return ptr;
}
__host__
Shape* Octahedron::Add(ShapeHolder* holder, Vector3 pos, Vector3 rot, float s, int blended) {
	Octahedron* ptr;
	cudaMallocManaged(&ptr, sizeof(Octahedron));
	*ptr = Octahedron(pos, rot, s, blended);
	holder->AddElement(ptr);
	return ptr;
}
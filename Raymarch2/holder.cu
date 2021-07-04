#include "lights.cu"
#pragma once

class Shape;

class AppendableArray {
public:
	void** values;

	int length;
	__host__
	void AddElement(void* ptr) {
		void** newVals;
		cudaMallocManaged(&newVals, sizeof(Uint32) * (length + 1));
		for (int i = 0; i < length; i++) {
			newVals[i] = values[i];
		}
		newVals[length] = ptr;
		void** stor = values;
		values = newVals;
		cudaFree(stor);
		length++;
	}
	__host__ __device__
	void* GetElement(int index) {
		return values[index];
	}
};

class ShapeHolder : public AppendableArray {
public:
	__host__ __device__
	Shape* GetShape(int index) {
		return (Shape*)values[index];
	}
};


class LightHolder : public AppendableArray {
public:
	__host__
	void AddLight(Vector3 pos, float intensity) {
		Light* ptr;
		cudaMallocManaged(&ptr, sizeof(Light));
		ptr->pos = pos;
		ptr->intensity = intensity;
		AddElement(ptr);
	}
	__host__ __device__
	Light* GetLight(int index) {
		return (Light*)values[index];
	}
};

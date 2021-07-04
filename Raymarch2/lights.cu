#include "vectors.cu"
#pragma once

class Light {
public:
	Vector3 pos;
	float intensity;
	Light(Vector3 newPos) {
		pos = newPos;
	}
};

#include "vectors.cu"
#include <stdio.h>
#include <SDL.h>
struct Ray {
	Vector3 pos;
	Vector3 dir;
	Vector3 col;
	float dist;
	float depth;
	int steps = 0;
	int hit = 0;
};
class ShapeHolder {
public:
	Shape** values;
	int length;
	__host__
		void AddShape(Shape* ptr) {
		Shape** newVals;
		cudaMallocManaged(&newVals, sizeof(Uint32) * (length + 1));
		for (int i = 0; i < length; i++) {
			newVals[i] = values[i];
		}
		newVals[length] = ptr;
		Shape** stor = values;
		values = newVals;
		cudaFree(stor);
		length++;
	}
	__host__
		Shape* CreateSphere(Vector3 pos, float rad, int blended) {
		Sphere* ptr;
		cudaMallocManaged(&ptr, sizeof(Sphere));
		*ptr = Sphere(pos, 0.6f, blended);
		AddShape(ptr);
		return ptr;
	}
	__host__
		Shape* CreatePlane(float height, int blended) {
		Plane* ptr;
		cudaMallocManaged(&ptr, sizeof(Plane));
		*ptr = Plane(height, blended);
		AddShape(ptr);
		return ptr;
	}
	__host__
		Shape* CreateCube(Vector3 pos, Vector3 rot, Vector3 bounds, int blended) {
		Cube* ptr;
		cudaMallocManaged(&ptr, sizeof(Cube));
		*ptr = Cube(pos, rot, bounds, blended);

		AddShape(ptr);
		return ptr;
	}
	__host__
		Shape* CreateHollowCube(Vector3 pos, Vector3 rot, Vector3 bounds, float thickness, int blended) {
		HollowCube* ptr;
		cudaMallocManaged(&ptr, sizeof(Cube));
		*ptr = HollowCube(pos, rot, bounds, thickness, blended);

		AddShape(ptr);
		return ptr;
	}
};
/*struct ShapeHolder {
	Shape** values;
	int length;
};*/


__device__
Uint32 colorMap(Uint32 r, Uint32 g, Uint32 b) {
	r <<= 16;
	g <<= 8;
	return r + g + b;
}

__device__
DistReturn smoothDist(DistReturn distA, DistReturn distB, float k) {
	DistReturn result;
	float h = __saturatef(0.5f + 0.5f * (distB.dist - distA.dist) / k);

	result.col = distB.col.Lerp(distA.col, h);
	result.dist = distB.dist * (1 - h) + distA.dist * h - k * h * (1.0f - h);
	return result;
}

__device__
DistReturn regDist(DistReturn distA, DistReturn distB) {
	DistReturn result;
	if (distA.dist < distB.dist) {
		result.col = distA.col;
		result.dist = distA.dist;
	}
	else {
		result.col = distB.col;
		result.dist = distB.dist;
	}
	return result;
}
__device__
DistReturn distToScene(Vector3 pos) {
	DistReturn dist;
	dist.dist = 9999;
	dist.col = Vector3(1, 1, 1);
	for (int i = 0; i < shapes.length; i++) {
		Shape* ptr = shapes.values[i];
		DistReturn newDist;
		switch (ptr->type) {
		case ('s'):
			newDist = ((Sphere*)ptr)->DistanceTo(pos);
			break;
		case ('c'):
			newDist = ((Cube*)ptr)->EstimatedDistance(pos);
			if (smoothDist(dist, newDist, 0.2f).dist < dist.dist) newDist = ((Cube*)ptr)->DistanceTo(pos);
			else newDist.dist = 9999;
			break;
		case ('p'):
			newDist = ((Plane*)ptr)->DistanceTo(pos);
			break;
		case ('h'):
			newDist = ((HollowCube*)ptr)->EstimatedDistance(pos);
			if (smoothDist(dist, newDist, 0.2f).dist < dist.dist) newDist = ((HollowCube*)ptr)->DistanceTo(pos);
			else newDist.dist = 9999;
			break;
		}
		if (ptr->blend) dist = smoothDist(dist, newDist, 0.2f);//fminf(dist.dist, newDist.dist);//smoothDist(dist, newDist, 2);
		else dist = regDist(dist, newDist);
	}
	return dist;
}

__device__
Vector3 getNormal(float dist, Vector3 pos) {
	Vector3 result;
	Vector3 offsets[] = {
		Vector3(0.01f,0,0),
		Vector3(0, 0.01f, 0),
		Vector3(0, 0, 0.01f)
	};
	result.x = dist - distToScene(pos.Sub(offsets[0])).dist;
	result.y = dist - distToScene(pos.Sub(offsets[1])).dist;
	result.z = dist - distToScene(pos.Sub(offsets[2])).dist;
	return result.normalised();
}

__device__
Ray genRay(Vector3 currPos, Vector3 currRot, float x, float y, Vector3* bounds, int debug) {
	struct Ray ray;
	ray.pos = currPos;
	/*ray.dir.x = 0;
	ray.dir.y = 0;
	ray.dir.z = 1;
	x /= SCREEN_WIDTH;
	x -= 0.5f;
	y /= SCREEN_WIDTH;
	y -= 0.5f;

	ray.dir = ray.dir.RotX(y * viewangle + currRot.x);
	ray.dir = ray.dir.RotY(x * viewangle + currRot.y);*/

	x /= SCREEN_WIDTH;
	x -= 0.5f;
	x *= bounds[0].x;
	y /= SCREEN_HEIGHT;
	y -= 0.5f;
	y *= bounds[1].y;
	ray.dir.x = x;
	ray.dir.y = y;
	ray.dir.z = 1;
	ray.dir = ray.dir.normalised();
	ray.dir = ray.dir.RotX(currRot.x);
	ray.dir = ray.dir.RotY(currRot.y);

	return ray;
}

__device__
Ray rayMarch(Ray ray, int debug, float threshold) {
	float disttravelled;
	DistReturn dist;
	dist = distToScene(ray.pos);
	disttravelled = 0;
	while (disttravelled < MAX_DIST && dist.dist > threshold) {
		ray.pos = ray.pos.Add(ray.dir.Mul(dist.dist));

		disttravelled += dist.dist;
		dist = distToScene(ray.pos);
		ray.steps++;
	}
	if (dist.dist <= MIN_DIST) {
		ray.hit = 1;
	}
	ray.dist = dist.dist;
	ray.col = dist.col;
	ray.depth = disttravelled;
	return ray;
}

__device__
float CalculateLighting(Light* lights, Vector3 position, int debug) {
	int noLights = *(&lights + 1) - lights;
	float lightVal = 0;

	for (int i = 0; i < 1; i++) {
		Ray ray;
		float dist = distToScene(position).dist;
		ray.pos = position.Add(getNormal(dist, position).Mul(0.05f));
		ray.dir = lights[i].pos.Sub(position).normalised();
		ray = rayMarch(ray, 0, MIN_DIST);
		if (!ray.hit) {
			lightVal += ray.dir.normalised().Dot(getNormal(dist, position));
		}
		else lightVal = 0;
	}
	if (lightVal < 0.0f) lightVal = 0.0f;

	return lightVal;
}

__global__
void ColourCalc(Uint32* pixels, Vector3 currPos, Vector3 currRot, Light* lights, Vector3* bounds) {

	//Blockid is vertical, threadid is horizontal
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int debug = 0;

	Ray ray = genRay(currPos, currRot, index % SCREEN_WIDTH, index / SCREEN_WIDTH, bounds, debug);

	/*ray.dir = ray.dir.RotX(currRot.x);
	ray.dir = ray.dir.RotY(currRot.y);*/
	ray = rayMarch(ray, debug, MIN_DIST);
	float lightVal = 0;
	if (ray.hit) lightVal = CalculateLighting(lights, ray.pos, debug);
	else lightVal = 1;
	//printf("Colour: %f, %f, %f\n", ray.col.x, ray.col.y, ray.col.z);
	unsigned int r, g, b;
	r = lightVal * ray.col.x;
	g = lightVal * ray.col.y;
	b = lightVal * ray.col.z;


	if (!ray.hit) {
		r = 15;
		g = 11;
		b = 21;
	}

	pixels[index] = colorMap(r, g, b);
}
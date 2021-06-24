
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <SDL.h>

#define SCREEN_WIDTH 1000
#define SCREEN_HEIGHT 600
#define viewangle 1.0472f
#define MAX_DIST 100
#define MIN_DIST 0.01f

#define MOVE_SPEED 3.0f
#define M_SENS 0.0005f

#define PI 3.141593f

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
		return sqrt(x * x + y * y + z * z);
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
		float cosThet = cos(angle);
		float sinThet = sin(angle);

		result.x = x;
		result.y = y * cosThet - z * sinThet;
		result.z = y * sinThet + z * cosThet;

		return result;
	}
	__device__ __host__
		Vector3 RotY(float angle) {
		struct Vector3 result;
		float cosThet = cos(angle);
		float sinThet = sin(angle);
		result.x = x * cosThet + z * sinThet;
		result.y = y;
		result.z = -x * sinThet + z * cosThet;
		return result;
	}
	__device__ __host__
	Vector3 RotZ(float angle) {
		struct Vector3 result;
		float cosThet = cos(angle);
		float sinThet = sin(angle);
		result.x = x * cosThet - y * sinThet;
		result.y = x * sinThet + y * cosThet;
		result.z = z;
		return result;
	}
	__device__ __host__
		float Dot(Vector3 b) {
		return x * b.x + y * b.y + z * b.z;
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
		result.x = x + (b.x-x) * s;
		result.y = y + (b.y-y) * s;
		result.z = z + (b.z-z) * s;
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


struct DistReturn {
	Vector3 col;
	float dist;
};


class Light {
public:
	Vector3 pos;
	float intensity;
	Light(Vector3 newPos) {
		pos = newPos;
	}
};

class Shape {
public:
	char type;
	Vector3 pos;
	Vector3 rot;
	Vector3 col;
	int blend;

	__device__ __host__
	Shape(float x, float y, float z, float rx, float ry, float rz, char typeof, int blended) {
		pos.x = x;
		pos.y = y;
		pos.z = z;

		rot.x = rx;
		rot.y = ry;
		rot.z = rz;

		type = typeof;
		blend = blended;
	}
	__device__ __host__
	Shape(Vector3 newpos, Vector3 newrot, char typeof, int blended) {
		pos = newpos;
		rot = newrot;

		type = typeof;
		blend = blended;
	}
	__device__
	DistReturn DistanceTo(Vector3 currPos) { 
		DistReturn result;
		result.col = col;
		result.dist = 1;
		return ;
	};
	__device__
	DistReturn EstimatedDistance(Vector3 currPos) {
		return DistanceTo(currPos);
	}

	__device__
	Vector3 TransformPoint(Vector3 currpos) {
		return currpos.Sub(pos).ApplyRot(rot.Negative());
	}

	__device__
	virtual Vector3 GetNormal(Vector3 surfacePos) { return Vector3(0, 0, 0); };
};

class Sphere : public Shape {
public:
	float radius;
	__device__ __host__
		Sphere(float x, float y, float z, float rad, int blended) : Shape(x, y, z, 0, 0, 0, 's', blended) {
		radius = rad;
	}
	__device__ __host__
		Sphere(Vector3 newpos, float rad, int blended) : Shape(newpos, Vector3(0,0,0), 's', blended) {
		radius = rad;
	}
	__device__
	DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = pos.Dist(currPos) - radius;
		result.col = col;
		return result;
	}
	__device__
	Vector3 GetNormal(Vector3 surfacePos) {
		return surfacePos.Sub(pos).normalised();
	}
};

class Cube : public Shape {
public:
	Vector3 bounds;
	__device__ __host__
	Cube(float x, float y, float z, float rx, float ry, float rz, float bx, float by, float bz, int blended) : Shape(x, y, z, rx, ry, rz, 'c', blended) {
		bounds = Vector3(bx,by,bz);
	}
	__device__ __host__
	Cube(Vector3 newpos, Vector3 newrot, Vector3 bound, int blended) : Shape(newpos, newrot, 'c', blended) {
		bounds = bound;
	}
	__device__
	DistReturn DistanceTo(Vector3 currPos) {
		Vector3 delta = TransformPoint(currPos);
		Vector3 q = delta.abs().Sub(bounds);
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
		return surfacePos.Sub(pos).normalised();
	}
	__device__
	DistReturn EstimatedDistance(Vector3 currPos) {
		DistReturn result;
		float rad = bounds.mag();
		result.dist = pos.Dist(currPos) - rad;
		return result;
	}
};
class HollowCube : public Shape {
public:
	Vector3 bounds;
	float thickness;
	__device__ __host__
		HollowCube(float x, float y, float z, float rx, float ry, float rz, float bx, float by, float bz, float e, int blended) : Shape(x, y, z, rx, ry, rz, 'h', blended) {
		bounds = Vector3(bx, by, bz);
		thickness = e;
	}
	__device__ __host__
		HollowCube(Vector3 newpos, Vector3 newrot, Vector3 bound, float e, int blended) : Shape(newpos, newrot, 'h', blended) {
		bounds = bound;
		thickness = e;
	}
	__device__
	DistReturn DistanceTo(Vector3 currPos) {
		Vector3 delta = TransformPoint(currPos);
		Vector3 p = delta.abs().Sub(bounds);
		Vector3 q = p;
		DistReturn result;
		q.x += thickness;
		q.y += thickness;
		q.z += thickness;
		q = q.abs();
		q.x -= thickness;
		q.y -= thickness;
		q.z -= thickness;
		
		float res;
		float a = Vector3(p.x, q.y, q.z).Max(Vector3(0, 0, 0)).mag() + fminf(fmaxf(p.x, fmaxf(q.y, q.z)), 0);
		float b = Vector3(q.x, p.y, q.z).Max(Vector3(0, 0, 0)).mag() + fminf(fmaxf(q.x, fmaxf(p.y, q.z)), 0);
		float c = Vector3(q.x, q.y, p.z).Max(Vector3(0, 0, 0)).mag() + fminf(fmaxf(q.x, fmaxf(q.y, p.z)), 0);

		result.dist = fminf(fminf(a,b),c);
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
		return surfacePos.Sub(pos).normalised();
	}
	__device__
	DistReturn EstimatedDistance(Vector3 currPos) {
		DistReturn result;
		float rad = bounds.mag();
		result.dist = pos.Dist(currPos) - rad;
		return result;
	}
};

class Plane : public Shape {
public:
	__device__ __host__
	Plane(Vector3 newpos, int blended) : Shape(newpos, Vector3(0,0,0), 'p', blended) { }
	__device__ __host__
	Plane(float height, int blended) : Shape(Vector3(0, height, 0), Vector3(0,0,0), 'p', blended) {}
	__device__
	DistReturn DistanceTo(Vector3 currPos) {
		DistReturn result;
		result.dist = currPos.y - pos.y;
		result.col = col;
		return result;
	}
	__device__
		Vector3 GetNormal(Vector3 surfacePos) {
		return Vector3(0, 1, 0);
	}
};


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

//Global
__managed__ ShapeHolder shapes;

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
Ray genRay(Vector3 currPos, Vector3 currRot,float x, float y, Vector3* bounds, int debug) {
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




int main(int argc, char** argv)
{

	//The window we'll be rendering to
	SDL_Window* window = NULL;

	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;

	Vector3 currPos = Vector3(0, 0, -10);
	Vector3 currRot = Vector3(0, 0, 0);


	Sphere* spherea = (Sphere*)shapes.CreateSphere(Vector3(0, 0, 0),0.6f, 1);
	Sphere* sphereb = (Sphere*)shapes.CreateSphere(Vector3(0, 3, 0), 0.6f, 1);
	Plane* plane = (Plane*)shapes.CreatePlane(-2.3f, 0);
	Cube* cube = (Cube*)shapes.CreateCube(Vector3(0, -2.3f, 0), Vector3(0, 0, 0), Vector3(3.0f, 0.01f, 100.0f), 0);
	//HollowCube* bounding = (HollowCube*)shapes.CreateHollowCube(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(1, 1, 1), 0.05f);*/

	spherea->col = Vector3(187, 134, 252);
	sphereb->col = Vector3(187, 134, 252);
	plane->col = Vector3(18, 18, 18);
	cube->col = Vector3(50, 50, 50);
	//bounding->col = Vector3(0, 1, 1);*/


	Vector3 veloc = Vector3(0, 0, 0);
	float deltaTime;

	Vector3* bounds;
	cudaMallocManaged(&bounds, 2 * sizeof(Vector3));

	float yViewAngle = SCREEN_HEIGHT / (float)SCREEN_WIDTH * viewangle;
	bounds[0] = Vector3(0, 0, 1).RotY(viewangle / 2);
	bounds[1] = Vector3(0, 0, 1).RotX(yViewAngle / 2);
	

	Light* lights;
	cudaMallocManaged(&lights, 1 * sizeof(Light));
	lights[0] = Light(Vector3(0, 5, 0));

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	void* pixelBuffer;

	cudaMallocManaged(&pixelBuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));
	//Initialize SDL
	if (SDL_Init(SDL_INIT_VIDEO) < 0)
	{
		printf("SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
	}
	else
	{
		//Create window
		window = SDL_CreateWindow("Raymarch", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
		if (window == NULL)
		{
			printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
		}
		else
		{
			//Get window surface
			screenSurface = SDL_GetWindowSurface(window);
			SDL_SetRelativeMouseMode(SDL_TRUE);

			


			int time = SDL_GetTicks();

			//Main loop flag
			bool quit = false;

			//Event handler
			SDL_Event e;
			//While application is running
			while (!quit)
			{
				//Handle events on queue
				while (SDL_PollEvent(&e) != 0)
				{
					switch (e.type) {
					case (SDL_MOUSEMOTION):
						currRot.x += e.motion.yrel*M_SENS;
						currRot.y += e.motion.xrel*M_SENS;
						break;
					case (SDL_KEYDOWN):
						switch (e.key.keysym.sym) {
							case SDLK_w:
								veloc.z = MOVE_SPEED;
								break;

							case SDLK_s:
								veloc.z = -MOVE_SPEED;
								break;

							case SDLK_a:
								veloc.x = -MOVE_SPEED;
								break;

							case SDLK_d:
								veloc.x = MOVE_SPEED;
								break;

							case SDLK_e:
								veloc.y = MOVE_SPEED;
								break;

							case SDLK_q:
								veloc.y = -MOVE_SPEED;
								break;

							case SDLK_ESCAPE:
								quit = true;
								break;
						}
						break;
					case (SDL_KEYUP):
						switch (e.key.keysym.sym) {
							case SDLK_w:
								veloc.z = 0;
								break;
							case SDLK_s:
								veloc.z = 0;
								break;

							case SDLK_a:
								veloc.x = 0;
								break;

							case SDLK_d:
								veloc.x = 0;
								break;

							case SDLK_e:
								veloc.y = 0;
								break;

							case SDLK_q:
								veloc.y = 0;
								break;
						}
						break;
					}
					//User requests quit
					if (e.type == SDL_QUIT)
					{
						quit = true;
					}
				}

				SDL_LockSurface(screenSurface);

				int blocks = SCREEN_HEIGHT * SCREEN_WIDTH / 1024;
				
				
				ColourCalc <<<blocks, 1024>>> ((Uint32*)pixelBuffer, currPos, currRot, lights, bounds);
				
				cudaMemcpyAsync(screenSurface->pixels, pixelBuffer, SCREEN_WIDTH* SCREEN_HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost, stream1); //(cudaStream_t)1);
				cudaDeviceSynchronize();
				
				/*sphere->pos.x = 20;//cos(SDL_GetTicks() / 1000.0f);
				sphere->pos.y = sin(SDL_GetTicks() / 1000.0f);
				cube->pos.x = 5*sin(SDL_GetTicks() / 1000.0f);
				cube->pos.z = 5*cos(SDL_GetTicks() / 1000.0f);
				cube->rot.y = SDL_GetTicks() / 100.0f;*/
				spherea->pos.x = 1*cos(SDL_GetTicks() / 10000.0f+1.58f);
				sphereb->pos.y = 1*cos(SDL_GetTicks() / 10000.0f+1.58f);

				

				currPos = currPos.Add(veloc.ApplyRot(currRot).Mul(deltaTime));
				lights[0] = currPos.Add(Vector3(0, 10, 0));

				deltaTime = (SDL_GetTicks() - time) / 1000.0f;
				printf("Time for frame: %ums\n", SDL_GetTicks() - time);
				time = SDL_GetTicks();

				//SDL_memcpy(screenSurface->pixels, pixels, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32));

				SDL_UnlockSurface(screenSurface);

				//Update the surface
				SDL_UpdateWindowSurface(window);
				

				
				
			}
		}
	}
	//Destroy window
	SDL_DestroyWindow(window);

	//Quit SDL subsystems
	SDL_Quit();

	return 0;
}
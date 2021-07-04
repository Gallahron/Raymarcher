#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SDL.h>

#include "shapes.cu"
#include "vectors.cu"
#include "holder.cu"
#include "distances.cu"
#include "scene.cu"

#define PI 3.141593f
#define SKIP_PIXELS 3

struct Ray {
	Vector3 pos;
	Vector3 dir;
	Vector3 col;
	float dist;
	float depth;
	int steps = 0;
	int hit = 0;
};

struct RenderInfo {
	int SCREEN_WIDTH;
	int SCREEN_HEIGHT;
	float viewangle;
	float MAX_DIST;
	float MIN_DIST;
	Vector3* bounds;
};

class Renderer {
public:
	RenderInfo* info;

	SDL_Window* window;
	SDL_Surface* screenSurface;
	__host__
	Renderer(int xres, int yres) {

		cudaMallocManaged(&info, sizeof(RenderInfo));
		info->SCREEN_WIDTH = xres;
		info->SCREEN_HEIGHT = yres;
		info->viewangle = 1.5704f;
		info->MAX_DIST = 100;
		info->MIN_DIST = 0.01f;

		window = SDL_CreateWindow("Raymarch", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, xres, yres, SDL_WINDOW_SHOWN);

		screenSurface = SDL_GetWindowSurface(window);

		//SDL_SetWindowPosition(window, 72, 100);
		SDL_SetRelativeMouseMode(SDL_TRUE);

		cudaMallocManaged(&info->bounds, 2 * sizeof(Vector3));

		float yViewAngle = yres / (float)xres * info->viewangle;
		info->bounds[0] = Vector3(0, 0, 1).RotY(info->viewangle / 2);
		info->bounds[1] = Vector3(0, 0, 1).RotX(yViewAngle / 2);

		cudaStreamCreate(&stream1);

		cudaMallocManaged(&pixelBuffer, xres * yres * sizeof(Uint32));
	}
	__host__
		void Draw(Transform camera, Scene scene);

	__host__
	void Quit() {
		//Destroy window
		SDL_DestroyWindow(window);

		//Quit SDL
		SDL_Quit();
	}

private:
	cudaStream_t stream1;
	void* pixelBuffer;
};

__device__
Uint32 colorMap(Uint32 r, Uint32 g, Uint32 b) {
	r <<= 16;
	g <<= 8;
	return r + g + b;
}


__device__
Ray genRay(Transform camera, float x, float y, RenderInfo* info, int debug) {
	struct Ray ray;
	ray.pos = camera.pos;

	x /= info->SCREEN_WIDTH;
	x -= 0.5f;
	x *= info->bounds[0].x;
	y /= info->SCREEN_HEIGHT;
	y -= 0.5f;
	y *= info->bounds[1].y;
	ray.dir.x = x;
	ray.dir.y = y;
	ray.dir.z = 1;
	ray.dir = ray.dir.normalised();
	ray.dir = ray.dir.RotX(camera.rot.x);
	ray.dir = ray.dir.RotY(camera.rot.y);

	return ray;
}

__device__
Ray rayMarch(Ray ray, Scene scene, RenderInfo* info, int maxDist, int debug) {
	float disttravelled;
	DistReturn dist;
	dist = scene.distToScene(ray.pos);
	disttravelled = 0;
	while (disttravelled < maxDist && dist.dist > info->MIN_DIST) {
		ray.pos = ray.pos.Add(ray.dir.Mul(dist.dist));
		dist = scene.distToScene(ray.pos);
		disttravelled += dist.dist;
		ray.steps++;
	}
	if (dist.dist <= info->MIN_DIST) {
		ray.hit = 1;
	}
	ray.dist = dist.dist;
	ray.col = dist.col;
	ray.depth = disttravelled;
	return ray;
}

__device__
float CalculateLighting(Scene scene, Vector3 position, RenderInfo* info, int debug) {
	//int noLights = *(&lights + 1) - lights;
	float lightVal = 0;

	for (int i = 0; i < scene.lights.length; i++) {
		Ray ray;
		float dist = scene.distToScene(position).dist;
		Vector3 normal = scene.getNormal(dist, position);
		ray.pos = position.Add(normal.Mul(0.05f));
		ray.dir = scene.lights.GetLight(i)->pos.Sub(position).normalised();
		ray = rayMarch(ray, scene, info, scene.lights.GetLight(i)->pos.Dist(ray.pos), 0);
		if (!ray.hit) {
			lightVal += ray.dir.normalised().Dot(normal);
		}
	}
	return __saturatef(lightVal);
}

__global__
void ColourCalc(Uint32* pixels, Transform trans, Scene scene, RenderInfo* info) {
	//Blockid is vertical, threadid is horizontal
	int index = (blockIdx.x * blockDim.x + threadIdx.x)*SKIP_PIXELS;

	int debug = 0;

	Ray ray = genRay(trans, index % info->SCREEN_WIDTH, index / info->SCREEN_WIDTH, info, debug);

	/*ray.dir = ray.dir.RotX(currRot.x);
	ray.dir = ray.dir.RotY(currRot.y);*/
	ray = rayMarch(ray, scene, info, info->MAX_DIST, debug);
	float lightVal = 0;
	if (ray.hit) lightVal = CalculateLighting(scene, ray.pos, info, debug);
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

void Renderer::Draw(Transform camera, Scene scene) {
		int noPixels = info->SCREEN_HEIGHT * info->SCREEN_WIDTH;
		int blocks = noPixels / 1024 / SKIP_PIXELS;

		
		SDL_LockSurface(screenSurface);

		ColourCalc << <blocks, 1024 >> > ((Uint32*)pixelBuffer, camera, scene, info);
		cudaMemcpyAsync(screenSurface->pixels, pixelBuffer, noPixels * sizeof(Uint32), cudaMemcpyDeviceToHost, stream1);

		cudaDeviceSynchronize();

		SDL_UnlockSurface(screenSurface);
		SDL_UpdateWindowSurface(window);
}
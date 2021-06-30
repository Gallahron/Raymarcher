
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "shapes.cu"
#include "vectors.cu"
#include "holder.cu"
#include "distances.cu"
#include <cuda_fp16.h>
//#include "engine.cu"

#include <stdio.h>
#include <SDL.h>

#define SCREEN_WIDTH 1777
#define SCREEN_HEIGHT 1000
#define viewangle 1.0472f
#define MAX_DIST 100
#define MIN_DIST 0.01f

#define MOVE_SPEED 3.0f
#define M_SENS 0.0005f

#define PI 3.141593f


struct Ray {
	Vector3 pos;
	Vector3 dir;
	Vector3 col;
	float dist;
	float depth;
	int steps = 0;
	int hit = 0;
};

struct devRay {
	devVector3 pos;
	devVector3 dir;
	devVector3 col;
	__half dist;
	__half depth;
	int steps = 0;
	int hit = 0;
};

/*struct ShapeHolder {
	Shape** values;
	int length;
};*/

class Scene {
public:
	ShapeHolder shapes;
	LightHolder lights;
};

class devScene {
public:
	devShapeHolder shapes;
	devLightHolder lights;

	devScene(Scene orig) {
		cudaMallocManaged(&lights.values, sizeof(Uint32) * orig.lights.length);
		lights.length = orig.lights.length;
		for (int i = 0; i < orig.lights.length; i++) {
			devLight* ptr;
			cudaMallocManaged(&ptr, sizeof(devLight));
			
			*ptr = devLight(*orig.lights.GetLight(i));
			
			lights.values[i] = ptr;
		}
		cudaMallocManaged(&shapes.values, sizeof(devShape) * orig.shapes.length);
		//printf("orig: %u\n",orig.shapes.length);
		shapes.length = orig.shapes.length;
		for (int i = 0; i < orig.shapes.length; i++) {
			devShape* ptr;
			Shape* curr = orig.shapes.GetShape(i);
			switch (curr->type) {
				case ('s'):
					cudaMallocManaged(&ptr, sizeof(devSphere));
					*ptr = devSphere(*(Sphere*)curr);
					break;
				case ('c'):
					cudaMallocManaged(&ptr, sizeof(devCube));
					*ptr = devCube(*(Cube*)curr);
					break;
				case ('p'):
					cudaMallocManaged(&ptr, sizeof(devPlane));
					*ptr = devPlane(*(Plane*)curr);
					break;
				case ('h'):
					cudaMallocManaged(&ptr, sizeof(devHollowCube));
					*ptr = devHollowCube(*(HollowCube*)curr);
					break;
			}
			shapes.values[i] = ptr;
		}
	}
	__device__
	devDistReturn distToScene(devVector3 pos) {
		devDistReturn dist;
		dist.dist = __int2half_rd(9999);
		dist.col = devVector3(1.0f, 1.0f, 1.0f);
		for (int i = 0; i < shapes.length; i++) {
			devShape* ptr = (devShape*)shapes.values[i];
			devDistReturn newDist;
			switch (ptr->type) {
			case ('s'):
				newDist = ((devSphere*)ptr)->DistanceTo(pos);
				break;
			case ('c'):
				newDist = ((devCube*)ptr)->DistanceTo(pos);
				//if (__hlt(smoothDist(dist, newDist, __float2half(0.2f)).dist, dist.dist)) newDist = ((devCube*)ptr)->DistanceTo(pos);
				//else newDist.dist = 9999;
				break;
			case ('p'):
				newDist = ((devPlane*)ptr)->DistanceTo(pos);
				break;
			case ('h'):
				newDist = ((devHollowCube*)ptr)->DistanceTo(pos);
				//if (__hlt(smoothDist(dist, newDist, __float2half(0.2f)).dist, dist.dist)) newDist = ((devHollowCube*)ptr)->DistanceTo(pos);
				//else newDist.dist = 9999;
				break;
			}
			if (ptr->blend) dist = smoothDist(dist, newDist, __float2half(0.2f));//fminf(dist.dist, newDist.dist);//smoothDist(dist, newDist, 2);
			else dist = regDist(dist, newDist);
		}
		return dist;
	}

	__device__
	devVector3 getNormal(__half dist, devVector3 pos) {
		devVector3 result;
		devVector3 offsets[] = {
			devVector3(0.01f,0,0),
			devVector3(0, 0.01f, 0),
			devVector3(0, 0, 0.01f)
		};
		result.x = __hsub(dist, distToScene(pos.Sub(offsets[0])).dist);
		result.y = __hsub(dist, distToScene(pos.Sub(offsets[1])).dist);
		result.z = __hsub(dist, distToScene(pos.Sub(offsets[2])).dist);
		return result.normalised();
	}
};

__device__
Uint32 colorMap(Uint32 r, Uint32 g, Uint32 b) {
	r <<= 16;
	g <<= 8;
	return r + g + b;
}


__device__
devRay genRay(devTransform trans, __half x, __half y, devVector3* bounds, int debug) {
	struct devRay ray;
	ray.pos = trans.pos;
	/*ray.dir.x = 0;
	ray.dir.y = 0;
	ray.dir.z = 1;
	x /= SCREEN_WIDTH;
	x -= 0.5f;
	y /= SCREEN_WIDTH;
	y -= 0.5f;

	ray.dir = ray.dir.RotX(y * viewangle + currRot.x);
	ray.dir = ray.dir.RotY(x * viewangle + currRot.y);*/

	x = __hdiv(x,__int2half_rd(SCREEN_WIDTH));
	x = __hsub(x,__float2half(0.5f));
	x = __hmul(x,bounds[0].x);
	y = __hdiv(y,__int2half_rd(SCREEN_HEIGHT));
	y = __hsub(y,__float2half(0.5f));
	y = __hmul(y,bounds[1].y);
	ray.dir.x = x;
	ray.dir.y = y;
	ray.dir.z = __int2half_rd(1);
	ray.dir = ray.dir.normalised();
	ray.dir = ray.dir.RotX(trans.rot.x);
	ray.dir = ray.dir.RotY(trans.rot.y);

	return ray;
}

__device__
devRay rayMarch(devRay ray, devScene scene, int debug, __half threshold) {
	__half disttravelled;
	devDistReturn dist;
	dist = scene.distToScene(ray.pos);
	disttravelled = __float2half(0);
	////printf("%f,", __half2float(dist.dist));
	while (__hlt(disttravelled, MAX_DIST) && __hgt(dist.dist, threshold)) {
		ray.pos = ray.pos.Add(ray.dir.Mul(dist.dist));

		disttravelled = __hadd(disttravelled, dist.dist);
		dist = scene.distToScene(ray.pos);
		ray.steps++;
	}
	if (__hle(dist.dist, MIN_DIST)) {
		ray.hit = 1;
	}
	ray.dist = dist.dist;
	ray.col = dist.col;
	ray.depth = disttravelled;
	return ray;
}

__device__
__half CalculateLighting(devScene scene, devVector3 position, int debug) {
	//int noLights = *(&lights + 1) - lights;
	__half lightVal = 0;

	for (int i = 0; i < scene.lights.length; i++) {
		devRay ray;
		__half dist = scene.distToScene(position).dist;
		ray.pos = position.Add(scene.getNormal(dist, position).Mul(0.05f));
		ray.dir = scene.lights.GetLight(i)->pos.Sub(position).normalised();
		ray = rayMarch(ray, scene, 0, __float2half(MIN_DIST));
		if (!ray.hit) {
			lightVal = __hadd(lightVal,ray.dir.normalised().Dot(scene.getNormal(dist, position)));
		}
	}
	if (__hlt(lightVal, __float2half(0.0f))) lightVal = __float2half(0.0f);

	return lightVal;
}

__global__
void ColourCalc(Uint32* pixels, devTransform trans, devScene scene, devVector3* bounds) {

	//Blockid is vertical, threadid is horizontal
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int debug = 0;
	//printf("g");
	devRay ray = genRay(trans, __int2half_rd(index % SCREEN_WIDTH), __int2half_rd(index / SCREEN_WIDTH), bounds, debug);

	/*ray.dir = ray.dir.RotX(currRot.x);
	ray.dir = ray.dir.RotY(currRot.y);*/
	ray = rayMarch(ray, scene, debug, __float2half(MIN_DIST));
	//printf("h");
	__half lightVal = __float2half(0);
	if (ray.hit) lightVal = CalculateLighting(scene, ray.pos, debug);
	else lightVal = __float2half(1);
	//printf("i");
	////printf("Colour: %f, %f, %f\n", ray.col.x, ray.col.y, ray.col.z);
	unsigned int r, g, b;
	r = __half2int_rd(__hmul(lightVal, ray.col.x));
	g = __half2int_rd(__hmul(lightVal, ray.col.y));
	b = __half2int_rd(__hmul(lightVal, ray.col.z));


	if (!ray.hit) {
		r = 15;
		g = 11;
		b = 21;
	}
	//printf("J");

	pixels[index] = colorMap(r, g, b);
	//printf("k");
}

void Draw(Uint32* pixels, Transform trans, Scene scene, Vector3* bounds) {
	devTransform newTrans = devTransform(trans);
	//printf("b");
	devScene deviceScene = devScene(scene);
	//printf("%u", deviceScene.shapes.length);
	//printf("c");

	for (int i = 0; i < deviceScene.shapes.length; i++) {
		//printf("%c\n", deviceScene.shapes.GetShape(i)->type);
	}
	int blocks = SCREEN_HEIGHT * SCREEN_WIDTH / 1024;
	devVector3* newBounds;
	//printf("1");
	cudaMallocManaged(&newBounds, sizeof(devVector3) * 2);
	newBounds[0] = devVector3(bounds[0]);
	newBounds[1] = devVector3(bounds[1]);
	//printf("2");
	ColourCalc <<<blocks, 1024>>> (pixels, newTrans, deviceScene, newBounds);
}

typedef struct Player {
	Transform trans;
	Vector3 veloc;
} Player;
int main(int argc, char** argv)
{
	//printf("a");
	//The window we'll be rendering to
	SDL_Window* window = NULL;

	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;

	Player player;
	player.trans.pos = Vector3(0, 0, -10);
	player.trans.rot = Vector3(0, 0, 0);

	Scene* scene;
	cudaMallocManaged(&scene, sizeof(Scene));


	Sphere* spherea = (Sphere*)scene->shapes.CreateSphere(Vector3(0, 0, 0),0.6f, 1);
	Sphere* sphereb = (Sphere*)scene->shapes.CreateSphere(Vector3(0, 3, 0), 0.6f, 1);
	//Sphere* spherec = (Sphere*)scene->shapes.CreateSphere(Vector3(0, 6, 0), 0.6f, 1);
	//Plane* plane = (Plane*)scene->shapes.CreatePlane(-2.3f, 0);
	//Cube* cube = (Cube*)scene->shapes.CreateCube(Vector3(0, -2.3f, 0), Vector3(0, 0, 0), Vector3(3.0f, 0.01f, 100.0f), 1);
	//HollowCube* bounding = (HollowCube*)shapes.CreateHollowCube(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(1, 1, 1), 0.05f);*/

	spherea->col = Vector3(187, 134, 252);
	//sphereb->col = Vector3(187, 134, 252);
	//spherec->col = Vector3(187, 134, 252);
	//plane->col = Vector3(18, 18, 18);
	//cube->col = Vector3(50, 50, 50);
	//bounding->col = Vector3(0, 1, 1);*/


	Vector3 veloc = Vector3(0, 0, 0);
	float deltaTime;

	Vector3* bounds;
	cudaMallocManaged(&bounds, 2 * sizeof(Vector3));

	float yViewAngle = SCREEN_HEIGHT / (float)SCREEN_WIDTH * viewangle;
	bounds[0] = Vector3(0, 0, 1).RotY(viewangle / 2);
	bounds[1] = Vector3(0, 0, 1).RotX(yViewAngle / 2);
	

	/*Light* lights;
	cudaMallocManaged(&lights, 1 * sizeof(Light));*/
	scene->lights.AddLight(Vector3(0, 5, 0), 1); //lights[0] = Light(Vector3(0, 5, 0));

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
		SDL_SetWindowPosition(window, 72, 0);
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
							player.trans.rot.x += e.motion.yrel*M_SENS;
							player.trans.rot.y += e.motion.xrel*M_SENS;
							break;
						case (SDL_KEYDOWN):
							switch (e.key.keysym.sym) {
								case SDLK_w:
									player.veloc.z = MOVE_SPEED;
									break;

								case SDLK_s:
									player.veloc.z = -MOVE_SPEED;
									break;

								case SDLK_a:
									player.veloc.x = -MOVE_SPEED;
									break;

								case SDLK_d:
									player.veloc.x = MOVE_SPEED;
									break;

								case SDLK_e:
									player.veloc.y = MOVE_SPEED;
									break;

								case SDLK_q:
									player.veloc.y = -MOVE_SPEED;
									break;

								case SDLK_ESCAPE:
									quit = true;
									break;
							}
							break;
						case (SDL_KEYUP):
							switch (e.key.keysym.sym) {
								case SDLK_w:
									player.veloc.z = 0;
									break;
								case SDLK_s:
									player.veloc.z = 0;
									break;

								case SDLK_a:
									player.veloc.x = 0;
									break;

								case SDLK_d:
									player.veloc.x = 0;
									break;

								case SDLK_e:
									player.veloc.y = 0;
									break;

								case SDLK_q:
									player.veloc.y = 0;
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
				
				Draw ((Uint32*)pixelBuffer, player.trans, *scene, bounds);
				cudaMemcpyAsync(screenSurface->pixels, pixelBuffer, SCREEN_WIDTH * SCREEN_HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost, stream1);
				cudaDeviceSynchronize();
				printf("DINGDINGDING!!!!");
				
				
				
				/*sphere->pos.x = 20;//cos(SDL_GetTicks() / 1000.0f);
				sphere->pos.y = sin(SDL_GetTicks() / 1000.0f);
				cube->pos.x = 5*sin(SDL_GetTicks() / 1000.0f);
				cube->pos.z = 5*cos(SDL_GetTicks() / 1000.0f);
				cube->rot.y = SDL_GetTicks() / 100.0f;*/
				//spherea->trans.pos.x = 1*cos(SDL_GetTicks() / 10000.0f+1.58f);
				//sphereb->trans.pos.y = 1*cos(SDL_GetTicks() / 10000.0f+1.58f);

				

				player.trans.pos = player.trans.pos.Add(player.veloc.ApplyRot(player.trans.rot).Mul(deltaTime));
				scene->lights.GetLight(0)->pos = player.trans.pos.Add(Vector3(0, 10, 0));

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
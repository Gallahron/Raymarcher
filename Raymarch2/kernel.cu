
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "shapes.cu"
#include "vectors.cu"
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
Ray genRay(Transform trans, float x, float y, Vector3* bounds, int debug) {
	struct Ray ray;
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
	ray.dir = ray.dir.RotX(trans.rot.x);
	ray.dir = ray.dir.RotY(trans.rot.y);

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
void ColourCalc(Uint32* pixels, Transform trans, Light* lights, Vector3* bounds) {

	//Blockid is vertical, threadid is horizontal
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int debug = 0;

	Ray ray = genRay(trans, index % SCREEN_WIDTH, index / SCREEN_WIDTH, bounds, debug);

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


typedef struct Player {
	Transform trans;
	Vector3 veloc;
} Player;
int main(int argc, char** argv)
{

	//The window we'll be rendering to
	SDL_Window* window = NULL;

	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;

	Player player;
	player.trans.pos = Vector3(0, 0, -10);
	player.trans.rot = Vector3(0, 0, 0);


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

	float3 a = make_float3(1.0f, 1.0f, 1.0f);
	float3 b = make_float3(1.0f, 1.0f, 1.0f);


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

				int blocks = SCREEN_HEIGHT * SCREEN_WIDTH / 1024;
				
				
				ColourCalc <<<blocks, 1024>>> ((Uint32*)pixelBuffer, player.trans, lights, bounds);
				
				cudaMemcpyAsync(screenSurface->pixels, pixelBuffer, SCREEN_WIDTH* SCREEN_HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost, stream1); //(cudaStream_t)1);
				cudaDeviceSynchronize();
				
				/*sphere->pos.x = 20;//cos(SDL_GetTicks() / 1000.0f);
				sphere->pos.y = sin(SDL_GetTicks() / 1000.0f);
				cube->pos.x = 5*sin(SDL_GetTicks() / 1000.0f);
				cube->pos.z = 5*cos(SDL_GetTicks() / 1000.0f);
				cube->rot.y = SDL_GetTicks() / 100.0f;*/
				spherea->trans.pos.x = 1*cos(SDL_GetTicks() / 10000.0f+1.58f);
				sphereb->trans.pos.y = 1*cos(SDL_GetTicks() / 10000.0f+1.58f);

				

				player.trans.pos = player.trans.pos.Add(veloc.ApplyRot(player.trans.rot).Mul(deltaTime));
				lights[0] = player.trans.pos.Add(Vector3(0, 10, 0));

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
#include <stdio.h>
#include "engine.cu"

#define MOVE_SPEED 3.0f
#define M_SENS 0.0005f

typedef struct Player {
	Transform trans;
	Vector3 veloc;
} Player;

int main(int argc, char** argv)
{
	//Open the window by initialising renderer object
	Renderer renderer = Renderer(1000, 1000);

	//Initialise camera
	Player player;
	player.trans.pos = Vector3(0, 0, -5);
	player.trans.rot = Vector3(0, 0, 0);

	//Allocate scene memory
	Scene* scene;
	cudaMallocManaged(&scene, sizeof(Scene));

	//Initialise primitives and set colours

	Sphere* sphereb = (Sphere*)scene->shapes.CreateSphere(Vector3(0, 3, 0), 0.6f, 0);
	Plane* plane = (Plane*)scene->shapes.CreatePlane(-2.3f, 0);
	Cube* cube = (Cube*)scene->shapes.CreateCube(Vector3(0, -2.3f, 0), Vector3(0, 0, 0), Vector3(3.0f, 0.01f, 100.0f), 0);


	Cube* wall0 = (Cube*)scene->shapes.CreateCube(Vector3(-10, 0, 0), Vector3(0, 0, 0), Vector3(1.0f, 10.f, 10.0f), 0);
	Cube* wall1 = (Cube*)scene->shapes.CreateCube(Vector3(10, 0, 0), Vector3(0, 0, 0), Vector3(1.0f, 10.f, 10.0f), 0);
	Cube* wall2 = (Cube*)scene->shapes.CreateCube(Vector3(0, 0, 10), Vector3(0, 0, 0), Vector3(10.0f, 10.f, 1.0f), 0);
	Cube* wall3 = (Cube*)scene->shapes.CreateCube(Vector3(0, 0, -10), Vector3(0, 0, 0), Vector3(10.0f, 10.f, 1.0f), 0);
	Sphere* spherea = (Sphere*)scene->shapes.CreateSphere(Vector3(-10, 0, 0), 3.0f, 2);

	spherea->col = Vector3(187, 134, 252);
	sphereb->col = Vector3(187, 134, 252);
	plane->col = Vector3(18, 18, 18);
	cube->col = Vector3(50, 50, 50);

	wall0->col = Vector3(255, 0, 0);
	wall1->col = Vector3(255, 0, 0);
	wall2->col = Vector3(255, 0, 0);
	wall3->col = Vector3(255, 0, 0);
	
	//Add light to scene
	scene->lights.AddLight(Vector3(0, 5, 0), 1);

	//Set up timekeeping variables
	int time = SDL_GetTicks();
	float deltaTime;

	//Main loop flag
	bool quit = false;

	//Flag to lock screen to allow for screenshots
	int lockMouse = 0;

	//Event handler
	SDL_Event e;

	//Loop until window is closed
	while (!quit)
	{
		//Iterate over all events
		while (SDL_PollEvent(&e) != 0)
		{
			switch (e.type) {
				//Camera look code
				case (SDL_MOUSEMOTION):
					if (!lockMouse) {
						player.trans.rot.x += e.motion.yrel * M_SENS;
						player.trans.rot.y += e.motion.xrel * M_SENS;
					}
					break;

				//Keydown logic
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
						case SDLK_SPACE:
							lockMouse = 1 - lockMouse;
							break;
						case SDLK_ESCAPE:
							quit = true;
							break;
						}
					break;

				//Keyup logic
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
			//If quit button pressed
			if (e.type == SDL_QUIT)
			{
				quit = true;
			}
		}

		//Move spheres
		//spherea->trans.pos.x = -1 + -1 * cos(SDL_GetTicks() / 1000.0f) / 3;
		//spherea->trans.pos.y = -0.3f + 1 * cos(SDL_GetTicks() / 1000.0f) / 3;
		sphereb->trans.pos.x = -1 + 1 * cos(SDL_GetTicks() / 1000.0f + 1.58f) / 3;
		sphereb->trans.pos.y = -0.3f + 1 * cos(SDL_GetTicks() / 1000.0f + 1.58f) / 3;
		//spherea->trans.pos = player.trans.pos;
		//Move player
		player.trans.pos = player.trans.pos.Add(player.veloc.ApplyRot(player.trans.rot).Mul(deltaTime));

		//Position light above player
		scene->lights.GetLight(0)->pos = player.trans.pos.Add(Vector3(0, 0, 0));

		//Draw frame
		renderer.Draw(player.trans, *scene);

		//Calculate time between frames
		deltaTime = (SDL_GetTicks() - time) / 1000.0f;

		//Debug text
		//printf("Time for frame: %ums\n", SDL_GetTicks() - time);
		printf("Distance to sphere: %f\n", subDist(wall0->DistanceTo(player.trans.pos), spherea->DistanceTo(player.trans.pos)).dist);
		//Update time since program start
		time = SDL_GetTicks();
	}

	return 0;
}

#include "holder.cu"
#pragma once
class Scene {
public:
	ShapeHolder shapes;
	LightHolder lights;

	__device__
		DistReturn distToScene(Vector3 pos) {
		DistReturn dist;
		dist.dist = 9999;
		dist.col = Vector3(1, 1, 1);
		for (int i = 0; i < shapes.length; i++) {
			Shape* ptr = (Shape*)shapes.values[i];
			DistReturn newDist;
			switch (ptr->type) {
				case ('s'):
					newDist = ((Sphere*)ptr)->DistanceTo(pos);
					break;
				case ('c'):
					newDist = ((Cube*)ptr)->DistanceTo(pos);
					break;
				case ('p'):
					newDist = ((Plane*)ptr)->DistanceTo(pos);
					break;
				case ('h'):
					newDist = ((HollowCube*)ptr)->DistanceTo(pos);
					break;
			}

			switch (ptr->blend) {
				case 0:
					dist = regDist(dist, newDist);
					break;
				case 1:
					dist = smoothDist(dist, newDist, 0.2f);
					break;
				case 2:
					dist = subDist(dist, newDist);
					break;
			}
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
};
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
__device__
devDistReturn smoothDist(devDistReturn distA, devDistReturn distB, half k) {
	devDistReturn result;
	float h = __saturatef(__hfma(0.5f, __hdiv(__hsub(distB.dist, distA.dist), k), __float2half(0.5f)));

	result.col = distB.col.Lerp(distA.col, h);
	result.dist = __hfma(distB.dist, __hsub(1, h), __hsub(__hmul(distA.dist, h), __hmul(__hmul(k, h), __hsub(1.0f, h))));
	return result;
}

__device__
devDistReturn regDist(devDistReturn distA, devDistReturn distB) {
	devDistReturn result;
	if (__hlt(distA.dist, distB.dist)) {
		result.col = distA.col;
		result.dist = distA.dist;
	}
	else {
		result.col = distB.col;
		result.dist = distB.dist;
	}
	return result;
}
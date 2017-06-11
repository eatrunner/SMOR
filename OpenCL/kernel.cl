__kernel void process1(const int w, const int h, const int TILE_DIM, __global const float* const xt, __global float* const yt, __global const float *mask1_, __global const float *mask2_, __global const float *shifts_)
{
	__local float mask1[9], mask2[9], shifts[9];
	float weightSum1 = 0, weightSum2 = 0;

	for (int i = 0; i < 9; ++i)
	{
		mask1[i] = mask1_[i];
		mask2[i] = mask2_[i];
		shifts[i] = shifts_[i];
		weightSum1 += mask1_[i];
		weightSum2 += mask2_[i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	
	const int ix = get_group_id(0)*TILE_DIM + get_local_id(0); //jezelil TILE_DIM==local_size wtedy mozna get_global_id
	const int iy = get_group_id(1)*TILE_DIM + get_local_id(1);
	if (ix == 0 || ix == w || iy == 0 || iy == h)
		return;
	int pos = ix + iy * w;
	float sum1 = 0, sum2 = 0;

	for (int shift = 0; shift < 9; shift++) {
		pos = pos + shifts[shift];
		if (pos < 0 || pos >= w*h) {
			continue;
		}
		sum1 += xt[pos] * mask1[shift];
		sum2 += xt[pos] * mask2[shift];
	}
	sum1 = weightSum1 == 0 ? sum1 : sum1 / weightSum1;
	sum2 = weightSum2 == 0 ? sum2 : sum2 / weightSum2;

	yt[pos] = min(255.0f, sqrt(sum1*sum1 + sum2*sum2));
}
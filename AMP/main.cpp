#include <amp.h>
#include <amp_math.h>
#include "FileUtils.h"

#include <math.h>
#include <string>
#include <ctime>
#include <iostream>

#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define abs(a)	   (((a) < 0) ? -(a) : (a))
#define TILE_DIM 32

using namespace std;
using namespace concurrency;
using namespace concurrency::fast_math;

void ampImageProcessing(unsigned char *image, unsigned char *out_image, int w, int h, float const *mask1_, float const *mask2_)
{
	accelerator	device(accelerator::default_accelerator);
	accelerator_view av = device.default_view;
	vector<float> dev_input(w*h), dev_output(w*h);
	vector<float> mask1, mask2;
	vector<int>shifts(9);
	for (int i = 0; i < 9; ++i)
	{
		mask1.push_back(mask1_[i]);
		mask2.push_back(mask2_[i]);
	}

	shifts = {
		-1 - w,		-w ,	-w + 1,
		-1,			0,		1,
		-1 + w ,	w ,		1 + w };

	for (int i = 0; i < w*h; ++i)
	{
		dev_input[i] = image[i];
		dev_output[i] = image[i];
	}

	array_view<float, 1> in(w*h, dev_input);
	array_view<float, 1> out(w*h, dev_output);
	array_view<float, 1> m1(3 * 3, mask1);
	array_view<float, 1> m2(3 * 3, mask2);
	array_view<int, 1> sh(3 * 3, shifts);
	out.discard_data();
	concurrency::extent<2>dim(h, w);

	clock_t begin = clock();

	parallel_for_each(dim.tile<TILE_DIM, TILE_DIM>(), [w, h, in, out, m1, m2, sh](tiled_index<TILE_DIM, TILE_DIM> tidx)restrict(amp)
	{
		const int y = tidx.tile[0] * TILE_DIM + tidx.local[0];
		const int x = tidx.tile[1] * TILE_DIM + tidx.local[1];
		float weightSum1 = 0, weightSum2 = 0;
		if (x == 0 || x == w || y == 0 || y == h)
			return;
		int pos = y*w + x;

		float sum1 = 0, sum2 = 0;
		for (int shift = 0; shift < 9; shift++) {
			pos = pos + sh[shift];
			if (pos < 0 || pos >= w*h) {
				continue;
			}
			sum1 += in[pos] * m1[shift];
			sum2 += in[pos] * m2[shift];
			weightSum1 += m1[shift];
			weightSum2 += m2[shift];
		}
		sum1 = weightSum1 == 0 ? sum1 : sum1 / weightSum1;
		sum2 = weightSum2 == 0 ? sum2 : sum2 / weightSum2;

		float tmp = sqrt(sum1*sum1 + sum2*sum2);
		out[pos] = tmp < 255.0f ? tmp : 255.0f;

	});
	out.synchronize();
	clock_t end = clock();


	for (int i = 0; i < w*h; ++i)
		out_image[i] = out[i];

	cout << "AMP time[ms]: " << double(end - begin) * 1000 / CLOCKS_PER_SEC << endl;


}

int main(int argc, char* argv[])
{
	const float mask1f1[] = {
		0.17, 0.67, 0.17,
		0.67, -3.33, 0.67,
		0.17, 0.67, 0.17 };

	const float mask2f1[] = {
		0, 0, 0,
		0, 0, 0,
		0, 0, 0 };

	const float mask1f2[] = {
		1, 2, 1,
		2, 4, 2,
		1, 2, 1 };

	const float mask2f2[] = {
		0, 0, 0,
		0, 0, 0,
		0, 0, 0 };


	const float mask1f3[] = {
		-1, 0, 1,
		-2, 0, 2,
		-1, 0, 1 };

	const float mask2f3[] = {
		-1, -2, -1,
		0, 0, 0,
		1, 2, 1 };
	int height, width, bytes_per_pixel;;
	vector<string> vfiles = { "p1.jpg", "p2.jpg", "p4.jpg", "p5.jpg" };
	vector<unsigned char*> in_data_ptr, out_data_ptr, out_ptr, in_ptr;

	for (int i = 0; i < vfiles.size(); ++i)
	{
		in_data_ptr.push_back(loadJPG(vfiles[i].c_str(), width, height, bytes_per_pixel));
		//3x filter
		for (int i = 0; i < 3; ++i)
		{
			out_data_ptr.push_back(loadJPG(vfiles[i].c_str(), width, height, bytes_per_pixel));
			out_ptr.push_back(new unsigned char[width*height]);
		}
		in_ptr.push_back(new unsigned char[width*height]);
	}
	for (int j = 0; j < in_data_ptr.size(); ++j)
		for (int i = 0; i < width*height; ++i)
		{
			in_ptr[j][i] = (in_data_ptr[j][3 * i] + in_data_ptr[j][3 * i + 1] + in_data_ptr[j][3 * i + 2]) / 3;
			if (in_data_ptr[j] == NULL) return 0;
		}

	for (int i = 0; i < in_ptr.size(); ++i)
	{
		ampImageProcessing(in_ptr[i], out_ptr[3 * i], width, height, mask1f1, mask2f1);
		ampImageProcessing(in_ptr[i], out_ptr[3 * i + 1], width, height, mask1f2, mask2f2);
		ampImageProcessing(in_ptr[i], out_ptr[3 * i + 2], width, height, mask1f3, mask2f3);
	}

	for (int j = 0; j < out_data_ptr.size(); ++j)
		for (int i = 0; i < width*height; ++i)
		{
			out_data_ptr[j][3 * i] = out_ptr[j][i];
			out_data_ptr[j][3 * i + 1] = out_ptr[j][i];
			out_data_ptr[j][3 * i + 2] = out_ptr[j][i];
		}

	for (int j = 0; j < in_data_ptr.size(); ++j)
	{
		string name = "OutAMPF1p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j], width, height);

		name = "OutAMPF2p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j + 1], width, height);

		name = "OutAMPF3p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j + 2], width, height);
	}

	for (int j = 0; j < in_data_ptr.size(); ++j)
	{
		delete[] in_data_ptr[j];
		delete[] in_ptr[j];
	}
	for (int j = 0; j < out_data_ptr.size(); ++j)
	{
		delete[] out_data_ptr[j];
		delete[] out_ptr[j];
	}

	system("pause");

	return 0;
}
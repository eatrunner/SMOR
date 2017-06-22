#include "FileUtils.h"

#include <math.h>
#include <string>
#include <ctime>
#include <iostream>
#include <vector>
#include <thread>
#include <sstream>

using namespace std;

const int TILE_DIM = 32;
void cpuImageProcessing(unsigned char *image, unsigned char *out_image, int w, int h, float const *mask1_, float const *mask2_)
{
	std::vector<std::thread> vt;
	vector<int>shifts(9);
	float const* mask1 = mask1_;

	float const* mask2 = mask2_;
	shifts = {
		-1 - w,		-w ,	-w + 1,
		-1,			0,		1,
		-1 + w ,	w ,		1 + w };
	clock_t begin = clock();
	for (int i = 0; i < h / TILE_DIM; ++i)
	{
		for (int j = 0; j < w / TILE_DIM; ++j)
			vt.push_back(thread([i, j, &w, &h, image, out_image, mask1, mask2, &shifts]() {
			for (int ii = 0; ii < TILE_DIM; ++ii)
				for (int jj = 0; jj < TILE_DIM; ++jj)
				{
					int y = i * TILE_DIM + ii;
					int x = j * TILE_DIM + jj;
					float weightSum1 = 0, weightSum2 = 0;
					if (x == 0 || x == w || y == 0 || y == h)
						return;
					int pos = y*w + x;

					float sum1 = 0, sum2 = 0;
					for (int shift = 0; shift < 9; shift++)
					{
						pos = pos + shifts[shift];
						if (pos < 0 || pos >= w*h) {
							continue;
						}
						sum1 += image[pos] * mask1[shift];
						sum2 += image[pos] * mask2[shift];
						weightSum1 += mask1[shift];
						weightSum2 += mask2[shift];
					}
					sum1 = weightSum1 == 0 ? sum1 : sum1 / weightSum1;
					sum2 = weightSum2 == 0 ? sum2 : sum2 / weightSum2;

					float tmp = sqrt(sum1*sum1 + sum2*sum2);
					out_image[pos] = tmp < 255.0f ? tmp : 255.0f;
				}
		}));
	}
	for (auto &t : vt)
		t.join();

	clock_t end = clock();

	cout << "MT cpu time[ms]: " << double(end - begin) * 1000 / CLOCKS_PER_SEC << endl;

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
		cpuImageProcessing(in_ptr[i], out_ptr[3 * i], width, height, mask1f1, mask2f1);
		cpuImageProcessing(in_ptr[i], out_ptr[3 * i + 1], width, height, mask1f2, mask2f2);
		cpuImageProcessing(in_ptr[i], out_ptr[3 * i + 2], width, height, mask1f3, mask2f3);
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
		string name = "OutMTF1p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j], width, height);

		name = "OutMTF2p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j + 1], width, height);

		name = "OutMTF3p" + to_string(j + 1) + ".jpg";
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
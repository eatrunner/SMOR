
//Realizacja w CUDA, OpenCL cl.hpp, C++ AMP trzech wybranych filtrow dla przykladowych obrazow statycznych - porownanie z MT CPU
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "FileUtils.h"

#include <math.h>
#include <string>
#include <ctime>
#include <iostream>

using namespace std;

#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define abs(a)	   (((a) < 0) ? -(a) : (a))

#define BLOCK_SIZE 32
#define MASK_SIZE 5

__constant__ int shifts[3 * 3];

__constant__ float mask1f1[] = {
	0.17f, 0.67f, 0.17f,
	0.67f, -3.33f, 0.67f,
	0.17f, 0.67f, 0.17f };
__constant__ int mask2f1[] = {
	0, 0, 0,
	0, 0, 0,
	0, 0, 0 };
__constant__ int mask1f2[] = {
	1, 2, 1,
	2, 4, 2,
	1, 2, 1 };
__constant__ int mask2f2[] = {
	0, 0, 0,
	0, 0, 0,
	0, 0, 0 };
__constant__ int mask1f3[] = {
	-1, -1, -1,
	0, 0, 0,
	1, 1, 1 };
__constant__ int mask2f3[] = { 
	-1, 0, 1,
	-1, 0, 1,
	-1, 0, 1 };


__global__ void cuImageProcess1(const float *dev_image, float *dev_out, int w, int h)
{
	int tx = threadIdx.x;   int ty = threadIdx.y;
	int bx = blockIdx.x;	int by = blockIdx.y;

	int thread_pos_x = bx *BLOCK_SIZE + tx;
	int thread_pos_y = by *BLOCK_SIZE + ty;
	float weightSum1 = 0, weightSum2 = 0;
	//ignorujemy krawedzie
	if (thread_pos_x == 0 || thread_pos_x == w || thread_pos_y == 0 || thread_pos_y == h)
		return;

	int pos = thread_pos_x + thread_pos_y*w;

	float sum1 = 0, sum2 = 0;
	for (int shift = 0; shift < 9; shift++) {
		pos = pos + shifts[shift];
		if (pos < 0 || pos >= w*h) {
			continue;
		}
		sum1 += dev_image[pos] * mask1f1[shift];
		sum2 += dev_image[pos] * mask2f1[shift];
		weightSum1 += mask1f1[shift];
		weightSum2 += mask2f1[shift];

	}
	sum1 = weightSum1 == 0 ? sum1 : sum1 / weightSum1;
	sum2 = weightSum2 == 0 ? sum2 : sum2 / weightSum2;
	dev_out[pos] = min(255.0f, sqrtf(sum1*sum1 + sum2*sum2));
	__syncthreads();

}

__global__ void cuImageProcess2(const float *dev_image, float *dev_out, int w, int h)
{
	int tx = threadIdx.x;   int ty = threadIdx.y;
	int bx = blockIdx.x;	int by = blockIdx.y;

	int thread_pos_x = bx *BLOCK_SIZE + tx;
	int thread_pos_y = by *BLOCK_SIZE + ty;
	float weightSum1 = 0, weightSum2 = 0;
	//ignorujemy krawedzie
	if (thread_pos_x == 0 || thread_pos_x == w || thread_pos_y == 0 || thread_pos_y == h)
		return;

	int pos = thread_pos_x + thread_pos_y*w;

	float sum1 = 0, sum2 = 0;
	for (int shift = 0; shift < 9; shift++) {
		pos = pos + shifts[shift];
		if (pos < 0 || pos >= w*h) {
			continue;
		}
		sum1 += dev_image[pos] * mask1f2[shift];
		sum2 += dev_image[pos] * mask2f2[shift];
		weightSum1 += mask1f2[shift];
		weightSum2 += mask2f2[shift];

	}
	sum1 = weightSum1 == 0 ? sum1 : sum1 / weightSum1;
	sum2 = weightSum2 == 0 ? sum2 : sum2 / weightSum2;
	dev_out[pos] = min(255.0f, sqrtf(sum1*sum1 + sum2*sum2));
	__syncthreads();

}

__global__ void cuImageProcess3(const float *dev_image, float *dev_out, int w, int h)
{
	int tx = threadIdx.x;   int ty = threadIdx.y;
	int bx = blockIdx.x;	int by = blockIdx.y;

	int thread_pos_x = bx *BLOCK_SIZE + tx;
	int thread_pos_y = by *BLOCK_SIZE + ty;
	float weightSum1 = 0, weightSum2 = 0;
	//ignorujemy krawedzie
	if (thread_pos_x == 0 || thread_pos_x == w || thread_pos_y == 0 || thread_pos_y == h)
		return;

	int pos = thread_pos_x + thread_pos_y*w;

	float sum1 = 0, sum2 = 0;
	for (int shift = 0; shift < 9; shift++) {
		pos = pos + shifts[shift];
		if (pos < 0 || pos >= w*h) {
			continue;
		}
		sum1 += dev_image[pos] * mask1f3[shift];
		sum2 += dev_image[pos] * mask2f3[shift];
		weightSum1 += mask1f3[shift];
		weightSum2 += mask2f3[shift];

	}
	sum1 = weightSum1 == 0 ? sum1 : sum1 / weightSum1;
	sum2 = weightSum2 == 0 ? sum2 : sum2 / weightSum2;
	dev_out[pos] = min(255.0f, sqrtf(sum1*sum1 + sum2*sum2));
	__syncthreads();

}

extern "C" bool cuImageProcessing1(unsigned char *image, unsigned char *out_image, int w, int h)
{
	int tmp_shifts[] = { -1 - w, -w , -w  + 1, -1, 0, 1, -1 + w , w , 1 + w  };
	
	cudaMemcpyToSymbol(shifts, tmp_shifts, 3 * 3 * sizeof(int));

	// convert to float
	float *pinned_input_image, *pinned_output_image;
	float *dev_input, *dev_output;

	cudaHostAlloc<float>((float**)&pinned_input_image,
		w *h  * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc<float>((float**)&pinned_output_image,
		w *h * sizeof(float), cudaHostAllocDefault);

	for (int i = 0; i < h*w; i++)
			pinned_input_image[i] = image[i];

	cudaMalloc((void**)&dev_input, w *h * sizeof(float));
	cudaMalloc((void**)&dev_output, w *h * sizeof(float));
	cudaMemcpy(dev_input, pinned_input_image, w *h * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(w / BLOCK_SIZE, h / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	clock_t begin = clock();
	cuImageProcess1 <<<dimGrid, dimBlock >>>(dev_input, dev_output, w , h );
	cudaDeviceSynchronize();
	clock_t end = clock();

	cudaMemcpy(pinned_output_image, dev_output, w * h  * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i < h * w; i++)
		out_image[i] = pinned_output_image[i];

	cudaFree(dev_input);
	cudaFree(dev_output);

	cudaFreeHost(pinned_input_image);
	cudaFreeHost(pinned_output_image);

	cout << "CUDA time[ms]: " << double(end - begin) * 1000 / CLOCKS_PER_SEC << endl;

	return true;
}
extern "C" bool cuImageProcessing2(unsigned char *image, unsigned char *out_image, int w, int h)
{
	int tmp_shifts[] = { -1 - w, -w , -w + 1, -1, 0, 1, -1 + w , w , 1 + w };

	cudaMemcpyToSymbol(shifts, tmp_shifts, 3 * 3 * sizeof(int));

	// convert to float
	float *pinned_input_image, *pinned_output_image;
	float *dev_input, *dev_output;

	cudaHostAlloc<float>((float**)&pinned_input_image,
		w *h * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc<float>((float**)&pinned_output_image,
		w *h * sizeof(float), cudaHostAllocDefault);

	for (int i = 0; i < h*w; i++)
		pinned_input_image[i] = image[i];

	cudaMalloc((void**)&dev_input, w *h * sizeof(float));
	cudaMalloc((void**)&dev_output, w *h * sizeof(float));
	cudaMemcpy(dev_input, pinned_input_image, w *h * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(w / BLOCK_SIZE, h / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	clock_t begin = clock();
	cuImageProcess2 << <dimGrid, dimBlock >> >(dev_input, dev_output, w, h);
	cudaDeviceSynchronize();
	clock_t end = clock();

	cudaMemcpy(pinned_output_image, dev_output, w * h * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i < h * w; i++)
		out_image[i] = pinned_output_image[i];

	cudaFree(dev_input);
	cudaFree(dev_output);

	cudaFreeHost(pinned_input_image);
	cudaFreeHost(pinned_output_image);

	cout << "CUDA time[ms]: " << double(end - begin) * 1000 / CLOCKS_PER_SEC << endl;

	return true;
}
extern "C" bool cuImageProcessing3(unsigned char *image, unsigned char *out_image, int w, int h)
{
	int tmp_shifts[] = { -1 - w, -w , -w + 1, -1, 0, 1, -1 + w , w , 1 + w };

	cudaMemcpyToSymbol(shifts, tmp_shifts, 3 * 3 * sizeof(int));

	// convert to float
	float *pinned_input_image, *pinned_output_image;
	float *dev_input, *dev_output;

	cudaHostAlloc<float>((float**)&pinned_input_image,
		w *h * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc<float>((float**)&pinned_output_image,
		w *h * sizeof(float), cudaHostAllocDefault);

	for (int i = 0; i < h*w; i++)
		pinned_input_image[i] = image[i];

	cudaMalloc((void**)&dev_input, w *h * sizeof(float));
	cudaMalloc((void**)&dev_output, w *h * sizeof(float));
	cudaMemcpy(dev_input, pinned_input_image, w *h * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid(w / BLOCK_SIZE, h / BLOCK_SIZE);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	clock_t begin = clock();
	cuImageProcess3 << <dimGrid, dimBlock >> >(dev_input, dev_output, w, h);
	cudaDeviceSynchronize();
	clock_t end = clock();

	cudaMemcpy(pinned_output_image, dev_output, w * h * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 1; i < h * w; i++)
		out_image[i] = pinned_output_image[i];

	cudaFree(dev_input);
	cudaFree(dev_output);

	cudaFreeHost(pinned_input_image);
	cudaFreeHost(pinned_output_image);

	cout << "CUDA time[ms]: " << double(end - begin) * 1000 / CLOCKS_PER_SEC << endl;

	return true;
}

int main(int argc, char* argv[])
{
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
		cuImageProcessing1(in_ptr[i], out_ptr[3 * i], width, height);
		cuImageProcessing2(in_ptr[i], out_ptr[3 * i + 1], width, height);
		cuImageProcessing3(in_ptr[i], out_ptr[3 * i + 2], width, height);
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
		string name = "OuCUDAF1p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j], width, height);

		name = "OuCUDAF2p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3 * j + 1], width, height);

		name = "OuCUDAF3p" + to_string(j + 1) + ".jpg";
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
}


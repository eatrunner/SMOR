#include "FileUtils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <time.h>
#include <chrono>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

using namespace std;

const int TILE_DIM = 32;


void clImageProcessing(unsigned char *image, unsigned char *out_image, int w, int h, float const *mask1_, float const *mask2_)
{
	int tmp_shifts[] = { -1 - w, -w, -w + 1, -1, 0, 1, -1 + w, w, 1 + w };
	std::vector<float> vx(w*h); //wektory odpowiedajace macierzy
	std::vector<float> vy(w*h);

	const float* mask1 = mask1_;

	const float* mask2 = mask2_;

	const float shifts[] = {
		-1 - w,		-w,		-w + 1,
		-1,			0,		 1,
		-1 + w,		w,		1 + w };


	std::iota(begin(vx), end(vx), 0.0f);

	for (int i = 0; i < h * w; i++)
		vx[i] = image[i];

	//set OpenCL platform start
	//tu kod definiujacy i ustalaj¹cy platforme OPENCL
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platforms.empty())
		throw std::runtime_error("no OpenCL platform available");

	const int platformInx = 1;

	const cl::Platform pl{ platforms[platformInx] };
	/*for (auto &plt : platforms)
	{
		std::cout << plt.getInfo<CL_PLATFORM_VERSION>() << '\n';
		std::cout << plt.getInfo<CL_PLATFORM_NAME>() << '\n';
		std::cout << plt.getInfo<CL_PLATFORM_VENDOR>() << '\n';

	}*/
	//set OpenCL platform finish

	//set OpenCL device for platform start
	//tu kod definiujacy i ustalaj¹cy Device dla OPENCL
	std::vector<cl::Device> all_devices;
	pl.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.empty())
		throw std::runtime_error("no OpenCL device available");
	const int deviceInx = 0;
	const cl::Device dev{ all_devices[deviceInx] };
	/*for (auto &devt : all_devices)
	{
		std::cout << devt.getInfo <CL_DEVICE_NAME>() << '\n';
		std::cout << devt.getInfo <CL_DEVICE_VENDOR>() << '\n';
		std::cout << devt.getInfo <CL_DEVICE_PROFILE>() << '\n';
		std::cout << devt.getInfo <CL_DEVICE_VERSION>() << '\n';
		std::cout << devt.getInfo <CL_DEVICE_NAME>() << '\n';
	}*/
	//set OpenCL device for platform finish

	//set OpenCL context for device start
	//tu kod definiujacy i ustalaj¹cy Context dla OPENCL
	cl::Context context({ dev });
	//set OpenCL context for device finish

	//set OpenCL program source start
	//tu kod odczytu kernela z pliku np. transpose1.cl
	// i kod wypelniajacy cl::Program::Sources

	std::ifstream fp;
	std::stringstream strStream;
	fp.open("./kernel.cl");
	if (!fp.good())
	{
		std::cout << "Cos nie pyklo\n";
		system("PAUSE");
		exit(1);
	}
	strStream << fp.rdbuf();
	fp.close();
	const std::string kernelS(strStream.str());
	cl::Program::Sources sources;
	sources.push_back({ kernelS.c_str(), kernelS.length() });
	//set OpenCL program source finish
	//set OpenCL program for context and source start
	//tu kod definiujacy Proggram OpenCL
	cl::Program program(context, sources);
	try
	{
		program.build({ dev });
	}
	catch (cl::Error error)
	{
		if (error.err() == CL_BUILD_PROGRAM_FAILURE)
		{
			std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
			std::cerr << log << std::endl;
		}
		throw(error);
	}
	cl::Event event;
	cl::Buffer vx_buffer(context, CL_MEM_READ_WRITE, vx.size() * sizeof(vx[0]));
	cl::Buffer vy_buffer(context, CL_MEM_READ_WRITE, vx.size() * sizeof(vx[0]));
	cl::Buffer m1_buffer(context, CL_MEM_READ_WRITE, 9 * sizeof(mask1[0]));
	cl::Buffer m2_buffer(context, CL_MEM_READ_WRITE, 9 * sizeof(mask2[0]));
	cl::Buffer sh_buffer(context, CL_MEM_READ_WRITE, 9 * sizeof(shifts[0]));

	cl::CommandQueue queue(context, dev, CL_QUEUE_PROFILING_ENABLE);
	queue.enqueueWriteBuffer(vx_buffer, CL_TRUE, 0, vx.size() * sizeof(vx[0]), vx.data());
	queue.enqueueWriteBuffer(vy_buffer, CL_TRUE, 0, vx.size() * sizeof(vx[0]), vy.data());
	queue.enqueueWriteBuffer(m1_buffer, CL_TRUE, 0, 9 * sizeof(mask1[0]), mask1);
	queue.enqueueWriteBuffer(m2_buffer, CL_TRUE, 0, 9 * sizeof(mask2[0]), mask2);
	queue.enqueueWriteBuffer(sh_buffer, CL_TRUE, 0, 9 * sizeof(shifts[0]), shifts);

	cl::Kernel kernel(program, "process1");
	kernel.setArg(0, w);
	kernel.setArg(1, h);
	kernel.setArg(2, TILE_DIM);
	kernel.setArg(3, vx_buffer);
	kernel.setArg(4, vy_buffer);
	kernel.setArg(5, m1_buffer);
	kernel.setArg(6, m2_buffer);
	kernel.setArg(7, sh_buffer);

		//tu wywolanie kernela OPENCL
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(w, h), cl::NDRange(TILE_DIM, TILE_DIM), NULL, &event);
		event.wait();
		cl_int start = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
		cl_int end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

		// read result vy from the device to array vy
		//odczyt wyniku
		queue.enqueueReadBuffer(vy_buffer, CL_TRUE, 0, vx.size() * sizeof(float), vy.data());

		//std::iota(begin(vx), end(vx), 0.0f);
		for (int i = 0; i < w*h; i++)
			 out_image[i] = vy[i];

		float durms = static_cast<float>(end - start) / 1000000;
		cout << "OpenCL time [ms]: " << durms << endl;

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
		clImageProcessing(in_ptr[i], out_ptr[3*i], width, height, mask1f1, mask2f1);
		clImageProcessing(in_ptr[i], out_ptr[3*i + 1], width, height, mask1f2, mask2f2);
		clImageProcessing(in_ptr[i], out_ptr[3*i + 2], width, height, mask1f3, mask2f3);
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
		string name = "OutOpenCLF1p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3*j], width, height);

		name = "OutOpenCLF2p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3*j + 1], width, height);

		name = "OutOpenCLF3p" + to_string(j + 1) + ".jpg";
		saveJPG(name.c_str(), out_data_ptr[3*j + 2], width, height);
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
#ifndef SIMPLEG
#define SIMPLEG

#include "structs.h"
#include "windows.h"

BITMAPINFO* prepareBitmapInfo(int im_width, int im_height);
void findIntegralImage(unsigned char *img,int xres,int yres,int **iimage);
void findIntegralImage(unsigned char *img,int xres,int yres,int *iimage);
void findIntegralImage(unsigned char *img,int xres,int yres,unsigned int *iimage);
void findIntegralImage(float *img,int xres,int yres,float *iimage);
void findIntegralImage(int *img,int xres,int yres,int *iimage);
void findRotatedIntegralImage(unsigned char *img,int xres,int yres,int *iimage);
void findRotatedIntegralImage(int *img,int xres,int yres,int *iimage);
bool rectangleIntersection(const SimpleRect &r1, const SimpleRect &r2);
float rectangleIntersectionRatio(const SimpleRect &r1, const SimpleRect &r2);

void smoothImage(unsigned char *image, int w, int h);
SimplePointD rotate2DPoint(SimplePointD &p, float angle);

void drawPixel(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b);
void drawPixel(int x, int y, int w, int h, byte *pixels, int val);
void drawBigCross(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b);
void drawBigCrossC(int x, int y, int w, int h, byte *pixels, unsigned char val, int size, int thickness);
void drawRectangleRGB(RECT rect, int x, int y, byte *pixelsRGB, int w, int h, int r, int g, int b);
void drawLine(int xa, int ya, int xb, int yb, int w, int h, byte *pixels, unsigned char r, unsigned char g,
				 unsigned char b);
void drawLine(int xa, int ya, int xb, int yb, int w, int h, byte *pixels, int val=255);
SimpleRect drawFaceBoundingRotated(int x, int xres, int y, int yres, int width, int height, 
							 unsigned char* pixels, double angle, int r, int g, int b, bool paint);
void drawCross(int x, int y, int thickness, int size, int w, int h, byte *RGBPixels, int r, int g, int b);
void drawCross(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b);
void drawCrossSimple(int x, int y, int w, int h, byte* pixels,byte val);
void copyRGBToGrayscale(byte *dest, byte *RGBSource, int w, int h);
void copyRGBToGrayscaleWithoutFlip(byte *dest, byte *RGBSource, int w, int h);
void copyRGBToGrayscale(float *dest, byte *RGBSource, int w, int h);
void copyRGBToGrayscale(byte *dest, byte *RGBSource, int w, int h, int extra);

// extraWidth must be difference between w and closest number that can be divided by 8
void copyGrayscaleToRGB(byte *RGBdest, byte *source, int w, int h, int extraWidth);
void rotatePoint(float angle, int x0, int y0, int &x, int &y);
void rotateImage(float angle, int x0, int y0, int iWidth, int iHeight, unsigned char *pbData, 
				 unsigned char **pbDataRows);

// returns specified rectangle area for the image
// x, y - anchoring point
// w, h - size of the new image 
unsigned char* cutImage(int x, int y, int w, int h, unsigned char *data, 
				   int &iWidth, int &iHeight, bool deleteData);

unsigned char *scaleImage2(double scale, unsigned char *data, int &iWidth, int &iHeight);
unsigned char *scaleImage(unsigned char *data, int &iWidth, int &iHeight);
void drawSmallCross(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b);
void scaleImage(double scale, unsigned char *data, int &iWidth, int &iHeight, unsigned char *newData);
int* buildHistogram(byte *image, int w, int h);
void equalizeHistogram(byte *image, int w, int h);
int kMeans(Feature *f, int nElements, int iRectangleSize, int iWidth, int iHeight, float fScaleRatio,
		   int nScales);

void erode(unsigned char *image, int w, int h);
void dilate(unsigned char *image, int w, int h);



unsigned char *scaleImageToSize(unsigned char *data, int dest_w, int
								 dest_h, int src_w, int src_h);

int distance(int x1, int y1, int x2, int y2);
float distance(float x1, float y1, float x2, float y2);
int distance(Feature f1, Feature f2);
int distance2(Feature f1, Feature f2);
float distanceExact(int x1, int y1, int x2, int y2);
float distanceExact(Feature &f1, Feature &f2);
float distanceExact(float x1, float y1, float x2, float y2);
double distanceExact(SimplePointD &p1, SimplePointD &p2);


void findIntegralImage(unsigned char *img,int xres,int yres,int **iimage);

void changeBrightness(unsigned char *image, int w, int h, int val);

void drawRectangleG(SimpleRect rect, int x, int y, byte *pixels, int w, int h, unsigned char val = 0);
void drawRectangleRGB(SimpleRect rect, int x, int y, byte *pixelsRGB, int w, int h, int r, int g, int b);
void copyRegion(unsigned char *dst, unsigned char *src, int w, int h, SimpleRect region);
void getPixelRGB(int x, int y, unsigned char *RGBPixels, int w, int h, unsigned char &R, 
	unsigned char &G, unsigned char &B);
double getSubpixelImageValue(SimplePointD &p, unsigned char *image, int w, int h);
double getSubpixelXGradient(SimplePointD &p, unsigned char *image, int w, int h);
double getSubpixelYGradient(SimplePointD &p, unsigned char *image, int w, int h);
void flipImageVertical(unsigned char *rgb_image, int w, int h, int bpp);
void convertRGBToBR(unsigned char *br_image, unsigned char *rgb_image, int w, int h);
void convertRGBToBRY(unsigned char *bry_image, unsigned char *rgb_image, int w, int h);


#endif
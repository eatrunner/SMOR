#ifndef FILEUTILS
#define FILEUTILS

#include <string>
#include "structs.h"
#include <vector>

using namespace std;

void saveJPG(const char *filename, unsigned char *data, int width, int height);
unsigned char * loadJPG(const char *filename, int &iWidth, int &iHeight, int &bpp);
void skipLine(FILE *f);
bool checkFileExistence(char *filename);
int readCoordinates(const char* filename, int *pointsx, int *pointsy);
vector<FaceCoordinates> readCoordinates(const char* filename);
void writeToTmp1(char* filename, float *buffer, int element_size, bool printZeros, int n);
void markCodePoint(char *label, int l=0); // write to the c:\label.txt label 

#endif

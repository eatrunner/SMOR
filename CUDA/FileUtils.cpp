
#include "FileUtils.h"
#include "structs.h"
#include "windows.h"
#include "SimpleGraphic.h"
#include <atlimage.h>

void saveJPG(const char * filename, unsigned char *data, int width, int height)
{
	CImage im;
	bool delete_data = false;

	if ((width%8)!=0)
	{
		int new_w = width + (8 - width%8);
		unsigned char *new_data = new unsigned char[new_w*height*3];
		memset(new_data, 0, new_w*height*3);
		for (int i=0; i<height; i++)
			memcpy(new_data + i*new_w*3, data + i*width*3, width*3);
		
		data = new_data;
		width = new_w;
		delete_data = true;
	}

	BITMAPINFO *bip = prepareBitmapInfo(width, height);

	im.Create(width, height, 24);
	HDC hdc = im.GetDC();

	StretchDIBits(hdc, 0, 0, width, height, 0, 0, width, height, 
				  data, bip, DIB_RGB_COLORS, SRCCOPY);

	/*im.Save(CStringW(filename));*/
	im.Save(LPCTSTR(filename));
	im.ReleaseDC();
	if (delete_data) delete[] data;
	delete bip;
}

unsigned char *loadJPG(const char *filename, int &iWidth, int &iHeight, int &bpp)
{
	CImage im;
    int pitch;
    byte *pdata;
	unsigned char *data;
	HRESULT res;

	//res = im.Load(CStringW(filename));
	res = im.Load(LPCTSTR(filename));

	if (res==E_FAIL) return NULL;
	
    pitch = im.GetPitch();
    pdata = (byte*)im.GetBits();

    iWidth = im.GetWidth();
    iHeight = im.GetHeight();

    bpp = abs(pitch/iWidth);
    data = new unsigned char[iWidth*iHeight*bpp];

	int start_pos;
	if (pitch>0) 
		start_pos = 0;
	else 
		start_pos = (iHeight-1) * iWidth * bpp;

	for (int i=0; i<iHeight; i++)
    {
        pdata = (byte*)im.GetBits() + pitch*i;
        for (int j=0; j<iWidth; j++) 
        {
			for (int k=0; k<bpp; k++)
				data[((i*iWidth) + j)*bpp + k] = *(pdata++);
        }
    }

	if (pitch<0) 
			flipImageVertical(data, iWidth, iHeight, bpp);

	return data;
}

void skipLine(FILE *f)
{
	int c = -1;

	while (c!='\n')
		c = getc(f);
}

bool checkFileExistence(char *filename)
{
	FILE *f;

	if (fopen_s(&f, filename, "rt")!=0)
		return false;

	fclose(f);
	return true;
}

void writeToTmp1(char* filename, float *buffer, int element_size, bool printZeros, int n)
{
	float *tmp;
	FILE *f;
	fopen_s(&f, filename, "wt");

	tmp = buffer;

	int cnt=0;
	for (int el=0; el<n; el++)
	{	
		if (el%element_size==0) 
		{
			fprintf(f, "\n\n");
			cnt = 0;
		}
		if (printZeros || tmp[el]!=0) 
		{
			fprintf(f, "%f ", /*cnt, */tmp[el]);
			cnt++;
		}
	}

	fclose(f);
}



int readCoordinates(const char *filename, int *pointsx, int *pointsy)
{
	int xcounter = 0, len;
	char temp[1024];

	strcpy(temp, filename);
	len = strlen(temp);

	temp[len-3] = 't';
	temp[len-2] = 'x';
	temp[len-1] = 't';

	FILE *f;

	fopen_s(&f, temp, "rt");

	if (f!=NULL)
	{
	  for (int i=0; i<N_POINTS; i++)
	    if (fscanf(f, "%d %d ", &pointsx[i], &pointsy[i])==EOF)
		{
			pointsx[i] = -1;
			pointsy[i] = -1;

			xcounter = 0;
		} else xcounter++;
	  fclose(f);
	} else 
	for (int i=0; i<N_POINTS; i++)
	{
		pointsx[i] = -1;
		pointsy[i] = -1;
	}

	return xcounter;
}

vector<FaceCoordinates> readCoordinates(const char* filename)
{
	int xcounter = 0, len;
	char temp[1024];
	int n_faces = 0;
	vector<FaceCoordinates> coords;

	strcpy(temp, filename);
	len = strlen(temp);

	temp[len-3] = 't';
	temp[len-2] = 'x';
	temp[len-1] = 't';

	FILE *f;

	fopen_s(&f, temp, "rt");
	SimplePoint le, re;
	FaceCoordinates single_face;

	if (f!=NULL)
	{
		fscanf(f, "%d", &n_faces);	
		
		for (int act_face=0; act_face<n_faces; act_face++)
			{
				fscanf(f, "%d %d %d %d", &le.x, &le.y, &re.x, &re.y);
				single_face.le = le;
				single_face.re = re;
				single_face.between_eyes_distance = distanceExact(le.x, le.y, re.x, re.y);
				single_face.between_eyes_point.x = (le.x + re.x)/2;
				single_face.between_eyes_point.y = (le.y + re.y)/2;
				coords.push_back(single_face);
			}
		fclose(f);
	}

	return coords;
}

void markCodePoint(char *label, int l) // write to the c:\label.txt label 
{
	char tmp[256];
	sprintf(tmp, "c:\\label_%d.txt", l);
	FILE *f;
	fopen_s(&f, tmp, "wt");

	fprintf(f, "%s", label);

	fclose(f);
}

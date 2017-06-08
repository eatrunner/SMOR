
#include "math.h"
#include "structs.h"
#include "SimpleGraphic.h"

#ifndef min
#define min(a, b)  (((a) < (b)) ? (a) : (b))
#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define abs(a)	   (((a) < 0) ? -(a) : (a))
#endif
void drawCross(int _x, int _y, int thickness, int size, int w, int h, byte *RGBPixels, int r, int g, int b)
{
	thickness-=1;

	for (int x=_x-size; x<=_x+size; x++)
		for (int y=_y - thickness; y<=_y+thickness; y++)
			drawPixel(x, y, w, h, RGBPixels, r, g, b);

	for (int y=_y-size; y<=_y+size; y++)
		for (int x=_x - thickness; x<=_x+thickness; x++)
			drawPixel(x, y, w, h, RGBPixels, r, g, b);

}

void erode(unsigned char *image, int w, int h)
{
	unsigned char *temp = new unsigned char[w*h];
	memset(temp, 0, w*h);
	int pos;

	for (int y=1; y<(h-1); y++)
	{
		pos = y*w+1;
		for (int x=1; x<(w-1); x++)
		{
			temp[pos] = min(min(min(min(image[pos], image[pos-1]), image[pos+1]), image[pos+w]), image[pos-w]);
			temp[pos] = min(min(min(min(temp[pos], image[pos-1-w]), image[pos+1-w]), image[pos-1+w]), image[pos+1+w]);
			pos++;
		}
	}
	memcpy(image, temp, w*h);

	delete []temp;	
}

void dilate(unsigned char *image, int w, int h)
{
	unsigned char *temp = new unsigned char[w*h];
	memset(temp, 0, w*h);
	int pos;

	for (int y=1; y<(h-1); y++)
	{
		pos = y*w+1;
		for (int x=1; x<(w-1); x++)
		{
			temp[pos] = max(max(max(max(image[pos], image[pos-1]), image[pos+1]), image[pos+w]), image[pos-w]);
			temp[pos] = max(max(max(max(temp[pos], image[pos-1-w]), image[pos+1-w]), image[pos-1+w]), image[pos+1+w]);
			pos++;
		}
	}
	memcpy(image, temp, w*h);

	delete []temp;	
}

double getSubpixelXGradient(SimplePointD &p, unsigned char *image, int w, int h)
{
	double x0, y0, x1, x2;
	double x_r, y_r;
	double val = 0.0;
	double a, b, c, d, diff_x, diff_y;

	if (p.x<1.0 || p.y<1.0 || p.x>(w-3) || p.y>(h-3)) return val;

	x0 = floor(p.x);
	y0 = floor(p.y);

	int pos = (unsigned short)x0 + (unsigned short)y0 * w;

	x_r = p.x - x0;
	y_r = p.y - y0;

	diff_x = 1.0-x_r;
	diff_y = 1.0-y_r;

	x1 = diff_x*diff_y*image[pos-1] + x_r*diff_y*image[pos] +
		  diff_x*y_r*image[pos+w-1] + x_r*y_r*image[pos+w];
	x2 = diff_x*diff_y*image[pos+1] + x_r*diff_y*image[pos+2] +
		  diff_x*y_r*image[pos+w+1] + x_r*y_r*image[pos+w+2];
	  
	//x1 = diff_x*image[pos-1] + x_r*image[pos];  
	//x2 = diff_x*image[pos+1] + x_r*image[pos+2];
		  
	return (x2-x1)*0.5;
}


double getSubpixelYGradient(SimplePointD &p, unsigned char *image, int w, int h)
{
	double x0, y0, x1, x2;
	double x_r, y_r;
	double val = 0.0;
	double a, b, c, d, diff_x, diff_y;

	if (p.x<1.0 || p.y<1.0 || p.x>(w-3) || p.y>(h-3)) return val;

	x0 = floor(p.x);
	y0 = floor(p.y);

	int pos = (unsigned short)x0 + (unsigned short)y0 * w;

	x_r = p.x - x0;
	y_r = p.y - y0;

	diff_x = 1.0-x_r;
	diff_y = 1.0-y_r;

/*	x1 = diff_x*diff_y*image[pos-1] + x_r*diff_y*image[pos] +
		  diff_x*y_r*image[pos+w-1] + x_r*y_r*image[pos+w];
	x2 = diff_x*diff_y*image[pos+1] + x_r*diff_y*image[pos+2] +
		  diff_x*y_r*image[pos+w+1] + x_r*y_r*image[pos+w+2];
*/

	x1 = diff_x*diff_y*image[pos-w] + x_r*diff_y*image[pos+1-w] +
		  diff_x*y_r*image[pos] + x_r*y_r*image[pos+1];
	x2 = diff_x*diff_y*image[pos+w] + x_r*diff_y*image[pos+1+w] +
		  diff_x*y_r*image[pos+w+w] + x_r*y_r*image[pos+w+w+1];
		  
	//x1 = diff_y*image[pos-w] + y_r*image[pos];
	//x2 = diff_y*image[pos+w] + y_r*image[pos+w+w+1];

	return (x2-x1)*0.5;
}

double getSubpixelImageValue(SimplePointD &p, unsigned char *image, int w, int h)
{
	double x0, y0;
	double x_r, y_r;
	double val = 0.0;
	double a, b, c, d, diff_x, diff_y;

	if (p.x<0.0 || p.y<0.0 || p.x>(w-2) || p.y>(h-2)) return val;

	x0 = floor(p.x);
	y0 = floor(p.y);

	int pos = (unsigned short)x0 + (unsigned short)y0 * w;

	x_r = p.x - x0;
	y_r = p.y - y0;

	diff_x = 1.0-x_r;
	diff_y = 1.0-y_r;

	val = diff_x*diff_y*image[pos] + x_r*diff_y*image[pos+1] +
		  diff_x*y_r*image[pos+w] + x_r*y_r*image[pos+w+1];
	
	return val;
}

SimplePointD rotate2DPoint(SimplePointD &p, float angle)
{
	SimplePointD p2;
	float sina, cosa;
	sina = sin(angle);
	cosa = cos(angle);
	p2.x = p.x * cosa - p.y * sina;
	p2.y = p.y * cosa + p.x * sina;

	return p2;
}

void smoothImage(unsigned char *image, int w, int h)
{
    int wmax = w-2;
    int hmax = h-2;
    int w2 = w*2;
    unsigned char *temp = new unsigned char[w*h];
    unsigned char *pos_dest, *pos_source;
	
	memcpy(temp, image, w*h);

    for (int y=2; y<hmax; y++)
    {
        pos_source = image + y*w;
        pos_dest = temp + y*w;
        for (int x=2; x<wmax; x++)
            pos_dest[x] = (int)((pos_source[x-2] + 2*pos_source[x-1] + 4*pos_source[x] +
                          2*pos_source[x+1] + pos_source[x+2])*0.1f);
    }

    for (int y=2; y<hmax; y++)
    {
        pos_source = temp + y*w;
        pos_dest = image + y*w;
        for (int x=2; x<wmax; x++)
            pos_dest[x] = (unsigned char)((pos_source[x-w2] + 2*pos_source[x-w] + 4*pos_source[x] +
                          2*pos_source[x+w] + pos_source[x+w2])*0.1f);
    }

    delete []temp;
}


void copyRegion(unsigned char *dst, unsigned char *src, int w, int h, SimpleRect region)
{
	int y_min = region.top;
	int y_max = region.bottom;
	int x_min = region.left;
	int rw = region.right - region.left;
	int pos;
	for (int y=y_min; y<y_max; y++)
	{
		pos = y*w + x_min;
		memcpy(dst + pos, src + pos, rw);
	}
}


void getPixelRGB(int x, int y, unsigned char *RGBPixels, int w, int h, unsigned char &R, 
	unsigned char &G, unsigned char &B)
{
	int l;
	if (x<w && y<h && x>0 && y>0)
	{
		l = 3*(h-y-1)*w + x*3;
		R = RGBPixels[l];
		G = RGBPixels[l+1];
		B = RGBPixels[l+2];
	}
}


void drawRectangleRGB(RECT rect, int x, int y, byte *pixelsRGB, int w, int h, int r, int g, int b)
{
	int i;

	for (i=max(0, (rect.left + x)); i<min((rect.right+x+1), w); i++)
	{
		drawPixel(i, y+rect.top, w, h, pixelsRGB, r, g, b);
		drawPixel(i, y+rect.bottom, w, h, pixelsRGB, r, g, b);
	}

	for (i=max(0, (rect.top + y)); i<min((rect.bottom+y+1), h); i++)
	{
		drawPixel(rect.left + x, i, w, h, pixelsRGB, r, g, b);
		drawPixel(rect.right + x, i, w, h, pixelsRGB, r, g, b);
	}
}


unsigned char *scaleImageToSize(unsigned char *data, int dest_w, int
 dest_h, int src_w, int src_h)
 {
   unsigned char *temp = new unsigned char[dest_w * dest_h];
   
   int x, y, sum, i, j, n, m;
   float nel;
   float scaleX = max(1.0f, (float)src_w/(float)dest_w);
   float scaleY = max(1.0f, (float)src_h/(float)dest_h);

   unsigned char *pos;

   for (x=0; x<dest_w; x++)
     for (y=0; y<dest_h; y++)
     {
       sum = 0;
       nel = 0;

       for (j=0; j<scaleY; j++)
       {
         n = (int)(y*scaleY)+j;
         m = (int)(x*scaleX);

         if (n>=src_h)
            n = src_h-1;

         pos = data + n*src_w;

         for (i=0; i<scaleX; i++)
         {
           if (m>=src_w) m = src_w-1;
           sum+=(*(pos+m));
           m++;
           nel++;

         }
       }
       temp[x + y*dest_w] = (unsigned char)((float)sum/nel);
     }
   
   return temp;
 }
 
BITMAPINFO* prepareBitmapInfo(int im_width, int im_height)
{
	BITMAPINFO *bip;
	bip = (BITMAPINFO*) new unsigned char[sizeof(BITMAPINFOHEADER)];
    bip->bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bip->bmiHeader.biWidth = im_width;
    bip->bmiHeader.biHeight = im_height;
    bip->bmiHeader.biPlanes = 1;
    bip->bmiHeader.biBitCount = 24;
    bip->bmiHeader.biCompression = BI_RGB;
    bip->bmiHeader.biSizeImage = 0;
    bip->bmiHeader.biXPelsPerMeter = 0;
    bip->bmiHeader.biYPelsPerMeter = 0;
    bip->bmiHeader.biClrUsed = 0;
    bip->bmiHeader.biClrImportant = 0;
	return bip;
}

void changeBrightness(unsigned char *image, int w, int h, int val)
{
	int new_val;
	for (int i=0; i<(w*h); i++)
	{
		new_val = min(max((*image)+val, 0), 255);
		*image = (unsigned char)new_val;
		image++;
	}
}

void drawBigCrossC(int x, int y, int w, int h, byte *pixels, unsigned char val, int size, int thickness)
{
	SimpleRect rect;
	rect.left = x-thickness;
	rect.right = x+thickness;
	rect.top = y-size;
	rect.bottom = y+size;
	drawRectangleG(rect, 0, 0, pixels, w, h, val);

	rect.left = x-size;
	rect.right = x+size;
	rect.top = y-thickness;
	rect.bottom = y+thickness;
	drawRectangleG(rect, 0, 0, pixels, w, h, val);

}

void drawBigCross(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b)
{
	(x, y, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-1, y-1, w, h, RGBPixels, r, g, b);
	drawSmallCross(x+1, y+1, w, h, RGBPixels, r, g, b);
	drawSmallCross(x, y-1, w, h, RGBPixels, r, g, b);
	drawSmallCross(x, y-2, w, h, RGBPixels, r, g, b);
	drawSmallCross(x, y+1, w, h, RGBPixels, r, g, b);
	drawSmallCross(x, y+2, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-1, y, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-2, y, w, h, RGBPixels, r, g, b);
	drawSmallCross(x+1, y, w, h, RGBPixels, r, g, b);
	drawSmallCross(x+2, y, w, h, RGBPixels, r, g, b);
}

void drawRectangleRGB(SimpleRect rect, int x, int y, byte *pixelsRGB, int w, int h, int r, int g, int b)
{
	int i;

	for (i=max(0, (rect.left + x)); i<min((rect.right+x+1), w); i++)
	{
		drawPixel(i, y+rect.top, w, h, pixelsRGB, r, g, b);
		drawPixel(i, y+rect.bottom, w, h, pixelsRGB, r, g, b);
	}

	for (i=max(0, (rect.top + y)); i<min((rect.bottom+y+1), h); i++)
	{
		drawPixel(rect.left + x, i, w, h, pixelsRGB, r, g, b);
		drawPixel(rect.right + x, i, w, h, pixelsRGB, r, g, b);
	}
}


void drawRectangleG(SimpleRect rect, int x, int y, byte *pixels, int w, int h, unsigned char val)
{
	int i;


	for (i=max(0, (rect.left + x)); i<min((rect.right+x), w-1); i++)
	{
		drawPixel(i, y+rect.top, w, h, pixels, val);
		drawPixel(i, y+rect.bottom, w, h, pixels, val);
	}

	for (i=max(0, (rect.top + y)); i<min((rect.bottom+y), h-1); i++)
	{
		drawPixel(rect.left + x, i, w, h, pixels, val);
		drawPixel(rect.right + x, i, w, h, pixels, val);
	}
}


void drawPixel(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b)
{
	int l;

	if (x<w && y<h && x>0 && y>0)
	{
		l = 3*(h-y)*w + x*3;
		RGBPixels[l]   = (unsigned char)r;
		RGBPixels[l+1] = (unsigned char)g;
		RGBPixels[l+2] = (unsigned char)b;
	}
}

void drawPixel(int x, int y, int w, int h, byte *pixels, int val)
{
	if (x<w && y<h && x>0 && y>0)
		pixels[x+y*w]   = (byte)val;
}

void drawLine(int xa, int ya, int xb, int yb, int w, int h, byte *pixels, int val)
{
	double arc=0;
	double wx, wy, div;
	double l;
	int x1, x2, y1, y2, sgnx=1, sgny=1, sgn=1;

	xa = max(0, xa);
	xb = max(0, xb);
	ya = max(0, ya);
	yb = max(0, yb);

	xa = min(w-1, xa);
	xb = min(w-1, xb);
	ya = min(h-1, ya);
	yb = min(h-1, yb);


	x1 = min(xa, xb);
	x2 = max(xa, xb);
	y1 = min(ya, yb);
	y2 = max(ya, yb);
	
	wx = x1 - x2;
	wy = y1 - y2;
	
	if (wx==0) return;
	div = wy/wx;

	if (xb>xa) sgnx = -1;
	if (yb>ya) sgny = -1;

	arc = atan(div); 

	if ((sgnx==1 && sgny==1) ||
	   (sgnx==-1 && sgny==-1)) ;
	 else sgn = -1;
		 
	
	for (l=x1; l<x2; l++)
	{
		if (sgn==-1) wy = (x2-l) * tan(arc); 
		  else wy = (l-x1) * tan(arc);

		drawPixel((int)l, y1+(int)wy, w, h, pixels, val);
	}

	
	for (l=y1; l<y2; l++)
	{
		if (sgn==-1) wx = (y2-l) / tan(arc);
		  else wx = (l-y1) / tan(arc);
		
		drawPixel(x1+(int)wx, (int)l, w, h, pixels, val);
	}

}


void drawLine(int xa, int ya, int xb, int yb, int w, int h, byte *pixels, unsigned char r, unsigned char g,
				 unsigned char b)
{
	double arc=0;
	double wx, wy, div;
	double l;
	int x1, x2, y1, y2, sgnx=1, sgny=1, sgn=1;

	x1 = min(xa, xb);
	x2 = max(xa, xb);
	y1 = min(ya, yb);
	y2 = max(ya, yb);
	
	wx = x1 - x2;
	wy = y1 - y2;
	
	div = wy/wx;

	if (xb>xa) sgnx = -1;
	if (yb>ya) sgny = -1;

	arc = atan(div); 

	if ((sgnx==1 && sgny==1) ||
	   (sgnx==-1 && sgny==-1)) ;
	 else sgn = -1;
		 
	
	for (l=x1; l<x2; l++)
	{
		if (sgn==-1) wy = (x2-l) * tan(arc); 
		  else wy = (l-x1) * tan(arc);

		drawPixel((int)l, y1+(int)wy, w, h, pixels, r, g, b);
	}

	
	for (l=y1; l<y2; l++)
	{
		if (sgn==-1) wx = (y2-l) / tan(arc);
		  else wx = (l-y1) / tan(arc);
		
		drawPixel(x1+(int)wx, (int)l, w, h, pixels, r, g, b);
	}

}

SimpleRect drawFaceBoundingRotated(int x, int xres, int y, int yres, int width, int height, 
							 unsigned char* pixels, double angle, int r, int g, int b, bool paint)
{
	SimpleRect boundingBox;

	double sin_a,cos_a,Sxd,Syd,S2x,S2y;
	int x1, y1, x2, y2, x3, y3;

	sin_a=sin(angle);  	  
    cos_a=cos(angle);

	Sxd=x+0.5*(cos_a*(xres-1)-sin_a*(yres-1));
    Syd=y+0.5*(cos_a*(yres-1)+sin_a*(xres-1));	  	  
    S2x=(xres-1)/2.0;
    S2y=(yres-1)/2.0;	

    x1=(int)(cos_a*(xres-1-S2x)+sin_a*S2y+Sxd);
    y1=(int)(sin_a*(xres-1-S2x)-cos_a*S2y+Syd);

	if (paint) drawLine(x, y, x1, y1, width, height, pixels, (byte)b, (byte)g, (byte)r);

	x2=(int)(cos_a*(xres-1-S2x)-sin_a*(yres-1-S2y)+Sxd);
  	y2=(int)(sin_a*(xres-1-S2x)+cos_a*(yres-1-S2y)+Syd);
	
    if (paint) drawLine(x1, y1, x2, y2, width, height, pixels, (byte)b, (byte)g, (byte)r);
		  		  
    x3=(int)(-cos_a*S2x-sin_a*(yres-1-S2y)+Sxd);
	y3=(int)(-sin_a*S2x+cos_a*(yres-1-S2y)+Syd);
		  
	if (paint)drawLine(x2, y2, x3, y3, width, height, pixels, (byte)b, (byte)g, (byte)r);
	if (paint)drawLine(x, y, x3, y3, width, height, pixels, (byte)b, (byte)g, (byte)r);

	boundingBox.top = max(0, min(y, y1) - 5);
	boundingBox.left = max(0, min(x, x3) - 5);
	boundingBox.bottom = min(max(y2, y3) + 5, height-1);
	boundingBox.right = min(max(x2, x1) + 5, width-1);


	return boundingBox;
}

void drawSmallCross(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b)
{
	drawPixel(x, y, w, h, RGBPixels, r, g, b);
	drawPixel(x+1, y, w, h, RGBPixels, r, g, b);
	drawPixel(x+2, y, w, h, RGBPixels, r, g, b);
	drawPixel(x+3, y, w, h, RGBPixels, r, g, b);
	drawPixel(x, y-1, w, h, RGBPixels, r, g, b);
	drawPixel(x, y-2, w, h, RGBPixels, r, g, b);
	drawPixel(x, y-3, w, h, RGBPixels, r, g, b);
	drawPixel(x-1, y, w, h, RGBPixels, r, g, b);
	drawPixel(x-2, y, w, h, RGBPixels, r, g, b);
	drawPixel(x-3, y, w, h, RGBPixels, r, g, b);
	drawPixel(x, y+1, w, h, RGBPixels, r, g, b);
	drawPixel(x, y+2, w, h, RGBPixels, r, g, b);
	drawPixel(x, y+3, w, h, RGBPixels, r, g, b);
}

void drawCross(int x, int y, int w, int h, byte *RGBPixels, int r, int g, int b)
{
	drawSmallCross(x, y, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-1, y-1, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-2, y-2, w, h, RGBPixels, r, g, b);
	drawSmallCross(x, y-1, w, h, RGBPixels, r, g, b);
	drawSmallCross(x, y-2, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-1, y, w, h, RGBPixels, r, g, b);
	drawSmallCross(x-2, y, w, h, RGBPixels, r, g, b);
}



void drawCrossSimple(int x, int y, int w, int h, byte* pixels,byte val)
{
	drawPixel(x, y, w, h, pixels, val);
	drawPixel(x-1, y, w, h, pixels, val);
	drawPixel(x-2, y, w, h, pixels, val);
	drawPixel(x-3, y, w, h, pixels, val);
	drawPixel(x+1, y, w, h, pixels, val);
	drawPixel(x+2, y, w, h, pixels, val);
	drawPixel(x+3, y, w, h, pixels, val);
	drawPixel(x, y-1, w, h, pixels, val);
	drawPixel(x, y-2, w, h, pixels, val);
	drawPixel(x, y-3, w, h, pixels, val);
	drawPixel(x, y+1, w, h, pixels, val);
	drawPixel(x, y+2, w, h, pixels, val);
	drawPixel(x, y+3, w, h, pixels, val);

}

void copyRGBToGrayscale(byte *dest, byte *RGBSource, int w, int h)
{
	int x, y, diff;
	byte *ptrs, *ptrd;
	for (y=0; y<h; y++)
	{
	  diff = h-y-1;
	  
	  ptrs = RGBSource + y*w*3;
	  ptrd = dest      + diff*w;

	  for (x=0; x<w; x++)
 	  {
//		*ptrd = (0.11*(*ptrs)+0.30*(*(ptrs+1))+.59*(*(ptrs+2)));
		*ptrd = ((*ptrs)+*(ptrs+1)+*(ptrs+2))/3;
		ptrs+=3;
		ptrd+=1;
	  }
	}

}

void copyRGBToGrayscaleWithoutFlip(byte *dest, byte *RGBSource, int w, int h)
{
	int x, y, diff;
	byte *ptrs, *ptrd;
	for (y=0; y<h; y++)
	{
	  diff = y;
	  
	  ptrs = RGBSource + y*w*3;
	  ptrd = dest      + diff*w;

	  for (x=0; x<w; x++)
 	  {
//		*ptrd = (0.11*(*ptrs)+0.30*(*(ptrs+1))+.59*(*(ptrs+2)));
		*ptrd = ((*ptrs)+*(ptrs+1)+*(ptrs+2))/3;
		ptrs+=3;
		ptrd+=1;
	  }
	}

}

void copyRGBToGrayscale(byte *dest, byte *RGBSource, int w, int h, int extra)
{
	int x, y, diff;
	byte *ptrs, *ptrd;
	for (y=0; y<h; y++)
	{
	  diff = h-y-1;
	  
	  ptrs = RGBSource + y*(w+extra)*3;
	  ptrd = dest      + diff*w;

	  for (x=0; x<w; x++)
 	  {
//		*ptrd = (0.11*(*ptrs)+0.30*(*(ptrs+1))+.59*(*(ptrs+2)));
		*ptrd = ((*ptrs)+*(ptrs+1)+*(ptrs+2))/3;
		ptrs+=3;
		ptrd+=1;
	  }
	}

}

void copyRGBToGrayscale(float *dest, byte *RGBSource, int w, int h)
{
	int x, y, diff;
	byte *ptrs;
	float *ptrd;
	for (y=0; y<h; y++)
	{
	  diff = h-y-1;
	  
	  ptrs = RGBSource + y*w*3;
	  ptrd = dest      + diff*w;

	  for (x=0; x<w; x++)
 	  {
   
//		*ptrd = 0.114*(*ptrs)+0.587*(*(ptrs+1))+0.299*(*(ptrs+2));
		*ptrd = (float)(((*ptrs)+*(ptrs+1)+*(ptrs+2))*0.33);
		ptrs+=3;
		ptrd+=1;
	  }
	}

}

// extraWidth must be difference between w and closest number that can be divided by 8
void copyGrayscaleToRGB(byte *RGBdest, byte *source, int w, int h, int extraWidth)
{
	int x, y;
	byte *ptrs, *ptrd;
	for (y=0; y<h; y++)
	{
	  ptrs = source + (h-y-1)*w;
	  ptrd = RGBdest + y*(w+extraWidth)*3;
	  for (x=0; x<w; x++)
 	  {
		*ptrd = *ptrs;
		*(ptrd+1) = *ptrs;
		*(ptrd+2) = *ptrs;
		ptrd+=3;
		ptrs+=1;
	  }
	
	  for (x=0; x<extraWidth; x++)
	  {
		  *ptrd = 0;
		  *(ptrd+1) = 0;
		  *(ptrd+2) = 0;
		  ptrd+=3;
	  }

	}


}

void flipImageVertical(unsigned char *rgb_image, int w, int h, int bpp)
{
	unsigned char *tmp = new unsigned char[w*h*bpp];
	memcpy(tmp, rgb_image, w*h*bpp);

	for (int y=0; y<h; y++)
		memcpy(rgb_image + y*w*bpp, tmp + (h-1-y)*w*bpp, w*bpp); 

	delete []tmp;
}


void convertRGBToBR(unsigned char *br_image, unsigned char *rgb_image, int w, int h)
{
	int wh = w*h;
	int R, G, B ,sum;

	memset(br_image, 0, w*h*2);
	for (int i=0; i<wh; i++)
	{
		R = *rgb_image;
		G = *(rgb_image+1);
		B = *(rgb_image+2);

		sum = R+G+B;

		if (sum==0)
		{
			rgb_image+=3;
			br_image+=2;
			continue;
		}

		*br_image = (G*255)/(R+G+B);
		*(br_image+1) = (B*255)/(R+G+B);

		rgb_image+=3;
		br_image+=2;
	}
}

void convertRGBToBRY(unsigned char *bry_image, unsigned char *rgb_image, int w, int h)
{
	int wh = w*h;
	int R, G, B ,sum;

	memset(bry_image, 0, w*h*3);
	for (int i=0; i<wh; i++)
	{
		R = *rgb_image;
		G = *(rgb_image+1);
		B = *(rgb_image+2);

		sum = R+G+B;

		if (sum==0)
		{
			rgb_image+=3;
			bry_image+=3;
			continue;
		}

		*bry_image = (G*255)/(R+G+B);
		*(bry_image+1) = (B*255)/(R+G+B);
		*(bry_image+2) = (R+G+B)*0.33;

		rgb_image+=3;
		bry_image+=3;
	}
}

float rectangleIntersectionRatio(const SimpleRect &r1, const SimpleRect &r2)
{
	float intersectionRatio1 = 0, intersectionRatio2 = 0;
	SimpleRect intersectingRect;
	int field1, field2, intersectionField;

	if (!rectangleIntersection(r1, r2)) return 0.0f;

	field1 = (r1.right - r1.left) * (r1.bottom - r1.top);
	field2 = (r2.right - r2.left) * (r2.bottom - r2.top);
	
	intersectingRect.left = max(r1.left, r2.left);
	intersectingRect.right = min(r1.right, r2.right);
	intersectingRect.top = max(r1.top, r2.top);
	intersectingRect.bottom = min(r1.bottom, r2.bottom);
	
	intersectionField = (intersectingRect.right - intersectingRect.left) * (intersectingRect.bottom - intersectingRect.top);
	
	intersectionRatio1 = min((float)intersectionField/(float)field1, field1/(float)intersectionField);
	intersectionRatio2 = min((float)intersectionField/(float)field2, field2/(float)intersectionField);

	return min(intersectionRatio1, intersectionRatio2);
}


bool rectangleIntersection(const SimpleRect &r1, const SimpleRect &r2)
{
     return !((r1.left) > (r2.right) ||
         (r1.bottom) < (r2.top) ||
         (r1.right) < (r2.left) ||
         (r1.top) > (r2.bottom));

}



void rotatePoint(float angle, int x0, int y0, int &x, int &y)
{
	int x2, y2;
	x2 = (int)(cos(angle)*(float)(x-x0) - sin(angle)*(float)(y-y0)) + x0;
	y2 = (int)(sin(angle)*(float)(x-x0) + cos(angle)*(float)(y-y0)) + y0;

	x = x2;
	y = y2;
	     	
}

void rotateImage(float angle, int x0, int y0, int iWidth, int iHeight, unsigned char *pbData, 
				 unsigned char **pbDataRows)
{
   byte *pbRotatedImage = new byte[iWidth*iHeight];
   byte **pbRotatedImageRows = new byte*[iHeight];

   memset(pbRotatedImage, 0, iWidth*iHeight);

   int i;
   int x2, y2;

   for (i=0; i<iHeight; i++)
	   pbRotatedImageRows[i] = pbRotatedImage + i*iWidth;
                   
   for (int x1 = 0; x1 < iWidth; x1++)
      for (int y1=0; y1 < iHeight; y1++)
      {
         x2 = (int)(cos(angle)*(x1-x0) - sin(angle)*(y1-y0)+x0);
         y2 = (int)(sin(angle)*(x1-x0) + cos(angle)*(y1-y0)+y0);	 
    	      
         if (x2>-1 && x2<iWidth && y2>-1 && y2<iHeight)
    	        pbRotatedImageRows[y1][x1] = pbDataRows[y2][x2];
      }

   memcpy(pbData, pbRotatedImage, iWidth*iHeight);
   delete []pbRotatedImage;
   delete []pbRotatedImageRows;
}

// returns specified rectangle area for the image
// x, y - anchoring point
// w, h - size of the new image 
// 
unsigned char* cutImage(int x, int y, int w, int h, unsigned char *data, 
				   int &iWidth, int &iHeight, bool deleteData)
{
   int i,j;
   byte *newData = new byte[w*h];

   memset(newData, 0, w*h);

   byte *ptr1, *ptr2;

   ptr1 = newData;

   for (i = 0; i < h; i++)
   {
     if ((y+i)>=iHeight || (y+i)<=0) continue;
     ptr2 = data + x + (i+y)*iWidth;

	 if ((i+y)<iHeight && (i+y)>=0)
       for (j = 0; j < w; j++)
	   {
		 if ((j+x)>=0 && (j+x)<iWidth) *ptr1 = *ptr2;
		 ptr1++;
		 ptr2++;
	   }
   }

   iWidth = w;
   iHeight = h;
   if (deleteData) delete []data;
   return newData;
}

unsigned char *scaleImage2(double scale, unsigned char *data, int &iWidth, int &iHeight)
{
   int iWOrg = iWidth;
   iWidth = (int)((double)iWidth/scale);
   iHeight = (int)((double)iHeight/scale);

   double countX=0, countY=0;
   
   byte *newData = new byte[iWidth*iHeight];
   byte *ptr1, *ptr2;    

   ptr1 = newData;
   ptr2 = data;
   for (int j=0; j<iHeight; j++)
   {
	   for (int i=0; i<iWidth; i++)
	   {
			*ptr1 = *ptr2;
			ptr1++;
			ptr2 = data + int(countX) + (int(countY)) * iWOrg;
			countX+=scale;
	   }
	 countY+=scale;
	 countX=0;
   }
   	 

 //  delete []data;

   return newData;
}

unsigned char *scaleImage(unsigned char *data, int &iWidth, int &iHeight)
{
   unsigned char *temp = new unsigned char[SIZE_X * SIZE_Y];
   int x, y, sum, i, j, n, m;
   float nel;
   float scaleX = (float)iWidth/(float)SIZE_X;
   float scaleY = (float)iHeight/(float)SIZE_Y;
  
   unsigned char *pos;

   for (x=0; x<SIZE_X; x++)
     for (y=0; y<SIZE_Y; y++)
     {
       sum = 0;
       nel = 0;
    
       for (j=0; j<max(1, scaleY); j++)
       {
         n = (int)(y*scaleY)+j;
         m = (int)(x*scaleX);

         if (n>=iHeight)
            n = iHeight-1;

         pos = data + n*iWidth; 

         for (i=0; i<max(1, scaleX); i++)
         {
           if (m>=iWidth) m = iWidth-1;
           sum+=(*(pos+m));
           m++;
           nel++;

         }
       }
       temp[x + y*SIZE_X] = (unsigned char)((float)sum/nel);
     }

   iWidth = SIZE_X;
   iHeight = SIZE_Y;
   return temp;
}


void scaleImage(double scale, unsigned char *data, int &iWidth, int &iHeight, unsigned char *newData)
{
   int iWOrg = iWidth;
   iWidth = (int)((double)iWidth/scale);
   iHeight = (int)((double)iHeight/scale);

   double countX=0, countY=0;
   
// byte *newData = new byte[iWidth*iHeight];
   byte *ptr1, *ptr2;    

   ptr1 = newData;
   ptr2 = data;
   for (int j=0; j<iHeight; j++)
   {
	   for (int i=0; i<iWidth; i++)
	   {
			*ptr1 = *ptr2;
			ptr1++;
			ptr2 = data + int(countX) + (int(countY)) * iWOrg;
			countX+=scale;
	   }
	 countY+=scale;
	 countX=0;
   }
   	 

   delete []data;

// return newData;
}

int* buildHistogram(byte *image, int w, int h)
{
	int *histogram = new int[256];
	int i, j;

	for (i=0; i<256; i++)
		histogram[i] = 0;

	for (i=0; i<w; i++)
		for (j=0; j<h; j++)
			histogram[image[i + j*w]]++;

	return histogram;

}

void equalizeHistogram(byte *image, int w, int h)
{
	int *histogram, *cumulative;
	int i, sum, tempval;
	byte *levels;
	int min_val, max_val;

	cumulative = new int[256];
	levels = new byte[256];
	histogram = buildHistogram(image, w, h);
	min_val = 0;
	for (i=0; i<=255; i++)
		if (histogram[i]>0) 
		{
			min_val = histogram[i];
			break;
		}
	sum = 0;
	for (i=0; i<=255; i++)
	{
		sum+=histogram[i];
		tempval = ((sum - min_val)/(double)(w*h - min_val)) * 255.0; 
		tempval = min(tempval, 255);
		tempval = max(tempval, 0);

		levels[i] = (byte)tempval;
	}	

	for (i=0; i<w*h; i++)
		image[i] = levels[image[i]];
 
	delete []cumulative;
	delete []levels;
	delete []histogram;
	return;
}
/*
int kMeans2(Feature *f, int nElements, int iRectangleSize, float fScaleRatio,
			int nScales)
{
    int i, j, scale, sumx, sumy, meanx, meany;
    int actualI;	
    int nCenters = 0, iAttached, iCentersUpToThisScale=0;
    bool changeMade = true, fFound = true;
    int minDistance, actualDistance, iX, iY;
    float x, y, fRectangleSize, fRectangleSize2;

    int *labels = new int[nElements];

    int *iScaleStart = new int[nScales];
    int *iScaleWidth = new int[nScales];
    int iScale = -1;


    fRectangleSize = (float)iRectangleSize;

    if (nElements==0) return 0;

    Feature *centers = new Feature[nElements];

    i=0;

    for (scale=0; scale<nScales; scale++)
    {
        nCenters = 0;
        sumx=0; sumy=0; iAttached=0; fRectangleSize2 = fRectangleSize/2;

        while (f[i].Sn==scale && i<nElements)
        {
            x = (float)(f[i].X);
            y = (float)(f[i].Y);
            actualI = iCentersUpToThisScale + nCenters;

            iX = (int)(floor(x/fRectangleSize)*fRectangleSize + fRectangleSize2);
            iY = (int)(floor(y/fRectangleSize)*fRectangleSize + fRectangleSize2);

            fFound = false;
            for (j=iCentersUpToThisScale; j<(iCentersUpToThisScale+nCenters); j++)
                if (centers[j].X == iX && centers[j].Y == iY)
                {
                    j = iCentersUpToThisScale+nCenters;
                    fFound = true;
                }

                if (fFound==false) 
                {
                    centers[actualI].X = iX;
                    centers[actualI].Y = iY;
                    centers[actualI].Sn = scale;
                    centers[actualI].F = false;
                    centers[actualI].S = f[i].S;
                    nCenters++;
                }

                i++;

        }

        iCentersUpToThisScale+=nCenters;
        fRectangleSize*=fScaleRatio;

    }

    nCenters = iCentersUpToThisScale;

    for (i=0; i<nCenters; i++)
    {
        if (centers[i].Sn!=iScale)
        {
            iScale = centers[i].Sn;
            iScaleStart[iScale] = i;
            iScaleWidth[iScale] = 0;
        }
        iScaleWidth[iScale]++;
    }

    while (changeMade)
    {
        for (i=0; i<nElements; i++)
        {
            iX = f[i].X;
            iY = f[i].Y;
            iScale = f[i].Sn;
            minDistance=99999999;

            for (j=iScaleStart[iScale]; j<(iScaleStart[iScale]+iScaleWidth[iScale]); j++)
                if ((actualDistance=(abs(iX-centers[j].X) + abs(iY-centers[j].Y)))<minDistance)
                {
                    minDistance = actualDistance;
                    labels[i] = j;
                }
        }

        changeMade = false;

        for(i=0; i<nCenters; i++)
        {
            iAttached = 0;
            sumx = 0;  sumy = 0;
            for (j=0; j<nElements; j++)
                if (labels[j]==i)
                {
                    iAttached++;
                    sumx+=f[j].X; 
                    sumy+=f[j].Y;
                }

                if (iAttached!=0)
                {
                    meanx = sumx/iAttached;
                    meany = sumy/iAttached;

                    if (meanx!=centers[i].X || meany!=centers[i].Y)
                    {
                        changeMade = true;
                        centers[i].X = meanx;
                        centers[i].Y = meany;
                    }
                }
        }
    }

    memcpy(f, centers, nCenters*sizeof(Feature));

	delete []iScaleStart;
	delete []iScaleWidth;

	delete []centers;
    delete []labels;
    return nCenters;
}
*/
int distance(Feature f1, Feature f2)
{
    return (int)sqrt(float((f1.X-f2.X)*(f1.X-f2.X) + (f1.Y-f2.Y)*(f1.Y-f2.Y)));
}

double distanceExact(SimplePointD &p1, SimplePointD &p2)
{
	return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
}

float distanceExact(Feature &f1, Feature &f2)
{
    return sqrt(float((f1.X-f2.X)*(f1.X-f2.X) + (f1.Y-f2.Y)*(f1.Y-f2.Y)));
}


int distance2(Feature f1, Feature f2)
{
    return (f1.X-f2.X)*(f1.X-f2.X) + (f1.Y-f2.Y)*(f1.Y-f2.Y);
}

int distance(int x1, int y1, int x2, int y2)
{
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

float distanceExact(int x1, int y1, int x2, int y2)
{
    return sqrt((float)((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)));
}

float distanceExact(float x1, float y1, float x2, float y2)
{
    return sqrt(((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)));
}

float distance(float x1, float y1, float x2, float y2)
{
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}



/*
int kMeans(Feature *f, int nElements, int iRectangleSize, int iWidth, int iHeight, float fScaleRatio,
		   int nScales)
{
    int i, j, scale, k, sumx, sumy, meanx, meany;	
    int nCenters = 0, iAttached;
    bool changeMade = true;
    byte flag=0;
    int minDistance, actualDistance;
    float s=0;

    if (nElements==0) return 0;

    Feature *tempCenters = new Feature[nElements];

	for (scale=0; scale<nScales; scale++)
	{
		for (i=0; i<iWidth; i+=iRectangleSize)
			for (j=0; j<iHeight; j+=iRectangleSize)
			{
				sumx=0; sumy=0; iAttached = 0;

				for (k=0; k<nElements; k++)
					if (f[k].Sn==scale && f[k].X>i && f[k].X<(i+iRectangleSize)
						&& f[k].Y>j && f[k].Y<(j+iRectangleSize))
					{
						iAttached++;
						sumx+=f[k].X;
						sumy+=f[k].Y;
						s = f[k].S;
						flag = f[k].F;
					}

					if (sumx>0 && sumy>0)
					{
						tempCenters[nCenters].X = sumx/iAttached;
						tempCenters[nCenters].Y = sumy/iAttached;
						tempCenters[nCenters].Sn = scale;
						tempCenters[nCenters].S = s;
						if (flag==1)
							tempCenters[nCenters].F = true;
						else tempCenters[nCenters].F = false;

						nCenters++;
					}
			}
			// for the moment without scaling (for comparison)
			iRectangleSize=(int)(iRectangleSize*fScaleRatio);
	}

    int *labels = new int[nElements];
    Feature *centers = new Feature[nCenters];
    for (i=0; i<nCenters; i++) centers[i]=tempCenters[i];

    delete []tempCenters;

    while (changeMade)
    {
        for (i=0; i<nElements; i++)
        {
            minDistance=99999999;
            for (j=0; j<nCenters; j++)
                if ((f[i].Sn==centers[j].Sn) &&
                    (actualDistance=distance(f[i].X, f[i].Y, centers[j].X, centers[j].Y))<minDistance)
                {
                    minDistance = actualDistance;
                    labels[i] = j;
                }
        }

        changeMade = false;

        for(i=0; i<nCenters; i++)
        {
            iAttached = 0;
            sumx = 0;  sumy = 0;
            for (j=0; j<nElements; j++)
                if (labels[j]==i)
                {
                    iAttached++;
                    sumx+=f[j].X; 
                    sumy+=f[j].Y;
                }

                if (iAttached!=0)
                {
                    meanx = sumx/iAttached;
                    meany = sumy/iAttached;

                    if (meanx!=centers[i].X || meany!=centers[i].Y)
                    {
                        changeMade = true;
                        centers[i].X = meanx;
                        centers[i].Y = meany;
                    }
                }
        }
    }

    for (i=0; i<nCenters; i++) f[i] = centers[i];

    delete []centers;
    delete []labels;
    return nCenters;
}
*/
void findIntegralImage(unsigned char *img,int xres,int yres,int **iimage)
{		
    int y_pos, y;

    for(int x=0;x<=xres;x++) iimage[0][x]=0;
    for(y=0;y<=yres;y++) iimage[y][0]=0;		
    for(y=1;y<=yres;y++) {
        y_pos=(y-1)*xres;			
        for(int x=1;x<=xres;x++) iimage[y][x]=iimage[y][x-1]+iimage[y-1][x]-iimage[y-1][x-1]+img[x-1+y_pos];
    }		

}

void findIntegralImage(unsigned char *img,int xres,int yres,int *iimage)
{		
	int y_pos, y;
	int *iiy, *iiym1;
	int iiw = xres+1;
	int iih = yres+1;

	for(int x=0;x<=xres;x++) iimage[x]=0;
	for(y=0;y<=yres;y++)  	 iimage[y*iiw]=0;

	iiy = iimage + xres + 1;
	iiym1 = iiy - iiw;

	for(y=1;y<=yres;y++) {
		y_pos=(y-1)*xres;	

		for(int x=1;x<=xres;x++) 
			iiy[x]=iiy[x-1]+iiym1[x]-iiym1[x-1]+img[x-1+y_pos];

		iiy+=iiw;
		iiym1+=iiw;	
	}

}

void findIntegralImage(int *img,int xres,int yres,int *iimage)
{		
	int y_pos, y;
	int *iiy, *iiym1;
	int iiw = xres+1;
	int iih = yres+1;

	for(int x=0;x<=xres;x++) iimage[x]=0;
	for(y=0;y<=yres;y++)  	 iimage[y*iiw]=0;

	iiy = iimage + xres + 1;
	iiym1 = iiy - iiw;

	for(y=1;y<=yres;y++) {
		y_pos=(y-1)*xres;	

		for(int x=1;x<=xres;x++) 
			iiy[x]=iiy[x-1]+iiym1[x]-iiym1[x-1]+img[x-1+y_pos];

		iiy+=iiw;
		iiym1+=iiw;	
	}

}

void findRotatedIntegralImage(unsigned char *img,int xres,int yres,int *iimage)
{		
	int y_pos, y;
	int *iiy, *iiym1;
	int iiw = xres+2;
	int iih = yres+1;

	for(int x=0;x<=(xres+1);x++) iimage[x]=0;
	for(y=0;y<=yres;y++)  	
	{
		iimage[y*iiw]=0;
		iimage[y*iiw+1]=0;
	}

	for (int y=1; y<iih; y++)
		for (int x=2; x<iiw; x++)
		{
			iimage[x + y*iiw] = iimage[x-1 + (y-1)*iiw] + iimage[x-1 + y*iiw] +
				img[x-2 + (y-1) * xres] - iimage[x-2 + (y-1) * iiw];
		}

	for (int y=(iih-2); y>0; y--)
		for (int x=(iiw-1); x>1; x--)
		{
			iimage[x + y*iiw]+=iimage[x-1 + (y+1)*iiw] - iimage[x-2 + y*iiw];
		}
}

void findRotatedIntegralImage(int *img,int xres,int yres,int *iimage)
{		
	int y_pos, y;
	int *iiy, *iiym1;
	int iiw = xres+2;
	int iih = yres+1;

	for(int x=0;x<=(xres+1);x++) iimage[x]=0;
	for(y=0;y<=yres;y++)  	
	{
		iimage[y*iiw]=0;
		iimage[y*iiw+1]=0;
	}

	for (int y=1; y<iih; y++)
		for (int x=2; x<iiw; x++)
		{
			iimage[x + y*iiw] = iimage[x-1 + (y-1)*iiw] + iimage[x-1 + y*iiw] +
				img[x-2 + (y-1) * xres] - iimage[x-2 + (y-1) * iiw];
		}

	for (int y=(iih-2); y>0; y--)
		for (int x=(iiw-1); x>1; x--)
		{
			iimage[x + y*iiw]+=iimage[x-1 + (y+1)*iiw] - iimage[x-2 + y*iiw];
		}
}

void findIntegralImage(unsigned char *img,int xres,int yres,unsigned int *iimage)
{		
	int y_pos, y;
	unsigned int *iiy, *iiym1;
	int iiw = xres+1;
	int iih = yres+1;

	for(int x=0;x<=xres;x++) iimage[x]=0;
	for(y=0;y<=yres;y++)  	 iimage[y*iiw]=0;

	iiy = iimage + xres + 1;
	iiym1 = iiy - iiw;

	for(y=1;y<=yres;y++) {
		y_pos=(y-1)*xres;	

		for(int x=1;x<=xres;x++) 
			iiy[x]=iiy[x-1]+iiym1[x]-iiym1[x-1]+img[x-1+y_pos];

		iiy+=iiw;
		iiym1+=iiw;	
	}

}

void findIntegralImage(float *img,int xres,int yres,float *iimage)
{		
	int y_pos, y;
	float *iiy, *iiym1;
	int iiw = xres+1;
	int iih = yres+1;

	for(int x=0;x<=xres;x++) iimage[x]=0.0;
	for(y=0;y<=yres;y++)  	 iimage[y*iiw]=0.0;

	iiy = iimage + xres + 1;
	iiym1 = iiy - iiw;

	for(y=1;y<=yres;y++) {
		y_pos=(y-1)*xres;	

		for(int x=1;x<=xres;x++) 
			iiy[x]=iiy[x-1]+iiym1[x]-iiym1[x-1]+img[x-1+y_pos];

		iiy+=iiw;
		iiym1+=iiw;	
	}

}

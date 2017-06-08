/*
	This is a part of the Discrete Area Filters Face Detection library, copyright to Jacek Naruniec, 2012.
	You can use this library for non-commercial use. If you are interested in commercial use, please contact
	the author of the library (Jacek Naruniec, email: jacek.naruniec@gmail.com")

	This file contains most of the structures used in the detector:
*/

#pragma once


#ifndef STRUCTS
#define STRUCTS


#include "consts.h"
#include <math.h>
#include <stdio.h>
#include <ostream>

#include <algorithm>

using namespace std;

#ifndef byte
typedef unsigned char byte;
#endif

struct SimplePoint
{
	int x, y;
};

struct SimpleRect
{
	int left;
	int right;
	int top;
	int bottom;
	int width; 
	int height;

	SimpleRect() {} 

	SimpleRect(int _left, int _top, int _width, int _height)
	{
		width = _width;
		height = _height;
		left = _left;
		right = _left + _width;
		top = _top;
		bottom = _top + height;
	}

	void crop(int l, int r, int t, int b)
	{
		left = max(l, left);  right = max(l, right);
		left = min(r, left);  right = min(r, right);
		top = max(t, top);    bottom = max(t, bottom);
		top = min(b, top);    bottom = min(b, bottom);
		width = right-left;
		height = bottom-top;
	}

	bool containPoint(SimplePoint p)
	{
		if (p.x>left && p.x<right && p.y>top && p.y<bottom)
			return true;
		else
			return false;
	}
};

struct PointWithScale
{
	PointWithScale(int _x = 0, int _y = 0, float _scale = 1.0f) : x(_x), y(_y), scale(_scale) { }
	void set(int _x, int _y, float _scale) { x = _x; y = _y; scale = _scale; }
	int x, y;
	float scale;
	string associated_filename;
	int user_value;
};


struct D3DPoint
{
	double x, y, z;

	D3DPoint() { set(0.0, 0.0, 0.0); } 

	D3DPoint(double _x, double _y, double _z)
	{ x = _x; y = _y; z = _z; }

	void set(double _x, double _y, double _z)
	{ x = _x; y = _y; z = _z; }

	D3DPoint& operator +=(D3DPoint &p)
	{	x+=p.x; y+=p.y; z+=p.z; return *this; }

	D3DPoint& operator /=(D3DPoint &p)
	{	x/=p.x; y/=p.y; z/=p.z; return *this; }

	D3DPoint& operator /=(double s)
	{	x/=s; y/=s; z/=s; return *this; }

	D3DPoint& operator *=(double s)
	{	x*=s; y*=s; z*=s; return *this; }

	D3DPoint operator -(D3DPoint &p)
	{	D3DPoint p_r; 
	p_r.x = x - p.x; p_r.y = y - p.y; p_r.z = z - p.z; 
	return p_r; }

	D3DPoint operator +(D3DPoint &p)
	{	D3DPoint p_r; 
	p_r.x = x + p.x; p_r.y = y + p.y; p_r.z = z + p.z; 
	return p_r; }

	double len()
	{
		return sqrt(x*x + y*y + z*z);
	}

};



/// Simple floating 2D point structure
struct SimplePointD
{
	float x, y;

	/// Default constructor
	SimplePointD() : x(0.0f), y(0.0f) {};

	/// Full constructor
	SimplePointD(float _x, float _y) : x(_x), y(_y){};

	SimplePointD(const SimplePoint& p) : x(p.x), y(p.y){};

	SimplePointD& operator +=(SimplePointD &p)
	{	x+=p.x; y+=p.y; return *this; }

	SimplePointD& operator /=(float s)
	{	x/=s; y/=s; return *this; }

	SimplePointD& operator *=(float s)
	{	x*=s; y*=s; return *this; }

	SimplePointD operator *(float s)
	{	SimplePointD p_r = *this; 
	p_r.x *= s; p_r.y *= s;
	return p_r; }

	SimplePointD operator -(SimplePointD &p)
	{	SimplePointD p_r; 
	p_r.x = x - p.x; p_r.y = y - p.y; 
	return p_r; }

	SimplePointD operator +(SimplePointD &p)
	{	SimplePointD p_r; 
	p_r.x = x + p.x; p_r.y = y + p.y;
	return p_r; }

	// Read http://stackoverflow.com/questions/2875985/c-no-match-for-operator-when-compiling-using-gcc
	SimplePointD& operator=(const SimplePointD& other_point)
	{
		x = other_point.x;
		y = other_point.y;
		return *this;
	}

	SimplePointD& operator=(const SimplePoint& other_point)
	{
		x = (float)other_point.x;
		y = (float)other_point.y;
		return *this;
	}

    /// Calculates euclidean distance of this point to the given point.
	float distance(const SimplePointD& toCompare) const
	{
		return sqrtf(powf(x - toCompare.x, 2.0f) + powf(y - toCompare.y, 2.0f));
	}

	/// Setting proper point values.
	void set(float _x, float _y)
	{ x = _x; y = _y; }

	friend ostream& operator<<(ostream& out, const SimplePointD& p)
	{
		out << "(" << p.x << ", " << p.y << ")";
		return out;
	}
};



struct SimpleRect2
{
	int left_top;
	int right_top;
	int left_bottom;
	int right_bottom;
};

class Point
{
public:
	int x;
	int y;
	bool set;
	int index;
	float s;

	Point();
};


class ReferenceDistances
{
public:
	int ref_point1;
	int ref_point2;

	int n_distances;
	float *distances;

	int *points1;
	int *points2;
	float *max_differences;
	float *min_dist, *max_dist;

	float max_threshold;

	ReferenceDistances();
	~ReferenceDistances();
	void createTables();
	void clearTables();

};

class Face
{
public:
	SimplePoint face_points[N_POINTS];		// if point wasn't found, (x, y) = (0, 0)
	SimplePoint facial_features[N_FACIAL_FEATURES];

	void clear();
};

class Feature
{
public:
	int X, Y;	// anchoring point
	float S;	// scale
	int Sn;		// scale index
	bool F;		// flag (no specified usage)
	int weight; // weight of the ff truncated to int
	int type;   // type of the point

	void set(int _x, int _y, float _S, int _Sn=-1, bool _F=true, int _weight=1);
	Feature();
	void operator+=(Feature &f);
	Feature& normalizeWeight();
};

class FacePart
{
public:
	Feature *features;
	int n_points;

	FacePart();
	~FacePart();

	void clear();
	SimpleRect getBounds();
	Feature calculateCenter();
	Feature calculateCenter(float scale_factor);
	void createTables(int _n_points);
	void readFromFile(FILE *f);
	void writeToFile(FILE *f);
	FacePart& operator=(FacePart &fp);
};

class Extractor
{
public:
	int type;
	int n;
	int R;
	int orientation;
	bool mirrored;

	float fScale;
	int nScale;

	SimpleRect *rects;

	Extractor();
	~Extractor();
	
	void destroyRects();
	void copy(Extractor &ex);
	void set(int _type, int _n, int _R, int _orientation);
	void write(FILE *f);
	void load(FILE *f);
};

class Mirrors
{
public:
	int mirrors[N_POINTS];

	Mirrors();

	void set();
};

struct ClassifierWindowBounds
{
	int minx;
	int miny;
	int maxx;
	int maxy;
};

// bounds from one point (p1) to another (p2)
struct Bound
{
	SimpleRect bound_rect;
	int p1;
	int p2;
};
/*
struct Statistics
{
public:
	double falseAcceptanceRate;	
	double falseRejectionRate;

	double trueAcceptanceRate;
	double trueRejectionRate;

	__int64 nFalseAcceptances;
	__int64 nFalseRejections;

	__int64 nTrueAcceptances;
	__int64 nTrueRejections;

	float pointsAccuracy[N_POINTS];
	float ffAccuracy[N_FACIAL_FEATURES];
	float faceAccuracy;

	float mean_time;

	int tag;	// additional
		
	Statistics();

	void calculateRates();
	void zero();
	void write(FILE *f);
};
*/

struct Corner
{
	int x;
	int y;
	int strength;
};

struct Edge
{
	int x, y;
	int strength;
	byte direction;
};

struct FaceCoordinates
{
	SimplePoint le, re, between_eyes_point;
	double between_eyes_distance;
};


#endif
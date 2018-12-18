#pragma once

#include "Solver.h"
#include "catpaw/cpToolBox.h"

#define TYPE_FLUID 0
#define TYPE_CLOTH 1
#define TYPE_BOUNDARY 2
#define TYPE_EMITTER 3
#define TYPE_RIGID 4
#define TYPE_DEFORMABLE 5
#define TYPE_GRANULAR 6
#define TYPE_GAS 7

#define TYPE_NULL 99

struct fluidvol {
	cfloat3 xmin;
	cfloat3 xmax;
	float volfrac[10];
	int group;
};

struct FluidSrc {
	cfloat3 srcpos;
	cfloat3 norm;
	float radius;
	float speed;
	char type;//geometry type
	int interval;
};

void loadFluidVolume(XMLElement* sceneEle, int typenum, vector<fluidvol>& fvs);


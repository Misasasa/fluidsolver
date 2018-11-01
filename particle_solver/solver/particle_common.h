#pragma once

#include "Solver.h"
#include "catpaw/cpToolBox.h"

struct fluidvol {
	cfloat3 xmin;
	cfloat3 xmax;
	float volfrac[10];
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


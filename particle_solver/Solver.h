#pragma once

#include "catpaw/vec_define.h"

class Solver {
public:
	virtual void Step()=0;
	virtual void HandleKeyEvent(char key)=0;

	vecf3 host_x;
	vecf4 host_color;
	int num_particles;

	cfloat3 domainMin;
	cfloat3 domainMax;
	float dt;
};
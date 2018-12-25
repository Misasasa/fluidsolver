#pragma once

#include "catpaw/vec_define.h"

class Solver {
public:
	virtual void Step() {
		printf("Using Base Step.\n");
	}
	virtual void HandleKeyEvent(char key) {
		printf("Using Base HandleKeyEvent.\n");
	}
	virtual void Eval(const char* expression){
		printf("Using Base Eval.\n");
	}

	vecf3 host_x;
	vecf4 host_color;
	int num_particles;

	cfloat3 domainMin;
	cfloat3 domainMax;
	float dt;
};
#pragma once

#include "catpaw/vec_define.h"

class Solver {
public:
	virtual void step()=0;
	virtual void HandleKeyEvent(char key)=0;

	vecf3 hPos;
	vecf4 hColor;
	int numP;

	cfloat3 domainMin;
	cfloat3 domainMax;
	float dt;
};
#pragma once

#include "catpaw/vec_define.h"

class Solver {
public:
	virtual void Step() {
	}
	virtual void HandleKeyEvent(char key) {
	}
	virtual void Eval(const char* expression){
	}
	
	virtual float GetTimeStep()=0;
	virtual cfloat3 GetDomainMin()=0;
	virtual cfloat3 GetDomainMax()=0;
};
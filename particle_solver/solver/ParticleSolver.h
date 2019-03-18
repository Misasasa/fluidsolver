#pragma once
#include "Solver.h"


class ParticleSolver : public Solver {
protected:
	int numParticles;
	vecf3 pos;
	vecf4 color;
	vecf3 vel;
	vecadj adjacentIndex;
	cfloat3 domainMin;
	cfloat3 domainMax;
	float dt;
	
public:
	
	vecf3* GetPos(){
		return &pos;
	};
	vecf4* GetColor() {
		return &color;
	}
	int GetNumParticles() {
		return numParticles;
	}
	float GetTimeStep() {
		return dt;
	}
	cfloat3 GetDomainMin() {
		return domainMin;
	}
	cfloat3 GetDomainMax() {
		return domainMax;
	}

};
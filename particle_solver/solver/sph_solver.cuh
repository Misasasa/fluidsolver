#pragma once

#include "catpaw/geometry.h"
#include "vector_types.h"
#include "catpaw/objbuilder.h"
#include "particle_common.h"

#define EPSILON 0.000001

#define MAX_OBJNUM 100

namespace sph{

struct SimParam_SPH {
	//particle data
	float global_relaxation;

	int maxpnum;
	int maxtypenum;
	int maxconstraintnum;
	int maxtrianglenum;
	int maxobjnum;


	float dx;
	cfloat3 gridxmin;
	cfloat3 gridxmax;
	cint3 gridres;

	cfloat3 gravity;
	float dt;

	float mass;
	float viscosity;
	float restdensity;

	//kernels
	float spacing;
	float smoothradius;
	float kpoly6;
	float kspiky;
	float kspikydiff;
	float klaplacian;
	float kernel_cubic;
	float kernel_cubic_gradient;


	//wc-sph
	float pressureK;

	//multiphase
	float densArr[10];
	float viscArr[10];
	float drift_dynamic_diffusion;
	float drift_turbulent_diffusion;

	//solid
	float solidK;
	float solidG;
	float Yield;

	//boundary
	cfloat3 softminx;
	cfloat3 softmaxx;
	float boundstiff;
	float bounddamp;
	float bmass;
	float bRestdensity;
	float bvisc;
};

struct SimData_SPH {

	//particle data
	cfloat3* pos;
	cfloat3* normal;
	cfloat3* vel;
	cfloat4* color;
	int* type;
	int* group;
	int* uniqueId;
	float* mass;
	float* density;
	float* pressure;
	cfloat3* force;
	

	cfloat3* sortedPos;
	cfloat3* sortedVel;
	cfloat3* sortedNormal;
	cfloat4* sortedColor;
	int* sortedType;
	int* sortedGroup;
	int* sortedUniqueId;
	float* sortedMass;
	int* indexTable;

	// DFSPH
	float* alpha;
	cfloat3* v_star; //predicted vel
	cfloat3* x_star; //predicted pos
	float* pstiff;
	cfloat3* sortedV_star;
	float* error;
	float* pstiff_sum;
	float* sortedPstiff_sum;

	// Multiphase Fluid
	float* vFrac;
	float* restDensity;
	cfloat3* drift_v;
	cfloat3* vol_frac_gradient;
	float* effective_mass;
	float* effective_density;
	float* phase_diffusion_lambda;
	float* vol_frac_change;

	float* sortedVFrac;
	float* sortedRestDensity;
	float* sorted_effective_mass;
	float* sorted_effective_density;


	//edge data
	//edgeConstraint* edgeCons;
	//edgeConsVar* edgeConsVar;

	//triangle data
	objtriangle* triangles;
	float* facetVol;
	float* facetArea;
	int*   facetObjId;
	cfloat3* baryCenter;
	int*	 baryCenterTriangleId;
	cfloat3* sortedBaryCenter;
	int*	 sortedBaryCenterTriangleId;

	//Sort
	int* particleHash;   //cellid of each particle
	int* particleIndex;  //particle index, sorted in process
	int* gridCellStart; //cell begin
	int* gridCellEnd;  //cell end
	int* gridCellCollisionFlag;

	int* baryCenterHash;
	int* baryCenterIndex;
	int* gridCellStartBaryCenter;
	int* gridCellEndBaryCenter;

	//Anisotropic Kernel
	cfloat3* avgpos;
	cmat3* covmat;
	cmat3* s;
	cmat3* u;
	cmat3* v;
};






void CopyParam2Device();
void CopyParamFromDevice();


void calcHash(
	SimData_SPH data,
	int numParticles
);

void sortParticle(
	SimData_SPH data,
	int numParticles
);
void reorderDataAndFindCellStart(
	SimData_SPH data,
	int numParticles,
	int numGridCells
);

void applyXSPH(
	SimData_SPH data,
	int numParticles
);

//Standard SPH
void computePressure(SimData_SPH data, int numP);
void computeForce(SimData_SPH data, int numP);
void advect(SimData_SPH data, int numP);





/******************************

            DFSPH

******************************/

void computeDensityAlpha(SimData_SPH data, int numP);

//compute non-pressure force & predict velocity
void computeNonPForce(SimData_SPH data, int numP);

//correct density error & update positions
void correctDensityError(SimData_SPH data,
	int numP,
	int maxiter,
	float ethres,
	bool bDebug
);

//correct divergence error & update velocity
void correctDivergenceError(SimData_SPH data, 
	int numP,
	int maxiter,
	float ethres,
	bool bDebug
);



/****************************************

			Multiphase SPH

****************************************/
void DFAlpha_Multiphase(SimData_SPH data, int numP);
void NonPressureForce_Multiphase(SimData_SPH data, int numP);
void EnforceDensity_Multiphase(SimData_SPH data,
	int numP,
	int maxiter,
	float ethres,
	bool bDebug
);
void EnforceDivergenceFree_Multiphase(SimData_SPH data,
	int numP,
	int maxiter,
	float ethres,
	bool bDebug
);
void EffectiveMass(SimData_SPH data, int numP);

void DriftVelocity(SimData_SPH data, int numP);

void PhaseDiffusion(SimData_SPH data, int numP);

};



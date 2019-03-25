#pragma once

#include "catpaw/geometry.h"
#include "vector_types.h"
#include "catpaw/objbuilder.h"
#include "particle_common.h"



struct SimParam_SPH {
	
	int maxpnum;
	int maxtypenum;
	int num_fluid_p;
	int num_deformable_p;


	float dx;
	cfloat3 gridxmin;
	cfloat3 gridxmax;
	cint3 gridres;

	cfloat3 gravity;
	float dt;

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
	float drift_thermal_diffusion;
	float surface_tension;
	float acceleration_limit;

	//solid
	float solidK;
	float solidG;
	float young;
	float Yield;
	float solid_visc;
	float plastic_flow;
	float dissolution; //dissolution constant factor
	float max_alpha[10]; //max volume fraction of each single phase
	float heat_flow_rate;
	float melt_point;
	float latent_heat;
	float heat_capacity[10];

	//boundary
	float boundary_visc;
	float boundary_friction;
	
	//cloth
	float kadj;
	float kdiag;
	float kbend;

	//Switch
	bool enable_dissolution;
	bool enable_melt;
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
	
	adjacent* adjacent_index;
	adjacent* sorted_adjacent_index;

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
	float* DF_factor;
	cfloat3* v_star; //predicted vel
	float* pstiff;
	cfloat3* sortedV_star;
	float* error;
	float* rho_stiff;
	float* sorted_rho_stiff;
	float* div_stiff;
	float* sorted_div_stiff;

	// Multiphase Fluid
	float* vFrac;
	float* restDensity;
	cfloat3* drift_v;
	cfloat3* vol_frac_gradient;
	float* effective_mass;
	float* effective_density;
	float* phase_diffusion_lambda;
	float* vol_frac_change;
	float* spatial_status;
	float* temperature;
	float* heat_buffer;

	// try to speed up
	cfloat3* flux_buffer;

	float* sortedVFrac;
	float* sortedRestDensity;
	float* sorted_effective_mass;
	float* sorted_temperature;
	float* sorted_heat_buffer;


	// Deformable Solid
	cmat3* strain_rate;
	cmat3* cauchy_stress;
	cmat3* gradient;
	cfloat3* vel_right;
	int* neighborlist;   //store uniqueid
	cfloat3* neighbordx; //store x0_ij
	float* length0;
	int* local_id;       //id within the same type
	cmat3* correct_kernel;
	cfloat3* x0;         //initial coordinates
	cmat3* rotation;
	cmat3* Fp;			//plastic part of deformation gradient
	int* trim_tag;
	float* spatial_color; //for cutting

	//used for iteration
	cfloat3* r_this;
	cfloat3* r_last;
	cfloat3* p_this;

	cmat3* sorted_cauchy_stress;
	cmat3* sorted_gradient;
	int*   sorted_local_id;


	// Grid Sort
	int* particleHash;   //cellid of each particle
	int* particleIndex;  //particle index, sorted in process
	int* gridCellStart; //cell begin
	int* gridCellEnd;  //cell end
	int* gridCellCollisionFlag;

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


//Standard SPH
void ComputePressure(SimData_SPH data, int numP);
void computeForce(SimData_SPH data, int numP);
void Advect(SimData_SPH data, int numP);





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
void DFSPHFactor_Multiphase(SimData_SPH data, int num_particles);
void NonPressureForce_Multiphase(SimData_SPH data, int num_particles);
void EnforceDensity_Multiphase(SimData_SPH data,
	int num_particles,
	int maxiter,
	float ethres_avg,
	float ethres_max,
	bool bDebug,
	bool warm_start
);
void EnforceDivergenceFree_Multiphase(SimData_SPH data,
	int num_particles,
	int maxiter,
	float ethres,
	bool bDebug,
	bool warm_start
);
void EffectiveMass(SimData_SPH data, int num_particles);
void DriftVelocity(SimData_SPH data, int num_particles);
void PhaseDiffusion(SimData_SPH data, int num_particles);
void PhaseDiffusion(SimData_SPH data, int num_particles, float* dbg, int frameNo);

void RigidParticleVolume(SimData_SPH data, int num_particles);
void InitializeDeformable(SimData_SPH data, int num_particles);

void MoveConstraintBoxAway(SimData_SPH data, int num_particles);
void DetectDispersedParticles(SimData_SPH data, int num_particles);


/*
Compare with Ren's method.
*/

void ComputeForceMultiphase(SimData_SPH data, int num_p);
void DriftVel_Ren(SimData_SPH data, int num_p);
void PhaseDiffusion_Ren(SimData_SPH data, int num_p);



/**********      Solid       ***********/
void ComputeTension(SimData_SPH data, int num_particles);
void UpdateSolidState(SimData_SPH data, int num_particles, int projection_type);
void UpdateSolidTopology(SimData_SPH data, int num_particles);

void HeatConduction(SimData_SPH data, int num_particles);

void AdvectScriptObject(SimData_SPH data, int num_particles, cfloat3 vel);


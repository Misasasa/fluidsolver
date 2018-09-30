
#ifndef PARAM
#define PARAM

#include "catpaw/geometry.h"

struct SimParam {

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
	float densArr[10];
	float viscArr[10];

	//kernels
	float spacing;
	float smoothradius;
	float kpoly6;
	float kspiky;
	float kspikydiff;
	float klaplacian;
	float kspline;


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

	//==================
	//      PBF
	//===================
	float pbfepsilon;
	float qfactor;
	float k_ti; //tensile instability
	float n_ti;
	float vorticityfactor;

	//surface tension
	float pairwise_k;
	float pairwise_c;
	float surface_threshold;
	float surface_stiff;

	//anisotropic kernel
	float dens_thres;
	float avgpos_k;
	float aniso_support;
	float aniso_thres;
	float aniso_thres_coplanar;




	//==================
	//      PBD CLOTH
	//===================
	float stretchStiff;
	float compressStiff;
	float bendingstiff;
	float volumeStiff;
	float k_damping;
	float stretchComp;
	float compressComp;
	
	//volume pressure
	float restvol;
	float pE;
	float resistance;

	float collisionDistance;
	float overlap;
	float selfcollisionDistance;
	float cloth_selfcd;
	float clothThickness;
	float collisionStiff;

	//porosity model
	float diffuseDistance;
	float k_absorb;
	float k_diffuse;
	float k_diffuse_gravity;
	float cloth_porosity;
	float cloth_density;
	float max_saturation;
	float k_dripBuf;
	float emitThres;
};


#endif
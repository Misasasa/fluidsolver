#include "catpaw/geometry.h"
#include "vector_types.h"
#include "catpaw/objbuilder.h"

#define TYPE_FLUID 0
#define TYPE_CLOTH 1
#define TYPE_BOUNDARY 2
#define TYPE_EMITTER 3
#define TYPE_NULL 99

#define EPSILON 0.000001

#define MAX_OBJNUM 100

struct edgeConsVar {
	float lambda1;
	float lambda2;
	float stiff1;
	float stiff2;
};

class SimulationObject {
public:
	char type;
	int id;
	int pnum;
	int connum;
	int trinum;
	int indexedvbosize;

	int startpid;   //the id of the first point
	int startconid; //the id of the first edge constraint
	int starttriid; //the id of the first triangle
	int startvboid; //the id of the first element in vbo

					//simulation parameters
	float nRT;
	bool bVolumeCorr;
	bool bInjectGas;
	bool bJetGas;
	float dx;
	SimulationObject() {
		dx=0;
		nRT=0;
		bVolumeCorr = false;
		bInjectGas = false;
		bJetGas = false;
	}
};

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

struct SimData {
	
	//particle data
	cfloat3* pos;
	cfloat3* oldPos;
	cfloat3* deltaPos;
	cfloat3* avgDeltaPos;
	cfloat3* normal;
	cfloat3* vel;
	cfloat4* color;
	int* numCons;
	int* type;
	int* group;
	int* uniqueId;
	char* jetFlag;
	float* lambda;
	float* invMass;
	float* mass;
	float* absorbBuf;
	float* dripBuf;
	float* normalizeAbsorb;
	float* normalizeDrip;
	float* deltaAbsorb;
	float* deltaDrip;

	cfloat3* sortedPos;
	cfloat3* sortedOldPos;
	cfloat3* sortedVel;
	cfloat4* sortedColor;
	int* sortedType;
	int* sortedGroup;
	int* sortedUniqueId;
	char* sortedJetFlag;
	float* sortedInvMass;
	float* sortedMass;
	float* sortedAbsorbBuf;
	float* sortedDripBuf;
	int* indexTable;

	//edge data
	edgeConstraint* edgeCons;
	edgeConsVar* edgeConsVar;

	//triangle data
	objtriangle* triangles;
	float* facetVol;
	float* facetArea;
	int*   facetObjId;
	cfloat3* baryCenter;
	int*	 baryCenterTriangleId;
	cfloat3* sortedBaryCenter;
	int*	 sortedBaryCenterTriangleId;

	//object data
	SimulationObject* objs;

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






void copyDeviceBuffer();
void fetchDeviceBuffer();

	
void calcHash(
	SimData data,
	int pnum
);
	
void sortParticle(
	SimData data,
	int pnum
);
void reorderDataAndFindCellStart(
	SimData data,
	int numParticles,
	int numGridCells
);

void sortBaryCenterCUDA(SimData data, int numTriangles, int numGridCells);
	
void predictPosition(
	SimData data,
	float dt,
	int numParticles
);
void calcLambda(
	SimData data,
	int numParticles
);
void calcDeltaPos(
	SimData data,
	int numParticles
);

void updatePos(
	SimData data,
	int numParticles
);

void calcStablizeDeltaPos(
	SimData data,
	int numParticles
);

void updateStablizePos(
	SimData data,
	int numParticles
);

void updateVel(
	SimData data,
	int numParticles,
	float dt
);

void applyXSPH(
	SimData data,
	int numParticles
);

void waterAbsorption(SimData data, int numParticles);
void waterDiffusion(SimData data, int numParticles);
void waterEmission(SimData data, int numParticles);


void calcEdgeCons(SimData data, int numEdgeCons);

void resetEdgeConsX(SimData data, int numEdgeCons);
void calcEdgeConsX(SimData data, int numEdgeCons);

void calcFacetVol(
	SimData data,
	int numTriangles
);

void calcVolDeltaPos(
	SimData data,
	int numTriangles,
	bool jetGas
);

void calcVolPressureDPos(SimData data, int numTriangles, vector<SimulationObject>& objvec);

float getJetVol(SimData data, int numTriangles);
void getJetVolPressure(SimData data, int numTriangles, vector<SimulationObject>& objvec);

float getVol(SimData data, int numTriangles);

void labelCollisionCell(SimData data, int numParticles);

void detectCollision(
	SimData data,
	int numParticles
);

void detectCollisionWithMesh(SimData data, int numParticles);

void calcParticleNormal(SimData data, int numTriangles, int numParticles);

#ifndef PBFSOLVER
#define PBFSOLVER

#include "pbf_gpu.cuh"
#include "catpaw/objbuilder.h"
#include "catpaw/cpXMLHelper.h"
#include "Solver.h"

#define RUN_GPU 0
#define RUN_CPU 1



struct fluidvol {
	cfloat3 xmin;
	cfloat3 xmax;
	float volfrac[10];
};


#define FLUIDSRC_ROUND 1
#define FLUIDSRC_SQUARE 2

struct FluidSrc {
	cfloat3 srcpos;
	cfloat3 norm;
	float radius;
	float speed;
	char type;//geometry type
	int interval;
};

class PBFSolver:public Solver{

public:
	
	//CPU data
	cfloat3* hPos;
	cfloat4* hColor;
	cfloat3* hVel;
	int* hType;
	int* hGroup;
	float* hMass;
	float* hInvMass;
	int* hUniqueId;
	int* hIndexTable;
	char* hJetFlag;
	float* hAbsorbBuf;
	float* hDripBuf;
	cfloat3* hPosWriteBuf;

	vector<edgeConstraint> edgeConstraints;
	vector<objtriangle> trianglelist;
	cfloat3* hBaryCenter;
	int* hBaryCenterTriangleId;

	vector<SimulationObject> objectvec;
	vector<float> nRTvec;

	//GPU data
	SimData dData;

	vector<int> fixedvec;
	vector<fluidvol> fluidvols;
	vector<FluidSrc> fluidsrcs;

	//opengl rendering
	cfloat3* hPosRender;
	cfloat4* hColorRender;
	vector<unsigned int> indexedVBO;

	


	//Params
	SimParam param;

	//scene parameters
	float   densratio[10];
	float   viscratio[10];

	int numParticles;
	int numGridCells;
	int numEdgeCons;
	int numTriangles;
	int numParticlesDeleted;

	int frameNo;
	int caseid;
	
	char runmode = 0;

	float time;

	bool bVolumeCorr;
	bool bEmitParticle;
	bool bInjectGas;
	bool bInitVol;
	bool bJetGas;
	bool bModelPorous;
	bool bReleaseSource;
	bool bNormalizePos; //when dumping particle data



	//=============================
	//           METHOD
	//=============================
	
	void step();
	void HandleKeyEvent(char key);

	void setupHostBuffer();
	void setupDeviceBuffer(); 
	void copy2Device();
	void copy2Device_partial(int begin, int end);
	void copy2Host();
	void copy2host_full();

	
	void sort();
	void sortBaryCenter();

	void solvePBF();
	

	//add particle
	int addDefaultParticle();
	void addfluidvolumes();
	void emitParticle();

	//boundary
	void addwall(cfloat3 min, cfloat3 max);
	void addopenbox(cfloat3 min, cfloat3 max, float thickness);
	void loadboundary(string fname);
	void loadParticleSample(string fname);


	void removeInvalidParticles();
	void emitWaterFromCloth();
	


	//Setup Scenes

	void parseParam(char* xmlpath);
	void loadParam(char* xmlpath);

	void loadFluidVols(XMLElement* sceneEle);

	int loadClothObj(ObjContainer& oc, cmat4 materialMat);
	
	void addCloth();
	void addBalloon();
	void addBalloon(cmat4& materialMat);
	void addArmadilo();
	void addWaterBag();
	void addMesh();

	void addFluidSource();
	void fluidSrcEmit();
	void addParticleFluidSource();
	void particleFluidSrcEmit();
	void releaseFluidSource();

	void setupBasicScene();
	void setupBalloonScene();
	void setupWaterClothScene();
	void setupInjection();
	void setupMultiBalloonScene();
	void setupArmadilloScene();

	void dumpRenderingData();
	void dumpSimulationData();
	void dumpSimulationDataText();
	void loadSimulationData(char* filepath);
	void loadSimulationData(char* filepath, cmat4& materialMat);
	void loadSimulationDataText(char* filepath, cmat4& materialMat);

};



#endif
#pragma once

#include "sph_solver.cuh"
#include "catpaw/objbuilder.h"
#include "catpaw/cpXMLHelper.h"
#include "Solver.h"
#include "particle_common.h"
#include "ParticleGenerator.h"

namespace sph{


#define SPH 0
#define DFSPH 1


class SPHSolver : public Solver {

public:

	//nooverride
	//vecf3	hPos;
	//vecf4	hColor;
	vecf3   hVel;
	vecf3	hNormal;
	veci	hType;
	veci	hGroup;
	vecf	hMass;
	vecf	hInvMass;
	veci	hUniqueId;
	veci	hIndexTable;
	//veci	hJetFlag;
	vecf	hDensity;


	SimData_SPH dData;
	
	vector<fluidvol> fvs;
	vector<FluidSrc> fss;

	//int numP;
	int numGC;
	int frameNo;
	int caseid;
	int runmode;

	float time;
	
	bool bEmitParticle;









	void step();
	void HandleKeyEvent(char key);

	void setupHostBuffer();
	void setupDeviceBuffer();
	void copy2Device();
	void copy2Device_partial(int begin, int end);
	void copy2Host();
	void copy2host_full();

	void sort();
	void solveSPH();
	void solveDFSPH();


	void setup();
	int addDefaultParticle();
	void addfluidvolumes();
	void fluidSrcEmit();

	void setupDFSPH();

	//void addwall(cfloat3 min, cfloat3 max);
	//void addopenbox(cfloat3 min, cfloat3 max, float thickness);
	//void loadboundary(string fname);
	//void loadParticleSample(string fname);
	void loadPO(ParticleObject* po);

	//Setup Scenes
	void parseParam(char* xmlpath);
	void loadParam(char* xmlpath);
	void loadFluidVols(XMLElement* sceneEle);

	void dumpRenderingData();
	void dumpSimulationData();
	void dumpSimulationDataText();
	void loadSimulationData(char* filepath);
	void loadSimulationData(char* filepath, cmat4& materialMat);
	void loadSimulationDataText(char* filepath, cmat4& materialMat);

	void setupFluidScene();
};

};
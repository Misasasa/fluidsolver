#pragma once

#include "sph_solver.cuh"
#include "catpaw/objbuilder.h"
#include "catpaw/cpXMLHelper.h"

#include "ParticleSolver.h"
#include "particle_common.h"
#include "ParticleGenerator.h"

#define SPH 0
#define DFSPH 1
#define MSPH 2
#define MSPH_REN 3

class SPHSolver : public ParticleSolver {

protected:
	veci	unique_id;
	veci	id_table;
	vecf3	normal;
	veci	type;
	
	SimData_SPH device_data;
	
	vector<fluidvol> fluid_volumes;
	vector<FluidSrc> fluid_sources;

	//int num_particles;
	int num_fluid_particles;
	int num_grid_cells;
	int frame_count;
	int dump_count;
	int case_id;
	int run_mode;

	//Counters, one for each type, such as fluid, deformable, rigid, etc.
	//This is mainly used for deformables, to keep tracking the entry id of
	//tht neighbor list.
	int localid_counter[100];

	float time;
	
	bool emit_particle_on;
	bool advect_scriptobject_on;

public:

	void Step();
	void HandleKeyEvent(char key);
	void Eval(const char* expression); //override

	void SetupHostBuffer();
	void SetupDeviceBuffer();
	void CopyParticleDataToDevice();
	void CopyParticleDataToDevice(int begin, int end);
	void CopyPosColorFromDevice();
	void CopySimulationDataFromDevice();

	void SortParticles();
	void SolveSPH();
	void SolveDFSPH();
	
	void Setup();
	int  AddDefaultParticle();

	// DFSPH
	void SetupDFSPH();

	/* Rigid bodies, deformables. */
	void LoadPO(ParticleObject* po);
	void LoadPO(ParticleObject* po, int type);

	//Setup Scenes
	void ParseParam(char* xmlpath);
	void LoadParam(char* xmlpath);
	
	void DumpSimulationDataText();
	void LoadSimulationDataText(char* filepath, cmat4& materialMat);
	void DumpRenderData();

	void SetupFluidScene();
};
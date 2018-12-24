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
#define MSPH 2


class SPHSolver : public Solver {

public:

	//nooverride
	//vecf3	host_x;
	//vecf4	host_color;
	vecf3   host_v;
	vecf3	host_normal;
	veci	host_type;
	veci	host_group;
	vecf	host_mass;
	vecf	host_inv_mass;
	veci	host_unique_id;
	veci	host_id_table;
	vecf	host_rest_density; //rest density
	vecf	host_vol_frac; //volume fraction
	vecf3	host_v_star;

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

	float time;
	
	bool emit_particle_on;









	void Step();
	void HandleKeyEvent(char key);

	void SetupHostBuffer();
	void SetupDeviceBuffer();
	void Copy2Device();
	void Copy2Device(int begin, int end);
	void CopyFromDevice();
	void CopyFromDeviceFull();

	void Sort();
	void SolveSPH();
	void SolveDFSPH();
	void SolveMultiphaseSPH(); //with DFSPH

	void Setup();
	int  AddDefaultParticle();
	void Addfluidvolumes();
	void EnableFluidSource();

	// DFSPH
	void SetupDFSPH();

	// Multiphase Fluid
	void SetupMultiphaseSPH();
	void AddMultiphaseFluidVolumes();

	void LoadPO(ParticleObject* po);

	//Setup Scenes
	void ParseParam(char* xmlpath);
	void LoadParam(char* xmlpath);
	
	void DumpSimulationDataText();
	void LoadSimulationDataText(char* filepath, cmat4& materialMat);
	void DumpRenderData();

	void SetupFluidScene();
};

};
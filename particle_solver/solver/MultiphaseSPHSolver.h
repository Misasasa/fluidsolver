#pragma once

#include "sph_solver.h"

class MultiphaseSPHSolver : public SPHSolver {
protected:
	veci	group;
	vecf	mass;
	vecf	inv_mass;
	
	vecf	rest_density;
	vecf	vol_frac;
	vecf3	v_star;
	vecf	temperature;
	vecf    heat_buffer;
	vector<cmat3> cauchy_stress;
	veci    localid;

public:
	void Step();
	void HandleKeyEvent(char key);
	void Eval(const char* expression); //override

	void SetupHostBuffer();
	void SetupDeviceBuffer();
	void CopyParticleDataToDevice();
	void CopyParticleDataToDevice(int begin, int end);
	void CopySimulationDataFromDevice();

	void Sort();
	void SolveMultiphaseSPH(); //with DFSPH
	void SolveMultiphaseSPHRen(); //with Ren et al. 13
	void SolveIISPH();

	void PhaseDiffusion_Host();

	void Setup();
	int  AddDefaultParticle();

	// DFSPH
	void SetupDFSPH();

	// Multiphase Fluid
	void SetupMultiphaseSPH();
	void AddMultiphaseFluidVolumes();
	void SetupMultiphaseSPHRen();
	void SetupIISPH();

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
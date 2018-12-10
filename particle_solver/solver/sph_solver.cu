#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>


#include "helper_cuda.h"
#include "helper_string.h"
#include "sph_kernel_impl.cuh"

#include "sph_solver.cuh"

namespace sph{

void CopyParam2Device() {
	cudaMemcpyToSymbol(dParam, &hParam, sizeof(SimParam_SPH));
}

void CopyParamFromDevice() {
	cudaMemcpyFromSymbol(&hParam, dParam, sizeof(SimParam_SPH));
}





void calcHash(SimData_SPH data,	int num_particles) {

	getLastCudaError("Kernel execution failed:before calc hash");
	uint num_blocks, num_threads;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	calcHashD << <num_blocks, num_threads >> > (data.particleHash,
		data.particleIndex,
		data.pos,
		num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: calc hash");
}



void sortParticle(SimData_SPH data,	int pnum) {
	thrust::sort_by_key(
		thrust::device_ptr<int>(data.particleHash),
		thrust::device_ptr<int>(data.particleHash + pnum),
		thrust::device_ptr<int>(data.particleIndex)
	);

}



void reorderDataAndFindCellStart(
	SimData_SPH data, 
	int num_particles,
	int numGridCells
) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	cudaMemset(data.gridCellStart, 0xffffffff, numGridCells * sizeof(uint));

	//shared memory size
	uint smemSize = sizeof(uint)*(num_threads + 1);
	
	reorderDataAndFindCellStartD << < num_blocks, num_threads, smemSize >> >(
		data,
		num_particles);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: reorder data");

}



void applyXSPH(SimData_SPH data, int num_particles) {


}



void computePressure(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);
	
	computeP <<< num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute pressure");

}

void computeForce(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	computeF <<< num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute force");

}

void advect(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	advectAndCollision <<< num_blocks, num_threads>>> (data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: advection");

}




//==================================================
//
//                     DFSPH
//
//==================================================

void computeDensityAlpha(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	computeDensityAlpha_kernel <<< num_blocks, num_threads>>> (data, num_particles);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute df alpha");

}


void computeNonPForce(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	computeNPF_kernel <<< num_blocks, num_threads>>> (data, num_particles);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute non-pressure force");

}

void correctDensityError(SimData_SPH data,
	int num_particles,
	int maxiter,
	float ethres,
	bool bDebug)
{
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[num_particles];
	/*
	cfloat3* dbg3 = new cfloat3[num_particles];
	cudaMemcpy(dbg3, data.v_star, num_particles*sizeof(cfloat3), cudaMemcpyDeviceToHost);
	FILE* fdbg;
	fdbg = fopen("vstar0.txt", "w+");
	for (int i=0; i<num_particles; i++)
	{
		fprintf(fdbg, "%d %f %f %f\n", i, dbg3[i].x, dbg3[i].y, dbg3[i].z);
	}
	fclose(fdbg);
	*/

	while (true && iter<maxiter) {
		solveDensityStiff <<< num_blocks, num_threads>>> (data, num_particles);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: solve density stiff");

		//get error
		
		cudaMemcpy(debug, data.error, num_particles*sizeof(float), cudaMemcpyDeviceToHost);
		error = -9999;
		for (int i=0; i<num_particles; i++) {
			error = debug[i]>error? debug[i]:error;
		}
			
		if(bDebug)
			printf("%d error: %f\n", iter, error);
		if (error<ethres)
			break;
		
		applyPStiff <<<num_blocks, num_threads>>>(data, num_particles);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply density stiff");

		iter++;
	}

	updatePosition <<<num_blocks, num_threads>>>(data, num_particles);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update position");

	/*
	//cfloat3* dbg3 = new cfloat3[num_particles];
	cudaMemcpy(dbg3, data.v_star, num_particles*sizeof(cfloat3), cudaMemcpyDeviceToHost);
	fdbg = fopen("vstar.txt","w+");
	for (int i=0; i<num_particles; i++)
	{
		fprintf(fdbg, "%d %f %f %f\n",i, dbg3[i].x, dbg3[i].y, dbg3[i].z);
	}
	fclose(fdbg);
	*/
}

void correctDivergenceError(SimData_SPH data,
	int num_particles,
	int maxiter,
	float ethres,
	bool bDebug)
{
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);
	
	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[num_particles];
	
	//warm start
	solveDivergenceStiff <<< num_blocks, num_threads>>> (data, num_particles);

	while (true && iter<maxiter) {
		solveDivergenceStiff <<< num_blocks, num_threads>>> (data, num_particles);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: compute divergence stiff");

		cudaMemcpy(debug, data.error, num_particles*sizeof(float), cudaMemcpyDeviceToHost);
		error = 0;
		for (int i=0; i<num_particles; i++)
			error = debug[i]>error? debug[i]:error;		
		
		if (error<ethres)
			break;
		
		applyPStiff <<<num_blocks, num_threads>>>(data, num_particles);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply divergence stiff");

		iter++;
	}
	if (bDebug)
		printf("%d error: %f\n", iter, error);
	UpdateVelocities<<<num_blocks, num_threads>>>(data,num_particles);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update velocities");

}















//==================================================
//
//                 Multiphase SPH
//
//==================================================

void DFAlpha_Multiphase(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	DFAlphaKernel_Multiphase <<< num_blocks, num_threads>>> (data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute df alpha multiphase");

}

void EffectiveMass(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	EffectiveMassKernel <<< num_blocks, num_threads>>> (data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update mass factor");

}

void NonPressureForce_Multiphase(SimData_SPH data, int num_particles) {
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	computeNPF_MPH_kernel <<< num_blocks, num_threads>>> (data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute non-pressure force multiphase");

}

void EnforceDensity_Multiphase(SimData_SPH data, int num_particles,
	int maxiter, float ethres, bool bDebug) {

	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[num_particles];

	while (true && iter<maxiter) {
		DensityStiff_Multiphase <<< num_blocks, num_threads>>> (data, num_particles);

		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: solve density stiff");

		//get error

		cudaMemcpy(debug, data.error, num_particles*sizeof(float), cudaMemcpyDeviceToHost);
		error = -9999;
		for (int i=0; i<num_particles; i++)
			error = debug[i]>error ? debug[i] : error;
		
		if (error<ethres) break;

		ApplyPressureKernel_Multiphase <<<num_blocks, num_threads>>>(data, num_particles);

		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply density stiff");

		iter++;
	}
	if (bDebug)	printf("%d density error: %f\n", iter, error);
	delete debug;

	updatePosition <<<num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update position");

}

void EnforceDivergenceFree_Multiphase(SimData_SPH data, int num_particles,
	int maxiter, float ethres, bool bDebug) {

	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[num_particles];

	while (true && iter<maxiter) {
		DivergenceFreeStiff_Multiphase <<< num_blocks, num_threads>>> (data, num_particles);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: compute divergence stiff");

		cudaMemcpy(debug, data.error, num_particles*sizeof(float), cudaMemcpyDeviceToHost);
		error = 0;
		for (int i=0; i<num_particles; i++)
			error = debug[i]>error ? debug[i] : error;
		if (error<ethres) break;

		ApplyPressureKernel_Multiphase <<<num_blocks, num_threads>>>(data, num_particles);
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply divergence stiff");

		iter++;
	}
	if (bDebug)	printf("%d divergence-free error: %f\n", iter, error);
	delete debug;

	UpdateVelocities<<<num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update velocities");
}

void DriftVelocity(SimData_SPH data, int num_particles) {
	
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	DriftVelocityKernel<<<num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: drift velocity.");

}



void PhaseDiffusion(SimData_SPH data, int num_particles) {
	
	uint num_threads, num_blocks;
	computeGridSize(num_particles, 256, num_blocks, num_threads);

	PredictPhaseDiffusionKernel <<<num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: predict phase diffusion.");

	PhaseDiffusionKernel<<<num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: phase diffusion.");

	UpdateVolumeFraction<<<num_blocks, num_threads>>>(data, num_particles);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update volume fraction.");
	
	/*
	float* dbg_pt = new float[num_particles*hParam.maxtypenum];
	cudaMemcpy(dbg_pt, data.vFrac, num_particles*hParam.maxtypenum*sizeof(float),
		cudaMemcpyDeviceToHost);
	float verify=0;
	for(int i=0; i<num_particles; i++)
		verify += dbg_pt[i*hParam.maxtypenum];
	printf("total volume fraction phase 0: %f\n", verify);
	delete dbg_pt;
	*/
}

};
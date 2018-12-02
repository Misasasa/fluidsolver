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

void copyDeviceBuffer() {
	cudaMemcpyToSymbol(dParam, &hParam, sizeof(SimParam_SPH));
}

void fetchDeviceBuffer() {
	cudaMemcpyFromSymbol(&hParam, dParam, sizeof(SimParam_SPH));
}





void calcHash(SimData_SPH data,	int numParticles) {

	getLastCudaError("Kernel execution failed:before calc hash");
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	calcHashD << <numBlocks, numThreads >> > (data.particleHash,
		data.particleIndex,
		data.pos,
		numParticles);
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
	int numParticles,
	int numGridCells
) {
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	cudaMemset(data.gridCellStart, 0xffffffff, numGridCells * sizeof(uint));

	//shared memory size
	uint smemSize = sizeof(uint)*(numThreads + 1);
	
	reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> >(
		data,
		numParticles);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: reorder data");

}



void applyXSPH(SimData_SPH data, int numParticles) {


}



void computePressure(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);
	
	computeP <<< numBlocks, numThreads>>>(data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute pressure");

}

void computeForce(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeF <<< numBlocks, numThreads>>>(data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute force");

}

void advect(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	advectAndCollision <<< numBlocks, numThreads>>> (data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: advection");

}




//==================================================
//
//                     DFSPH
//
//==================================================

void computeDensityAlpha(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeDensityAlpha_kernel <<< numBlocks, numThreads>>> (data, numP);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute df alpha");

}


void computeNonPForce(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeNPF_kernel <<< numBlocks, numThreads>>> (data, numP);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute non-pressure force");

}

void correctDensityError(SimData_SPH data,
	int numP,
	int maxiter,
	float ethres,
	bool bDebug)
{
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[numP];
	/*
	cfloat3* dbg3 = new cfloat3[numP];
	cudaMemcpy(dbg3, data.v_star, numP*sizeof(cfloat3), cudaMemcpyDeviceToHost);
	FILE* fdbg;
	fdbg = fopen("vstar0.txt", "w+");
	for (int i=0; i<numP; i++)
	{
		fprintf(fdbg, "%d %f %f %f\n", i, dbg3[i].x, dbg3[i].y, dbg3[i].z);
	}
	fclose(fdbg);
	*/

	while (true && iter<maxiter) {
		solveDensityStiff <<< numBlocks, numThreads>>> (data, numP);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: solve density stiff");

		//get error
		
		cudaMemcpy(debug, data.error, numP*sizeof(float), cudaMemcpyDeviceToHost);
		error = -9999;
		for (int i=0; i<numP; i++) {
			error = debug[i]>error? debug[i]:error;
		}
			
		if(bDebug)
			printf("%d error: %f\n", iter, error);
		if (error<ethres)
			break;
		
		applyPStiff <<<numBlocks, numThreads>>>(data, numP);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply density stiff");

		iter++;
	}

	updatePosition <<<numBlocks, numThreads>>>(data, numP);
	
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update position");

	/*
	//cfloat3* dbg3 = new cfloat3[numP];
	cudaMemcpy(dbg3, data.v_star, numP*sizeof(cfloat3), cudaMemcpyDeviceToHost);
	fdbg = fopen("vstar.txt","w+");
	for (int i=0; i<numP; i++)
	{
		fprintf(fdbg, "%d %f %f %f\n",i, dbg3[i].x, dbg3[i].y, dbg3[i].z);
	}
	fclose(fdbg);
	*/
}

void correctDivergenceError(SimData_SPH data,
	int numP,
	int maxiter,
	float ethres,
	bool bDebug)
{
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);
	
	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[numP];
	
	//warm start
	solveDivergenceStiff <<< numBlocks, numThreads>>> (data, numP);

	while (true && iter<maxiter) {
		solveDivergenceStiff <<< numBlocks, numThreads>>> (data, numP);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: compute divergence stiff");

		cudaMemcpy(debug, data.error, numP*sizeof(float), cudaMemcpyDeviceToHost);
		error = 0;
		for (int i=0; i<numP; i++)
			error = debug[i]>error? debug[i]:error;		
		
		if (error<ethres)
			break;
		
		applyPStiff <<<numBlocks, numThreads>>>(data, numP);
		
		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply divergence stiff");

		iter++;
	}
	if (bDebug)
		printf("%d error: %f\n", iter, error);
	updateVelocities<<<numBlocks, numThreads>>>(data,numP);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update velocities");

}




//==================================================
//
//                 Multiphase SPH
//
//==================================================

void computeDFAlpha_MPH(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeDFAlpha_MPH_kernel <<< numBlocks, numThreads>>> (data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute df alpha multiphase");

}

void updateMassFac(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	updateMassFac_kernel <<< numBlocks, numThreads>>> (data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update mass factor");

}

void computeNonPForce_MPH(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeNPF_MPH_kernel <<< numBlocks, numThreads>>> (data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute non-pressure force multiphase");

}

void correctDensity_MPH(SimData_SPH data, int numP,
	int maxiter, float ethres, bool bDebug) {

	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[numP];

	while (true && iter<maxiter) {
		solveDensityStiff <<< numBlocks, numThreads>>> (data, numP);

		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: solve density stiff");

		//get error

		cudaMemcpy(debug, data.error, numP*sizeof(float), cudaMemcpyDeviceToHost);
		error = -9999;
		for (int i=0; i<numP; i++) {
			error = debug[i]>error ? debug[i] : error;
		}

		if (bDebug)
			printf("%d error: %f\n", iter, error);
		if (error<ethres)
			break;

		applyPStiff_MPH <<<numBlocks, numThreads>>>(data, numP);

		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply density stiff");

		iter++;
	}

	updatePosition <<<numBlocks, numThreads>>>(data, numP);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update position");

}

void correctDivergence_MPH(SimData_SPH data, int numP,
	int maxiter, float ethres, bool bDebug) {

	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	float error;
	int iter = 0;
	//jacobi iteration
	float* debug = new float[numP];

	//warm start
	solveDivergenceStiff <<< numBlocks, numThreads>>> (data, numP);

	while (true && iter<maxiter) {
		solveDivergenceStiff <<< numBlocks, numThreads>>> (data, numP);

		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: compute divergence stiff");

		cudaMemcpy(debug, data.error, numP*sizeof(float), cudaMemcpyDeviceToHost);
		error = 0;
		for (int i=0; i<numP; i++)
			error = debug[i]>error ? debug[i] : error;

		if (error<ethres)
			break;

		applyPStiff_MPH <<<numBlocks, numThreads>>>(data, numP);

		cudaThreadSynchronize();
		getLastCudaError("Kernel execution failed: apply divergence stiff");

		iter++;
	}
	if (bDebug)
		printf("%d error: %f\n", iter, error);
	updateVelocities<<<numBlocks, numThreads>>>(data, numP);

	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: update velocities");


}

void computeDriftVelocity(SimData_SPH data, int numP) {
	
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeDriftVelocity_kernel<<<numBlocks, numThreads>>>(data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute drift velocity");

}

void computePhaseDiffusion(SimData_SPH data, int numP) {
	
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computePhaseDiffusion_kernel<<<numBlocks, numThreads>>>(data, numP);
	cudaThreadSynchronize();
	getLastCudaError("Kernel execution failed: compute drift velocity");

}

};
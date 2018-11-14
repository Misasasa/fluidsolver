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
	
	getLastCudaError("Kernel execution failed: reorder data");

}



void applyXSPH(SimData_SPH data, int numParticles) {


}



void computePressure(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);
	
	computeP <<< numBlocks, numThreads>>>(data, numP);
}

void computeForce(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeF <<< numBlocks, numThreads>>>(data, numP);
}

void advect(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	advectAndCollision <<< numBlocks, numThreads>>> (data, numP);
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
}


void computeNonPForce(SimData_SPH data, int numP) {
	uint numThreads, numBlocks;
	computeGridSize(numP, 256, numBlocks, numThreads);

	computeNPF_kernel <<< numBlocks, numThreads>>> (data, numP);
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
		//get error
		
		cudaMemcpy(debug, data.pstiff, numP*sizeof(float), cudaMemcpyDeviceToHost);
		error = 0;
		for(int i=0; i<numP; i++)
			error += abs(debug[i]);
		if(bDebug)
			printf("%d error: %f\n", iter, error);
		if (error<ethres)
			break;
		
		applyPStiff <<<numBlocks, numThreads>>>(data, numP);
		iter++;
	}

	updatePosition <<<numBlocks, numThreads>>>(data, numP);
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
		
		cudaMemcpy(debug, data.pstiff, numP*sizeof(float), cudaMemcpyDeviceToHost);
		error = 0;
		for (int i=0; i<numP; i++)
			error += abs(debug[i]);
		
		if (bDebug)
			printf("%d error: %f\n", iter, error);
		if (error<ethres)
			break;
		
		applyPStiff <<<numBlocks, numThreads>>>(data, numP);
		iter++;
	}

	updateVelocities<<<numBlocks, numThreads>>>(data,numP);
}







};
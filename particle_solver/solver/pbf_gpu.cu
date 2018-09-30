

#include <catpaw/CatTimer.h>


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <thrust\device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>


#include "helper_cuda.h"
#include "helper_string.h"
#include "pbf_kernel_impl.cuh"



typedef unsigned int uint;


void copyDeviceBuffer() {
	cudaMemcpyToSymbol(dParam, &hParam, sizeof(SimParam));
}

void fetchDeviceBuffer() {
	cudaMemcpyFromSymbol(&hParam, dParam, sizeof(SimParam));
}


uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}

//==================================================
//
//
//                 COUNTING SORT
//
//
//===================================================
	
	

	

	

void calcHash(
	SimData data,
	int numParticles
) 
{

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

void sortParticle(
	SimData data,
	int pnum
) {
	thrust::sort_by_key(
		thrust::device_ptr<int>(data.particleHash),
		thrust::device_ptr<int>(data.particleHash + pnum),
		thrust::device_ptr<int>(data.particleIndex)
	);
}

void reorderDataAndFindCellStart(
	SimData data,
	int numParticles,
	int numCells
) {
	uint numThreads, numBlocks;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
		
	cudaMemset(data.gridCellStart, 0xffffffff, numCells * sizeof(uint));

	//shared memory size
	uint smemSize = sizeof(uint)*(numThreads + 1);
		
	reorderDataAndFindCellStartD << < numBlocks, numThreads, smemSize >> >(
		data,
		numParticles);

	getLastCudaError("Kernel execution failed: reorder data");
}


void sortBaryCenterCUDA(SimData data, int numTriangles, int numCells) {
	uint numThreads, numBlocks;
	computeGridSize(numTriangles, 256, numBlocks, numThreads);
	//step 1

	calcBaryCenterHashD << <numBlocks, numThreads >> > (data, numTriangles);
	getLastCudaError("Kernel execution failed: calc bary center hash");

	//step 2

	thrust::sort_by_key(
		thrust::device_ptr<int>(data.baryCenterHash),
		thrust::device_ptr<int>(data.baryCenterHash + numTriangles),
		thrust::device_ptr<int>(data.baryCenterIndex)
	);

	//step 3

	cudaMemset(data.gridCellStartBaryCenter, 0xffffffff, numCells * sizeof(uint));

	//shared memory size
	uint smemSize = sizeof(uint)*(numThreads + 1);

	reorderBaryCenterAndFindCellStartD << < numBlocks, numThreads, smemSize >> >(
		data,
		numTriangles);

	getLastCudaError("Kernel execution failed: reorder bary center");

}
	
float systime = 0;

	
void predictPosition(
	SimData data,
	float dt,
	int numParticles
	) {

	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	predictPositionD<<< numBlocks, numThreads >>>(
		data,
		dt, 
		numParticles);
	getLastCudaError("Kernel execution failed: predict pos");
}

//=========================================================
//
//
//                #Solve Constraints#
//
//
//=========================================================


void calcLambda(
	SimData data,
	int numParticles
) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	calcLambdaD <<< numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: calc lambda");
}

void calcDeltaPos(
	SimData data,
	int numParticles
) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	calcDeltaPosD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: calc deltapos");
}

void updatePos(
	SimData data,
	int numParticles
) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	updatePosD <<< numBlocks, numThreads >>>(
		data,
		numParticles
	);
	getLastCudaError("Kernel execution failed: update pos");
}

void calcStablizeDeltaPos(SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	calcStablizeDeltaPosD <<<numParticles, numThreads>>>(data,numParticles);
	getLastCudaError("Kernel execution failed: calc stablize deltapos");
}

void updateStablizePos(SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	updateStablizePosD <<<numParticles, numThreads>>>(data, numParticles);
	getLastCudaError("Kernel execution failed: calc stablize deltapos");
}

void calcEdgeCons(SimData data, int numEdgeCons) {
	uint numBlocks, numThreads;
	computeGridSize(numEdgeCons, 256, numBlocks, numThreads);
	calcEdgeConsD << < numBlocks, numThreads >> >(
		data,
		numEdgeCons
		);
	getLastCudaError("Kernel execution failed: calc edge constraint");
}

void resetEdgeConsX(SimData data, int numEdgeCons) {
	uint numBlocks, numThreads;
	computeGridSize(numEdgeCons, 256, numBlocks, numThreads);
	resetEdgeConsXD << < numBlocks, numThreads >> >(
		data,
		numEdgeCons
		);
	getLastCudaError("Kernel execution failed: reset edge constraint X");
}

void calcEdgeConsX(SimData data, int numEdgeCons) {
	uint numBlocks, numThreads;
	computeGridSize(numEdgeCons, 256, numBlocks, numThreads);
	/*calcRubberEdgeConsXD << < numBlocks, numThreads >> >(
		data,
		numEdgeCons
		);*/
	calcEdgeConsXD << < numBlocks, numThreads >> >(
		data,
		numEdgeCons
		);
	getLastCudaError("Kernel execution failed: calc edge constraint X");
}


void calcFacetVol(
	SimData data,
	int numTriangles
) {
	uint numBlocks, numThreads;
	computeGridSize(numTriangles, 256, numBlocks, numThreads);
	calcFacetVolD << < numBlocks, numThreads >> > (
		data,
		numTriangles
		);
	getLastCudaError("Kernel execution failed: calc vol constraint");
}

void calcVolDeltaPos(
	SimData data,
	int numTriangles,
	bool jetGas
) {
	thrust::device_ptr<float> begin(data.facetVol);
	thrust::device_ptr<float> end = begin + numTriangles;
	float totalVol = thrust::reduce(begin, end);

	begin = thrust::device_pointer_cast(data.facetArea);
	end = begin + numTriangles;
	float surfaceArea = thrust::reduce(begin, end);

	float restvol = hParam.restvol;
	float dx = (restvol - totalVol) / surfaceArea;
	dx *= hParam.volumeStiff;
	printf("%f\n",dx);

	uint numBlocks, numThreads;
	computeGridSize(numTriangles, 256, numBlocks, numThreads);

	calcVolDeltaPosD << < numBlocks, numThreads >> > (data, numTriangles);
	getLastCudaError("Kernel execution failed: calc vol deltaPos");

}

void calcVolPressureDPos(SimData data, int numTriangles, vector<SimulationObject>& objvec) {
	
	thrust::device_ptr<float> facetVol(data.facetVol);
	
	//thrust::device_ptr<float> begin(data.facetVol);
	//thrust::device_ptr<float> end = begin + numTriangles;
	bool updateVol = false;

	for (int i=0; i<objvec.size(); i++) {

		if(!objvec[i].bVolumeCorr)
			continue;
		updateVol = true;

		thrust::device_ptr<float> begin = facetVol + objvec[i].starttriid;
		thrust::device_ptr<float> end = begin + objvec[i].trinum;

		float totalVol = thrust::reduce(begin, end);
		if (totalVol < 20)
			totalVol = 20;

		begin = thrust::device_pointer_cast(data.facetArea + objvec[i].starttriid );
		end = begin + objvec[i].trinum;
		float surfaceArea = thrust::reduce(begin, end);

		float nRT = objvec[i].nRT;
		float pI = nRT / totalVol;

		float pE = hParam.pE;
		//printf("%f %f %f\n", pI, nRT, totalVol);
		float totalMass = 1;
		float dx = (pI - pE) * surfaceArea / totalMass * hParam.volumeStiff;
		objvec[i].dx = dx;
	}

	if (updateVol) {
		cudaMemcpy(data.objs, objvec.data(), objvec.size()*sizeof(SimulationObject), cudaMemcpyHostToDevice);

		uint numBlocks, numThreads;
		computeGridSize(numTriangles, 256, numBlocks, numThreads);

		calcVolDeltaPosD << < numBlocks, numThreads >> > (data, numTriangles);
		getLastCudaError("Kernel execution failed: calc vol pressure deltaPos");
	}

}

float getVol(SimData data, int numTriangles) {
	thrust::device_ptr<float> begin(data.facetVol);
	thrust::device_ptr<float> end = begin + numTriangles;
	float totalVol = thrust::reduce(begin, end);
	return totalVol;
}

float getJetVol(SimData data, int numTriangles) {
	thrust::device_ptr<float> begin(data.facetVol);
	thrust::device_ptr<float> end = begin + numTriangles;
	float totalVol = thrust::reduce(begin, end);
	
	float surfaceArea = 6;

	float restvol = hParam.restvol;
	float dx = (restvol - totalVol) / surfaceArea;
	dx *= hParam.volumeStiff;
	
	return dx * surfaceArea * 50;
}

void getJetVolPressure(SimData data, int numTriangles, vector<SimulationObject>& objvec) {
	//thrust::device_ptr<float> begin(data.facetVol);
	//thrust::device_ptr<float> end = begin + numTriangles;
	thrust::device_ptr<float> facetVol(data.facetVol);

	for (int i=0; i<objvec.size(); i++) {
		if(!objvec[i].bJetGas)
			continue;

		thrust::device_ptr<float> begin = facetVol + objvec[i].starttriid;
		thrust::device_ptr<float> end = begin + objvec[i].trinum;

		float totalVol = thrust::reduce(begin, end);
		if (totalVol < 10)
			totalVol = 10;
		float surfaceArea = 6;

		//float nRT = hParam.restvol;
		float nRT = objvec[i].nRT;
		float pI = nRT / totalVol;
		float pE = hParam.pE;
		float dnRT = (pI - pE)*pI * surfaceArea*hParam.dt / hParam.resistance;

		if (dnRT < 0.00001)
			dnRT = 0;
		objvec[i].nRT -= dnRT;

		if (objvec[i].nRT < 10)
			objvec[i].nRT = 10;
	}
}

void labelCollisionCell(SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	labelCollisionCellD << <numBlocks, numThreads >> > (data, numParticles);
	getLastCudaError("Kernel execution failed: detect collision");
}

void detectCollision(
	SimData data,
	int numParticles
) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
		
	detectCollisionD << <numBlocks, numThreads >> > (data, numParticles);
	getLastCudaError("Kernel execution failed: detect collision");
}

void detectCollisionWithMesh (SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);

	detectCollisionWithMeshD << <numBlocks, numThreads >> > (data, numParticles);
	getLastCudaError("Kernel execution failed: detect collision");
}

void calcParticleNormal(SimData data, int numTriangles, int numParticles) {
	uint numBlocks, numThreads;
	
	computeGridSize(numTriangles, 256, numBlocks, numThreads);
	calcFacetNormalD << <numBlocks, numThreads >> > (data, numTriangles);
	getLastCudaError("Kernel execution failed: calc facet normal");

	computeGridSize(numParticles, 256, numBlocks, numThreads);
	calcParticleNormalD << <numBlocks, numThreads >> > (data, numParticles);
	getLastCudaError("Kernel execution failed: calc particle normal");
}



void updateVel(
	SimData data,
	int numParticles,
	float dt
) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	updateVelD << < numBlocks, numThreads >> >(
		data,
		numParticles,
		dt);
	getLastCudaError("Kernel execution failed: update vel");
}

void applyXSPH(
	SimData data,
	int numParticles
) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	applyXSPHD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: apply xpsh");
}



bool start=true;
	

void waterAbsorption(SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	waterAbsorptionD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: water absorption");
}


void waterDiffusion(SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	waterDiffusionPredictD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: water diffusion predict");

	waterDiffusionD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: water diffusion");

	updateDiffusionD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: update diffusion");

}

void waterEmission(SimData data, int numParticles) {
	uint numBlocks, numThreads;
	computeGridSize(numParticles, 256, numBlocks, numThreads);
	waterEmissionD << < numBlocks, numThreads >> >(
		data,
		numParticles);
	getLastCudaError("Kernel execution failed: water emission");
}


//=======================================
//
//
//         #Update Particle State#
//
//
//=======================================




/*void diffuse_fluidphase () {
	kernel_absorbfluid<<<hParam.nblock_p, hParam.nthread_p>>>();
	cudaThreadSynchronize();

	kernel_diffuse_predict<<<hParam.nblock_p, hParam.nthread_p>>>();
	cudaThreadSynchronize();

	kernel_diffuse<<<hParam.nblock_p, hParam.nthread_p>>>();
	cudaThreadSynchronize();
}*/






/*void surfacetension () {
	kernel_yangtao_model <<<hParam.nblock_p, hParam.nthread_p>>>();
	cudaThreadSynchronize();

	kernel_computeNormal<<<hParam.nblock_p, hParam.nthread_p>>>();
	cudaThreadSynchronize();

	kernel_computeCurvature<<<hParam.nblock_p, hParam.nthread_p>>>();
	cudaThreadSynchronize();
}*/






//=======================================
//
//
//         Convariance Matrix
//
//
//=======================================


//void computeCovmat () {

//	kernel_computeAvgpos <<<hParam.nblock_p, hParam.nthread_p>>>();
//	cudaThreadSynchronize();

//	kernel_computeCovmat <<<hParam.nblock_p, hParam.nthread_p>>>();
//	cudaThreadSynchronize();

//	//return;

//	float* debug = (float*)malloc(sizeof(float)*9*hParam.pnum);
//	cudaMemcpy(debug, hParam.covmat, sizeof(cmat3)*hParam.pnum, cudaMemcpyDeviceToHost);
//	FILE* fp = fopen("covmat.txt","w+");
//	for (int i=0; i<hParam.pnum; i++) {
//		fprintf(fp, "%d\n%f %f %f\n%f %f %f\n %f %f %f\n",i,debug[i*9],debug[i*9+1],debug[i*9+2],
//			debug[i*9+3], debug[i*9+4], debug[i*9+5],
//			debug[i*9+6], debug[i*9+7], debug[i*9+8]);
//	}
//	fclose(fp);

//	cudaMemcpy(debug, hParam.u, sizeof(cmat3)*hParam.pnum, cudaMemcpyDeviceToHost);
//	fp = fopen("u.txt", "w+");
//	for (int i=0; i<hParam.pnum; i++) {
//		fprintf(fp, "%d\n%f %f %f\n%f %f %f\n %f %f %f\n", i, debug[i*9], debug[i*9+1], debug[i*9+2],
//			debug[i*9+3], debug[i*9+4], debug[i*9+5],
//			debug[i*9+6], debug[i*9+7], debug[i*9+8]);
//	}
//	fclose(fp);

//	cudaMemcpy(debug, hParam.s, sizeof(cmat3)*hParam.pnum, cudaMemcpyDeviceToHost);
//	fp = fopen("s.txt", "w+");
//	for (int i=0; i<hParam.pnum; i++) {
//		fprintf(fp, "%d\n%f %f %f\n%f %f %f\n %f %f %f\n", i, debug[i*9], debug[i*9+1], debug[i*9+2],
//			debug[i*9+3], debug[i*9+4], debug[i*9+5],
//			debug[i*9+6], debug[i*9+7], debug[i*9+8]);
//	}
//	fclose(fp);

//	cudaMemcpy(debug, hParam.v, sizeof(cmat3)*hParam.pnum, cudaMemcpyDeviceToHost);
//	fp = fopen("v.txt", "w+");
//	for (int i=0; i<hParam.pnum; i++) {
//		fprintf(fp, "%d\n%f %f %f\n%f %f %f\n %f %f %f\n", i, debug[i*9], debug[i*9+1], debug[i*9+2],
//			debug[i*9+3], debug[i*9+4], debug[i*9+5],
//			debug[i*9+6], debug[i*9+7], debug[i*9+8]);
//	}
//	fclose(fp);

//	free(debug);

//}

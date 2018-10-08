
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "pbfsolver.h"
#include "catpaw/cpXMLHelper.h"


extern SimParam hParam;

#include <algorithm>
using namespace std;

void PBFSolver::copy2Device() {
	
	int constraintnum = edgeConstraints.size();
	int trianglenum = trianglelist.size();
	printf("constraint number: %d\n", constraintnum);

	cudaMemcpy(dData.pos,	hPos,		numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.vel,	hVel,		numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color, hColor,		numParticles * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.type,	hType,		numParticles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.group, hGroup,		numParticles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.mass,  hMass,      numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.invMass,	hInvMass,	numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.uniqueId,	hUniqueId,	numParticles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.jetFlag,   hJetFlag,   numParticles * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.dripBuf, hDripBuf, numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.absorbBuf, hAbsorbBuf, numParticles * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dData.edgeCons,	edgeConstraints.data(), constraintnum * sizeof(edgeConstraint), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.triangles, trianglelist.data(),	trianglenum * sizeof(objtriangle), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.objs,		objectvec.data(),		objectvec.size()*sizeof(SimulationObject), cudaMemcpyHostToDevice);

	copyDeviceBuffer();
}


void PBFSolver::removeInvalidParticles() {
	for (int i = 0; i<numParticles; i++) {
		if (hType[i] != TYPE_FLUID)
			continue;
		else if (hMass[i] < EPSILON) //invalid
		{
			hType[i] = TYPE_NULL;
			hPos[i].Set(-9999, -9999, -9999);
			numParticlesDeleted++;
		}
	}
}

void PBFSolver::emitWaterFromCloth() {
	
	int start = numParticles;
	int pid = start;

	for (int i = 0; i<numParticles; i++) {
		if (hType[i] != TYPE_CLOTH)
			continue;
		if (hDripBuf[i] > hParam.emitThres) //invalid
		{
			if (pid == hParam.maxpnum)
				return;
			if (pid == numParticles) {
				hUniqueId[pid] = numParticles++;
				hJetFlag[pid] = 0;
			}
			else {
				numParticlesDeleted--;
			}
			
			hPos[pid] = hPos[i]+cfloat3(0,-1,0);
			hVel[pid] = cfloat3(0,0,0);
			hColor[pid] = cfloat4(0.7, 0.75, 0.95, 0.8);
			hMass[pid] = hParam.emitThres;
			hDripBuf[i] -= hParam.emitThres;

			hType[pid] = TYPE_FLUID;
			hInvMass[pid] = 1 / hMass[pid];
			pid++;
		}
	}

	copy2Device_partial(start, pid);
}

void PBFSolver::copy2Host() {
	
	cudaMemcpy(hMass,		dData.mass,			sizeof(float)*numParticles,		cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos,		dData.pos,			sizeof(cfloat3)*numParticles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel,		dData.vel,			sizeof(cfloat3)*numParticles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(hUniqueId,	dData.uniqueId,		sizeof(int)*numParticles,		cudaMemcpyDeviceToHost);
	cudaMemcpy(hIndexTable, dData.indexTable,	hParam.maxpnum * sizeof(int),	cudaMemcpyDeviceToHost);
	cudaMemcpy(hColor,		dData.color,		numParticles * sizeof(cfloat4), cudaMemcpyDeviceToHost);


	if (bModelPorous) {
		cudaMemcpy(hType,		dData.type,		sizeof(int)*numParticles,	cudaMemcpyDeviceToHost);
		cudaMemcpy(hDripBuf,	dData.dripBuf,	sizeof(float)*numParticles, cudaMemcpyDeviceToHost);

		removeInvalidParticles();
		emitWaterFromCloth();
	
		cudaMemcpy(dData.mass,		hMass,		sizeof(float)*numParticles,		cudaMemcpyHostToDevice);
		cudaMemcpy(dData.type,		hType,		sizeof(int)*numParticles,		cudaMemcpyHostToDevice);
		cudaMemcpy(dData.pos,		hPos,		sizeof(cfloat3)*numParticles,	cudaMemcpyHostToDevice);
		cudaMemcpy(dData.dripBuf,	hDripBuf,	sizeof(float)*numParticles,		cudaMemcpyHostToDevice);
	}

	//Rendering
	for (int i = 0; i < numParticles; i++) {
		hPosRender[i] = hPos[hIndexTable[i]];
		hColorRender[i] = hColor[hIndexTable[i]];
	}

}

#define GRID_UNDEF 99999999

void PBFSolver::sortBaryCenter() {
	
	sortBaryCenterCUDA(dData, numTriangles, numGridCells);

	cudaMemcpy(dData.baryCenter,			dData.sortedBaryCenter,				numTriangles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.baryCenterTriangleId,	dData.sortedBaryCenterTriangleId,	numTriangles * sizeof(int),		cudaMemcpyDeviceToDevice);
}

void PBFSolver::sort() {
	calcHash(dData,numParticles);

	sortParticle(dData,numParticles);

	cudaMemset(dData.gridCellCollisionFlag, 0, sizeof(char)*numGridCells);

	reorderDataAndFindCellStart(dData,numParticles,numGridCells);

	cudaMemcpy(dData.pos,		dData.sortedPos,		numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.oldPos,	dData.sortedOldPos,		numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.vel,		dData.sortedVel,		numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.color,		dData.sortedColor,		numParticles * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.type,		dData.sortedType,		numParticles * sizeof(int),		cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.group,		dData.sortedGroup,		numParticles * sizeof(int),		cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.invMass,	dData.sortedInvMass,	numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.mass,		dData.sortedMass,		numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.uniqueId,	dData.sortedUniqueId,	numParticles * sizeof(int),		cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.jetFlag,	dData.sortedJetFlag,	numParticles * sizeof(char),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.absorbBuf, dData.sortedAbsorbBuf,	numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.dripBuf,   dData.sortedDripBuf,	numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
}

void PBFSolver::solvePBF() {
	//Solve fluid constraints.
	for (int i = 0; i < 3; i++) {
		calcLambda(dData, numParticles);
		calcDeltaPos(dData, numParticles);
		updatePos(dData, numParticles);
	}

	//collision detection
	detectCollision(dData, numParticles);
	updatePos(dData, numParticles);
	

	//Update Velocity.
	updateVel(dData, numParticles, hParam.dt);
	applyXSPH(dData, numParticles);

}


void PBFSolver::step() {
	if (numParticles > 0) {

		predictPosition( dData, hParam.dt, numParticles);

		sort();

		solvePBF();

		copy2Host();
	}

	//'e' emit particle
	if (bEmitParticle) {
		fluidSrcEmit();
	}

	frameNo++;
	time += hParam.dt;
}








inline float rand_in_range(float min, float max) {
	return min + (float)rand()/RAND_MAX * (max-min);
}

void PBFSolver::copy2host_full() {
	//sync data
	cudaMemcpy(hPos, dData.pos,		sizeof(cfloat3)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hColor, dData.color, sizeof(cfloat4)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dData.vel,		sizeof(cfloat3)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hType, dData.type,	sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hInvMass, dData.invMass,		sizeof(float)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hUniqueId, dData.uniqueId,	sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hJetFlag, dData.jetFlag, sizeof(char)*numParticles, cudaMemcpyDeviceToHost);
}


void PBFSolver::emitParticle() {
	//lock source facet
	cfloat3 norm;
	cfloat3 srcpos;
	
	cfloat3 x1,x2,x3;
	x1 = hPos[hIndexTable[799]];
	x2 = hPos[hIndexTable[792]];
	x3 = hPos[hIndexTable[798]];

	cfloat3 x1x2, x1x3;
	x1x2 = x2-x1;
	x1x3 = x3-x1;
	norm = cross(x1x2, x1x3); //outward
	norm  = norm * (-1); //inward

	srcpos = (x1+x2+x3)/3.0f; //barycentric center
	//srcpos = cfloat3((float)rand()/RAND_MAX,10, (float)rand()/RAND_MAX);
	//norm = cfloat3(0,-1,0);
	float area = norm.mode()*3;
	norm = norm / norm.mode();
	float speed = 30;
	
	int frameinterval = roundf( 1 / (area*speed*hParam.dt));
	int start = numParticles;
	//frameinterval = 10;

	if (frameNo % 1==0) {
		int pid = addDefaultParticle();
		hPos[pid] = srcpos + norm*0.5;
		hVel[pid] = cfloat3(0,-0.1,0);//norm;// * speed;
		hColor[pid] = cfloat4(0.7, 0.75, 0.95, 0.8);
		hMass[pid] = hParam.restdensity*pow(hParam.spacing, 3);
		hType[pid] = TYPE_FLUID;
		hInvMass[pid] = 1 / hMass[pid];
	}

	copy2Device_partial(start, numParticles);
}

void PBFSolver::copy2Device_partial(int begin, int end) {
	int len = end - begin;
	
	cudaMemcpy(dData.pos+begin,		hPos+begin,		len * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.vel+begin,		hVel+begin,		len * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color+begin,	hColor + begin,		len * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.type+begin,	hType + begin,		len * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.mass+begin,	hMass + begin,		len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.invMass+begin,	hInvMass + begin,	len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.uniqueId+begin,	hUniqueId + begin, len * sizeof(int), cudaMemcpyHostToDevice);
	
}
#ifndef PBF_KERNEL
#define PBF_KERNEL

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "catpaw/geometry.h"
#include "pbf_gpu.cuh"

typedef unsigned int uint;


SimParam hParam;
__device__ SimParam dParam; 

#define GRID_UNDEF 99999999

__device__ uint calcGridHash(cint3 gridPos)
{
	gridPos.x = gridPos.x & (dParam.gridres.x - 1);  // wrap grid, assumes size is power of 2
	gridPos.y = gridPos.y & (dParam.gridres.y - 1);
	gridPos.z = gridPos.z & (dParam.gridres.z - 1);
	return gridPos.y * dParam.gridres.x* dParam.gridres.z + gridPos.z*dParam.gridres.x + gridPos.x;
}

__device__ cint3 calcGridPos(cfloat3 p) {
	cint3 gridPos;
	gridPos.x = floor((p.x - dParam.gridxmin.x) / dParam.dx);
	gridPos.y = floor((p.y - dParam.gridxmin.y) / dParam.dx);
	gridPos.z = floor((p.z - dParam.gridxmin.z) / dParam.dx);
	return gridPos;
}

__global__ void calcHashD(
	int* ParticleHash,
	int* ParticleIndex,
	cfloat3* Pos,
	int pnum) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= pnum)	return;

	cfloat3 x = Pos[i];
	uint hash;

	if(x.x<-999)
		hash = GRID_UNDEF;
	else {
		cint3 gridPos = calcGridPos(x);
		hash = calcGridHash(gridPos);
	}
	

	ParticleHash[i] = hash;
	ParticleIndex[i] = i;
}
__global__ void reorderDataAndFindCellStartD(
	SimData data,
	int numParticles
) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	extern __shared__ uint sharedHash[];
	uint hash;
	       
	if (index < numParticles)
	{
		hash = data.particleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = data.particleHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			if(hash!=GRID_UNDEF)
				data.gridCellStart[hash] = index;

			if (index > 0)
				data.gridCellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			if(hash!=GRID_UNDEF)
				data.gridCellEnd[hash] = index + 1;
		}

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = data.particleIndex[index];
		cfloat3 pos = data.pos[sortedIndex];       // macro does either global read or texture fetch
		cfloat3 vel = data.vel[sortedIndex];       // see particles_kernel.cuh

		data.sortedPos[index] = pos;
		data.sortedOldPos[index] = data.oldPos[sortedIndex];
		data.sortedVel[index] = vel;
		data.sortedColor[index] = data.color[sortedIndex];
		data.sortedType[index] = data.type[sortedIndex];
		data.sortedGroup[index] = data.group[sortedIndex];
		data.sortedInvMass[index] = data.invMass[sortedIndex];
		data.sortedMass[index] = data.mass[sortedIndex];
		data.sortedUniqueId[index] = data.uniqueId[sortedIndex];
		data.sortedJetFlag[index] = data.jetFlag[sortedIndex];
		data.sortedAbsorbBuf[index] = data.absorbBuf[sortedIndex];
		data.sortedDripBuf[index] = data.dripBuf[sortedIndex];
		data.indexTable[data.sortedUniqueId[index]] = index;

	}
}

__global__ void calcBaryCenterHashD(
	SimData data,
	int numTriangles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numTriangles)	return;

	//update barycenter position
	int triId = data.baryCenterTriangleId[i];

	int pid[3];
	cfloat3 x(0, 0, 0);
	for (int j = 0; j < 3; j++) {
		pid[j] = data.indexTable[data.triangles[triId].plist[j]];
		x += data.pos[pid[j]];
	}
	x = x / 3.0f;
	data.baryCenter[i] = x;

	uint hash;
	cint3 gridPos = calcGridPos(x);
	hash = calcGridHash(gridPos);

	data.baryCenterHash[i] = hash;
	data.baryCenterIndex[i] = i;
}

__global__ void reorderBaryCenterAndFindCellStartD(
	SimData data,
	int numTriangles
) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	extern __shared__ uint sharedHash[];
	uint hash;

	if (index < numTriangles)
	{
		hash = data.baryCenterHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = data.baryCenterHash[index - 1];
		}
	}

	__syncthreads();

	if (index < numTriangles)
	{
		
		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			if (hash != GRID_UNDEF)
				data.gridCellStartBaryCenter[hash] = index;

			if (index > 0)
				data.gridCellEndBaryCenter[sharedHash[threadIdx.x]] = index;
		}

		if (index == numTriangles - 1)
		{
			if (hash != GRID_UNDEF)
				data.gridCellEndBaryCenter[hash] = index + 1;
		}

		// Now use the sorted index to reorder the bary center data
		uint sortedIndex = data.baryCenterIndex[index];
		
		data.sortedBaryCenter[index] = data.baryCenter[sortedIndex];
		data.sortedBaryCenterTriangleId[index] = data.baryCenterTriangleId[sortedIndex];
	}
}






__global__ void predictPositionD(
	SimData data,
	float dt,
	int numParticles
)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if(data.type[index]==TYPE_NULL) return;

	if (data.invMass[index] < 0.00001) {
		data.vel[index].Set(0, 0, 0);
		data.oldPos[index] = data.pos[index];
	}
	else {
			
		if(data.type[index]==TYPE_CLOTH)
			data.vel[index] *= dParam.k_damping;
		//else
		data.vel[index] += dParam.gravity*dt;

		data.oldPos[index] = data.pos[index];
		data.pos[index] += data.vel[index] * dt;
	}
		
}


__device__ void lambdaCell(
	cint3 gridPos,
	int index,
	cfloat3 pos,
	SimData data,

	float& gradCsum,
	cfloat3& gradCi,
	float& density
)
{
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float restden = dParam.restdensity;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = data.gridCellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index && (data.type[j]==TYPE_FLUID))
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2 || d2 < 0.000001)
					continue;

				float d = sqrt(d2);

				float c2 = sr2 - d2;
				density += c2*c2*c2;// * dParam.pdata[pj].m;

				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d / restden;

				cfloat3 gradCj = xij*nablaw;

				gradCsum += gradCj.x*gradCj.x + gradCj.y*gradCj.y + gradCj.z*gradCj.z;
				gradCi += xij*nablaw;

			}
		}
	}
}


__global__ void calcLambdaD(
	SimData data,
	int numParticles
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index] != TYPE_FLUID) return;

	cfloat3 pos = data.pos[index];


	cint3 gridPos = calcGridPos(pos);
	float lambda = 0;
	cfloat3 gradCi(0, 0, 0);
	float gradCsum = 0;
	float density = 0;

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				lambdaCell(neighbourPos,
					index,
					pos,
					data,
					gradCsum,
					gradCi,
					density);
			}
		}
	}

	float sr2 = dParam.smoothradius * dParam.smoothradius;
	density += sr2*sr2*sr2;
	density *= dParam.kpoly6;
	gradCsum += gradCi.x*gradCi.x + gradCi.y*gradCi.y + gradCi.z*gradCi.z;

	lambda = -(density / dParam.restdensity - 1) / (gradCsum + dParam.pbfepsilon);
	//if (lambda > 0)
	//	lambda = 0;

	data.lambda[index] = lambda;
}

	
__device__ cfloat3 deltaPosCell(
	cint3 gridPos,
	int index,
	cfloat3 pos,

	SimData data) {

	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float restden = dParam.restdensity;

	float q = sr* dParam.qfactor;
	float wq = dParam.kpoly6 * pow((sr2 - q*q), 3);
	float k = dParam.k_ti;
	float n = dParam.n_ti;
	cfloat3 deltaPos(0, 0, 0);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = data.gridCellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index && data.type[j]==TYPE_FLUID)                // check not colliding with self
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2 || d2 < 0.000001)
					continue;

				float d = sqrt(d2);

				float c2 = sr2 - d2;

				float w = dParam.kpoly6 * c2 * c2 * c2;
				float ratio = w / wq;
				float scorr = -k * pow(ratio, n);

				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d;

				deltaPos += xij * nablaw * (data.lambda[index] + data.lambda[j] + scorr) / restden;
			}
		}
	}
	return deltaPos;
}

__global__ void calcDeltaPosD(
	SimData data,
	int numParticles
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index] != TYPE_FLUID) return;
	cfloat3 pos = data.pos[index];


	cint3 gridPos = calcGridPos(pos);
	cfloat3 deltapos(0, 0, 0);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				deltapos += deltaPosCell(
					neighbourPos,
					index,
					pos,
					data);
			}
		}
	}

	//cfloat3 dx = clampBoundary(pos);

	data.deltaPos[index] = deltapos;// +dx;
	data.numCons[index] += 1;
}


__device__ cfloat3 clampBoundary(cfloat3 pos) {
	cfloat3 dx(0, 0, 0);
	//y
	float diff = dParam.softminx.y - pos.y;
	if (diff > 0) {
		dx += cfloat3(0, diff, 0);
	}

	diff = pos.y - dParam.softmaxx.y;
	if (diff > 0) {
		dx += cfloat3(0, -diff, 0);
	}

	//x
	diff = dParam.softminx.x - pos.x;
	if (diff > 0) {
		dx += cfloat3(diff, 0, 0);
	}
	diff = -(dParam.softmaxx.x - pos.x);
	if (diff > 0) {
		dx += cfloat3(-diff, 0, 0);
	}
	//z
	diff = dParam.softminx.z - pos.z;
	if (diff > 0) {
		dx += cfloat3(0, 0, diff);
	}
	diff = -dParam.softmaxx.z + pos.z;
	if (diff > 0) {
		dx += cfloat3(0, 0, -diff);
	}
	return dx;
}

__global__ void updatePosD(
	SimData data,
	int numParticles
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if(data.type[index]==TYPE_NULL) return;

	cfloat3 clamp = clampBoundary(data.pos[index]);

	if (data.numCons[index] > 0) {
		data.pos[index] += data.avgDeltaPos[index] / data.numCons[index];
		data.numCons[index] = 0;
		data.avgDeltaPos[index].Set(0, 0, 0);
	}
	
	/*if(data.type[index]==TYPE_FLUID)
		printf("1: %f %f %f\n", data.pos[index].x, 
			data.pos[index].y, data.pos[index].z);*/

	data.pos[index] += data.deltaPos[index] * dParam.global_relaxation;

	/*if (data.type[index]==TYPE_FLUID && data.deltaPos[index].mode()>EPSILON)
		printf("2: %f %f %f, %f %f %f\n", 
			data.deltaPos[index].x,
			data.deltaPos[index].y,
			data.deltaPos[index].z,
			data.pos[index].x,
			data.pos[index].y, 
			data.pos[index].z);*/

	data.deltaPos[index].Set(0, 0, 0);

	data.pos[index] += clamp;
}

	
	
__global__ void updateVelD(
	SimData data,
	int numParticles,
	float dt
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if(data.type[index]==TYPE_NULL) return;

	data.vel[index] = (data.pos[index] - data.oldPos[index]) / dt;
}


__device__ cfloat3 xsphCell(
	cint3 gridPos,
	int index,
	cfloat3 pos,

	SimData data
) {
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	cfloat3 xsph(0, 0, 0);

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = data.gridCellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index && data.type[j]==TYPE_FLUID)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2 || d2 < 0.000001)
					continue;

				float d = sqrt(d2);

				float c2 = sr2 - d2;

				float w = dParam.kpoly6 * c2 * c2 * c2;
				cfloat3 vij = data.vel[j] - data.vel[index];

				xsph += vij *w;
			}
		}
	}
	return xsph;
}


__global__ void applyXSPHD(
	SimData data,
	int numParticles
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index] != TYPE_FLUID) return;

	cfloat3 p = data.pos[index];

	cint3 gridPos = calcGridPos(p);
	cfloat3 xsph(0, 0, 0);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				xsph += xsphCell(
					neighbourPos,
					index,
					p,
					data);
			}
		}
	}

	data.vel[index] += xsph * dParam.viscosity;
}

__inline__ __device__ void atomicadd_float3(cfloat3& vec, cfloat3 t) {
	atomicAdd((float*)&vec.x, t.x);
	atomicAdd((float*)&vec.y, t.y);
	atomicAdd((float*)&vec.z, t.z);
}

__global__ void calcEdgeConsD(
	SimData data,
	int numEdgeCons
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numEdgeCons)	return;
		
	//stretch constraint
	edgeConstraint& ec = data.edgeCons[index];

	int pid1 = data.indexTable[ec.p1];
	int pid2 = data.indexTable[ec.p2];

	cfloat3 p1 = data.pos[pid1];
	cfloat3 p2 = data.pos[pid2];
	float w1 = data.invMass[pid1];
	float w2 = data.invMass[pid2];


	if (w1<EPSILON && w2<EPSILON)
		return;


	cfloat3 deltap1, deltap2;
	cfloat3 p2p1 = p1 - p2;
	float c = (p2p1.mode() - ec.L0)/p2p1.mode();
	//float c = (p2p1.mode() - ec.L0);
		
	if (c>0) {
		c *= dParam.stretchStiff;
	}
	else {
		c *= dParam.compressStiff;
	}

	deltap1 = p2p1 * c * (-1) * w1 / (w1 + w2);
	deltap2 = p2p1 * c * w2 / (w1 + w2);
		


	//bending constraint
	if (ec.p3 != -1) {
		int pid3 = data.indexTable[ec.p3];
		int pid4 = data.indexTable[ec.p4];
		cfloat3 p3 = data.pos[pid3];
		cfloat3 p4 = data.pos[pid4];
		float w3 = data.invMass[pid3];
		float w4 = data.invMass[pid4];
		//substract p1 from all positions to get simpler expressions
		p2 = p2 - p1;
		p3 = p3 - p1;
		p4 = p4 - p1;
		cfloat3 _n1 = cross(p2, p3);
		cfloat3 n1 = _n1 / _n1.mode();
		cfloat3 _n2 = cross(p2, p4);
		cfloat3 n2 = _n2 / _n2.mode();
		float d = dot(n1, n2);
		if (d>1)
			d = 1;
		if (d<-1)
			d = -1;

		float t1, t2;
		t1 = cross(p2, p3).mode();
		t2 = cross(p2, p4).mode();

		cfloat3 q3 = (cross(p2, n2) + cross(n1, p2)*d) / t1;
		cfloat3 q4 = (cross(p2, n1) + cross(n2, p2)*d) / t2;
		cfloat3 q2 = (cross(p3, n2) + cross(n1, p3)*d) / t1*(-1)
			- (cross(p4, n1) + cross(n2, p4)*d) / t2;
		cfloat3 q1 = q2*(-1) - q3 - q4;

		float denom = w1* pow(q1.mode(), 2)
			+ w2 * pow(q2.mode(), 2)
			+ w3 * pow(q3.mode(), 2)
			+ w4 * pow(q4.mode(), 2);

		float nom = -sqrt(1 - d*d)  * (acos(d) - ec.Phi0);
		float factor = nom / fmax(denom, 0.000001f);

		factor *= dParam.bendingstiff;

		deltap1 += q1 * w1 * factor;
		deltap2 += q2 * w2 * factor;
		cfloat3 deltap3 = q3 * w3 * factor;
		cfloat3 deltap4 = q4 * w4 * factor;

		atomicadd_float3(data.deltaPos[pid1], deltap1);
		atomicadd_float3(data.deltaPos[pid2], deltap2);
		atomicadd_float3(data.deltaPos[pid3], deltap3);
		atomicadd_float3(data.deltaPos[pid4], deltap4);
		//atomicAdd(&dParam.pdata[ec.p1].constraintnum, 2);
		//atomicAdd(&dParam.pdata[ec.p2].constraintnum, 2);
		//atomicAdd(&dParam.pdata[ec.p3].constraintnum, 1);
		//atomicAdd(&dParam.pdata[ec.p4].constraintnum, 1);
	}
	else {
		atomicadd_float3(data.deltaPos[pid1], deltap1);
		atomicadd_float3(data.deltaPos[pid2], deltap2);
		//atomicAdd(&dParam.pdata[ec.p1].constraintnum, 1);
		//atomicAdd(&dParam.pdata[ec.p2].constraintnum, 1);
	}

}




__global__ void calcFacetVolD(SimData data, int numTriangles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numTriangles)	return;

	int pid1, pid2, pid3;
	pid1 = data.triangles[index].plist[0];
	pid2 = data.triangles[index].plist[1];
	pid3 = data.triangles[index].plist[2];
	pid1 = data.indexTable[pid1];
	pid2 = data.indexTable[pid2];
	pid3 = data.indexTable[pid3];

	cfloat3 x1, x2, x3;
	x1 = data.pos[pid1];
	x2 = data.pos[pid2];
	x3 = data.pos[pid3];
		
	cfloat3 temp;
	temp = cross(x1, x2);
	float vol = dot(temp, x3) / 6.0f;
	data.facetVol[index] = vol;
	data.facetArea[index] = cross(x1 - x2, x1 - x3).mode() * 0.5f;
}


__global__ void calcVolDeltaPosD(
	SimData data,
	int numTriangles
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numTriangles)	return;

	objtriangle& t = data.triangles[index];

	int pid[3];
	for (int i = 0; i < 3; i++)
		pid[i] = data.indexTable[t.plist[i]];
	

	cfloat3 x1, x2, x3;
	x1 = data.pos[pid[0]];
	x2 = data.pos[pid[1]];
	x3 = data.pos[pid[2]];
	float dx = data.objs[ t.objectId ].dx;
	bool jetGas = data.objs[t.objectId].bJetGas;

	cfloat3 norm = cross(x2 - x1, x3 - x1);
	if (norm.mode()>EPSILON) {
		norm = norm / norm.mode() * dx;
	}
	else {
		norm.Set(0,0,0);
	}
	

	for (int i = 0; i < 3; i++) {
		if (data.invMass[pid[i]] < EPSILON)
			continue;
		if (data.jetFlag[pid[i]] && jetGas)
			continue;
		
		atomicadd_float3(data.avgDeltaPos[pid[i]], norm);
		atomicAdd(&data.numCons[pid[i]], 1);
	}		
	return;
}

__device__ void collideCell(
	uint gridHash,
	int index,
	cfloat3 pos,
	SimData data
) {
	

	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];

	float w1 = data.invMass[index];
	float cd = 0;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		// iterate over particles in this cell
		uint endIndex = data.gridCellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d = xij.mode();

				if (data.type[index] == TYPE_FLUID){
					if(data.type[j] == TYPE_FLUID || data.type[j]==TYPE_EMITTER)
						continue;
					else
						cd = dParam.collisionDistance;
				}
				
				if (data.type[index]==TYPE_CLOTH){
					if (data.type[j] == TYPE_CLOTH) {
						if(data.group[j]==data.group[index])
							cd = dParam.cloth_selfcd;
						else
							cd = dParam.collisionDistance;
					}
						
					else
						cd = dParam.collisionDistance;
				}
						

				if (d >= cd)
					continue;

				float w2 = data.invMass[j];
				if ( (w1 + w2) < 0.000001)
					continue;
				
				//if (data.type[index] == TYPE_CLOTH && data.type[j] == TYPE_CLOTH)
				
				float push = cd - d;
				data.deltaPos[index] += xij * push * w1 / (w1 + w2) * dParam.collisionStiff;
				//else{
				//	data.avgDeltaPos[index] += xij * (cd - d) * w1 / (w1 + w2);
				//	data.numCons[index] += 1;
				//}
			}
		}
	}
}

__global__ void labelCollisionCellD(SimData data, int numParticles) {
	

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index]!=TYPE_CLOTH && data.type[index]!=TYPE_BOUNDARY) 
		return;

	cfloat3 p = data.pos[index];

	cint3 gridPos = calcGridPos(p);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				uint gridHash = calcGridHash(neighbourPos);
				atomicAdd(&data.gridCellCollisionFlag[gridHash], 1);
			}
		}
	}
}

__global__ void detectCollisionD(SimData data, int numParticles)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index]==TYPE_EMITTER) return;

	cfloat3 p = data.pos[index];
	cfloat3 deltaPos(0, 0, 0);

	cint3 gridPos = calcGridPos(p);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				uint gridHash = calcGridHash(neighbourPos);
				if (data.gridCellCollisionFlag[gridHash] == 0)
					continue;
				collideCell(gridHash, index, p,	data);
			}
		}
	}
}



__device__ void pointTriangleCollide(
	int index,
	objtriangle& t,
	cfloat3 pos,
	SimData data) {
	
	int pid[3];
	cfloat3 vertices[3];
	for (int k = 0; k < 3; k++) {
		pid[k] = data.indexTable[t.plist[k]];
		if(pid[k]==index)
			return;

		vertices[k] = data.pos[pid[k]];
	}

	cfloat3 v0 = vertices[1] - vertices[0]; // p2-p1
	cfloat3 v1 = vertices[2] - vertices[0]; // p3-p1
	cfloat3 v2 = pos - vertices[0];
	
	cfloat3 normal = cross(v0,v1);
	if(normal.mode()<EPSILON) //singular
		return;

	normal /= normal.mode();
	float distance = dot(normal, v2);
	float originDistance = dot(normal, data.oldPos[index]-vertices[0]);
	bool bAbove = true;
	bool bCross = distance*originDistance < 0;

	if (distance<0) //p under facet
	{
		bAbove = false;
		normal *= -1;
		distance *= -1;
	}
	cfloat3 pInPlane = v2 - normal * distance;
	
	//check inside triangle
	float dot00 = dot(v0,v0);
	float dot01 = dot(v0,v1);
	float dot02 = dot(v0,v2);
	float dot11 = dot(v1,v1);
	float dot12 = dot(v1,v2);
	float denom = dot00*dot11 - dot01*dot01;
	if (abs(denom)<EPSILON)
	{
		//printf("denom zero error 1!\n");
		printf("%d %f %f %f, %d %f %f %f\n", 
			pid[0], vertices[0].x, vertices[0].y, vertices[0].z,
			pid[2], vertices[2].x, vertices[2].y, vertices[2].z);
		return;
	}

	float u = (dot11*dot02 - dot01*dot12) / denom;
	float v = (dot00*dot12 - dot01*dot02) / denom;

	bool bInside;


	float collisionDistance;
	float softedge = 0;
	if (data.type[index]==TYPE_CLOTH)
		collisionDistance = dParam.clothThickness;
	else {
		collisionDistance = dParam.collisionDistance;
		//softedge = 1;
	}
		
	

	if(u>=-softedge && v>=-softedge && (u+v)<1+softedge)
		bInside = true;
	else
		bInside = false;
	

	if (bInside && (bCross || distance<collisionDistance)) {
		//handle collision
		//v0: p2, v1: p3, v2: p

		float C;
		if (bCross) {
			printf("cross detected.\n");
			C = -distance - collisionDistance;
		}
		else {
			C = distance - collisionDistance;
		}

		cfloat3 q, q1, q2, q3;
		cfloat3 n = cross(v0,v1);
		float n2 = n.x*n.x + n.y*n.y + n.z*n.z;
		float nlen = sqrt(n2);
		cfloat3 p3xn = cross(v1,n);
		cfloat3 p3xp = cross(v1,v2);
		q2.x = 1/n2 * ( p3xp.x * nlen - (n.y+n.z)/nlen*p3xn.x);
		q2.y = 1/n2 * ( p3xp.y * nlen - (n.x+n.z)/nlen*p3xn.y);
		q3.z = 1/n2 * ( p3xp.z * nlen - (n.x+n.y)/nlen*p3xn.z);

		cfloat3 nxp2 = cross(v0,n);
		cfloat3 pxp2 = cross(v2,v0);
		q3.x = 1/n2 * (pxp2.x * nlen - (n.y+n.z)/nlen*nxp2.x);
		q3.y = 1/n2 * (pxp2.y * nlen - (n.x+n.z)/nlen*nxp2.y);
		q3.z = 1/n2 * (pxp2.z * nlen - (n.x+n.y)/nlen*nxp2.z);

		q = n / nlen;

		q1 = q*(-1) -q2 -q3;

		float w,w1,w2,w3;
		w = data.invMass[index];
		w1 = data.invMass[pid[0]];
		w2 = data.invMass[pid[1]];
		w3 = data.invMass[pid[2]];
		//w1=w2=w3=0;

		if(w+w1+w2+w3<EPSILON)
			return;

		cfloat3 deltap1, deltap2, deltap3, deltap;
		
		denom = w1* pow(q1.mode(), 2)
			+ w2 * pow(q2.mode(), 2)
			+ w3 * pow(q3.mode(), 2)
			+ w * pow(q.mode(), 2);
		if (denom<EPSILON) {
			printf("denom zero error 2!\n");
			return;
		}

		float factor = dParam.collisionStiff * C*(-1) / denom;
		if(!bAbove)
			factor *= -1;

		deltap1 = q1 * w1 * factor;
		deltap2 = q2 * w2 * factor;
		deltap3 = q3 * w3 * factor;
		deltap = q * w * factor;

		//if (data.type[index]==TYPE_FLUID) {
		//	printf("%f %f %f\n", deltap.x, deltap.y, deltap.z);
		//}
		
		//printf("%f %f %f\n",deltap.x, deltap.y, deltap.z);
		
		atomicadd_float3(data.deltaPos[pid[0]], deltap1);
		atomicadd_float3(data.deltaPos[pid[1]], deltap2);
		atomicadd_float3(data.deltaPos[pid[2]], deltap3);
		atomicadd_float3(data.deltaPos[index], deltap);
		//atomicAdd(&data.numCons[index],1);
		//atomicAdd(&data.numCons[pid[0]], 1);
		//atomicAdd(&data.numCons[pid[1]], 1);
		//atomicAdd(&data.numCons[pid[2]], 1);

	}
}

__device__ void collideWithMeshCell(cint3 gridPos, int index, cfloat3 pos, SimData data) {
	uint gridHash = calcGridHash(gridPos);

	uint startIndex = data.gridCellStartBaryCenter[gridHash];

	float w1 = data.invMass[index];
	float collisionDistance;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		uint endIndex = data.gridCellEndBaryCenter[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			int triangleId = data.baryCenterTriangleId[j];
			objtriangle& t = data.triangles[triangleId];
			
			pointTriangleCollide(index, t, pos, data);
		}
	}
}

__global__ void detectCollisionWithMeshD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;

	cfloat3 p = data.pos[index];
	cfloat3 deltaPos(0, 0, 0);

	cint3 gridPos = calcGridPos(p);
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				collideWithMeshCell(
					neighbourPos,
					index,
					p,
					data);
			}
		}
	}
}



__global__ void calcFacetNormalD(SimData data, int numTriangles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numTriangles)	return;

	int pid[3];
	for (int i = 0; i < 3; i++)
		pid[i] = data.indexTable[data.triangles[index].plist[i]];


	cfloat3 x1, x2, x3;
	x1 = data.pos[pid[0]];
	x2 = data.pos[pid[1]];
	x3 = data.pos[pid[2]];

	cfloat3 norm = cross(x2 - x1, x3 - x1);
	if (norm.mode()>EPSILON)
		norm = norm / norm.mode();
	else
		norm.Set(0, 0, 0);

	for (int i = 0; i < 3; i++) {
		atomicadd_float3(data.normal[pid[i]], norm);
	}
}

__global__ void calcParticleNormalD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index] != TYPE_CLOTH) return;

	data.normal[index] = data.normal[index] / data.normal[index].mode();
}



__global__ void calcStablizeDeltaPosD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if (data.type[index] == TYPE_NULL) return;
		
	cfloat3 pos = data.pos[index];

	cint3 gridPos = calcGridPos(pos);
	cfloat3 deltapos(0, 0, 0);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				uint gridHash = calcGridHash(neighbourPos);
				collideCell(
					gridHash,
					index,
					pos,
					data);
			}
		}
	}
}

__global__ void updateStablizePosD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)	return;
	if(data.type[index]==TYPE_NULL) return;

	/*if (data.numCons[index] > 0) {
		data.pos[index] += data.avgDeltaPos[index] / data.numCons[index];
		data.oldPos[index] += data.avgDeltaPos[index] / data.numCons[index];
		data.avgDeltaPos[index].Set(0, 0, 0);
		data.numCons[index] = 0;
	}*/
	
	data.pos[index] += data.deltaPos[index] * 0.35;
	data.oldPos[index] += data.deltaPos[index] * 0.35;
	data.deltaPos[index].Set(0, 0, 0);
	

	
	
}



__global__ void resetEdgeConsXD(SimData data, int numEdgeCons) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numEdgeCons)
		return;
	edgeConstraint& ec = data.edgeCons[i];
	edgeConsVar& ecVar = data.edgeConsVar[i];
	ecVar.lambda1 = 0;
	ecVar.lambda2 = 0;


	int pid1 = data.indexTable[ec.p1];
	int pid2 = data.indexTable[ec.p2];
	cfloat3 p1 = data.pos[pid1];
	cfloat3 p2 = data.pos[pid2];
	cfloat3 p2p1 = p1-p2;
	float c = p2p1.mode() - ec.L0;
	//if(c>=0)
	ecVar.stiff1 = dParam.stretchComp;
	//else
	//	ecVar.stiff1 = dParam.compressComp;
	ecVar.stiff2 = dParam.bendingstiff;
	//ecVar.stiff1 = 0;
	//ecVar.stiff2 = 0;
}
__global__ void calcEdgeConsXD(SimData data, int numEdgeCons) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numEdgeCons)
		return;

	//stretch constraint
	edgeConstraint& ec = data.edgeCons[i];
	edgeConsVar& ecVar = data.edgeConsVar[i];
	int pid1 = data.indexTable[ec.p1];
	int pid2 = data.indexTable[ec.p2];

	cfloat3 p1 = data.pos[pid1];
	cfloat3 p2 = data.pos[pid2];
	float w1 = data.invMass[pid1];
	float w2 = data.invMass[pid2];

	if (w1<EPSILON && w2<EPSILON)
		return;


	cfloat3 deltap1, deltap2;
	float dlambda;
	cfloat3 p2p1 = p1-p2;
	float c = (p2p1.mode() - ec.L0)/ec.L0;
	if(c<EPSILON)
		c=0;
	float stiff = ecVar.stiff1;
	stiff /= dParam.dt*dParam.dt;

	dlambda = (c*(-1) - stiff * ecVar.lambda1)/((w1+w2)/ec.L0/ec.L0 + stiff);
	ecVar.lambda1 += dlambda;
	p2p1 = p2p1 / p2p1.mode(); //normalize
	deltap1 = p2p1 * dlambda * w1/ec.L0;
	deltap2 = p2p1 * dlambda*(-1) * w2/ec.L0;



	//bending constraint
	if (ec.p3 != -1) {
		int pid3 = data.indexTable[ec.p3];
		int pid4 = data.indexTable[ec.p4];
		cfloat3 p3 = data.pos[pid3];
		cfloat3 p4 = data.pos[pid4];
		float w3 = data.invMass[pid3];
		float w4 = data.invMass[pid4];
		//substract p1 from all positions to get simpler expressions
		p2 = p2 - p1;
		p3 = p3 - p1;
		p4 = p4 - p1;
		cfloat3 _n1 = cross(p2, p3);
		cfloat3 n1 = _n1 / _n1.mode();
		cfloat3 _n2 = cross(p2, p4);
		cfloat3 n2 = _n2 / _n2.mode();
		float d = dot(n1, n2);
		if (d>1)
			d=1;
		if (d<-1)
			d=-1;

		float t1, t2;
		t1 = cross(p2, p3).mode();
		t2 = cross(p2, p4).mode();

		cfloat3 q3 = (cross(p2, n2)+cross(n1, p2)*d) / t1;
		cfloat3 q4 = (cross(p2, n1)+cross(n2, p2)*d) / t2;
		cfloat3 q2 =  (cross(p3, n2)+cross(n1, p3)*d) / t1*(-1)
			-(cross(p4, n1)+cross(n2, p4)*d) / t2;
		cfloat3 q1 = q2*(-1) - q3 - q4;

		stiff = ecVar.stiff2 / dParam.dt / dParam.dt;
		float denom = w1* pow(q1.mode(), 2)
			+ w2 * pow(q2.mode(), 2)
			+ w3 * pow(q3.mode(), 2)
			+ w4 * pow(q4.mode(), 2);
		denom += stiff;
		float nom = -sqrt(1-d*d)  * (acos(d) - ec.Phi0) - stiff * ecVar.lambda2;
		dlambda = nom / denom;
		ecVar.lambda2 += dlambda;

		deltap1 += q1 * dlambda * w1;
		deltap2 += q2 * dlambda * w2;
		cfloat3 deltap3 = q3 * dlambda * w3;
		cfloat3 deltap4 = q4 * dlambda * w4;

		atomicadd_float3(data.deltaPos[pid1], deltap1);
		atomicadd_float3(data.deltaPos[pid2], deltap2);
		atomicadd_float3(data.deltaPos[pid3], deltap3);
		atomicadd_float3(data.deltaPos[pid4], deltap4);
		/*atomicAdd(&pp1.constraintnum, 1);
		atomicAdd(&pp2.constraintnum, 1);
		atomicAdd(&pp3.constraintnum, 1);
		atomicAdd(&pp4.constraintnum, 1);*/
	}
	else {
		atomicadd_float3(data.deltaPos[pid1], deltap1);
		atomicadd_float3(data.deltaPos[pid2], deltap2);
	}


}


__global__ void calcRubberEdgeConsXD(SimData data, int numEdgeCons) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numEdgeCons)
		return;

	//stretch constraint
	edgeConstraint& ec = data.edgeCons[i];
	edgeConsVar& ecVar = data.edgeConsVar[i];
	int pid1 = data.indexTable[ec.p1];
	int pid2 = data.indexTable[ec.p2];

	cfloat3 p1 = data.pos[pid1];
	cfloat3 p2 = data.pos[pid2];
	float w1 = data.invMass[pid1];
	float w2 = data.invMass[pid2];

	if (w1<EPSILON && w2<EPSILON)
		return;


	cfloat3 deltap1, deltap2;
	float dlambda;
	cfloat3 p2p1 = p1-p2;
	float L = p2p1.mode();
	float F;
	float dFdx;

	float c = (p2p1.mode() - ec.L0);
	
	float stiff = ecVar.stiff1;
	stiff /= dParam.dt*dParam.dt;

	//dlambda = (c*(-1) - stiff * ecVar.lambda1)/(w1+w2+stiff);
	float k = 2;
	F = stiff/ec.L0/ec.L0/L * (1 - pow(ec.L0/L,k));
	dFdx = stiff/ec.L0/ec.L0/L/L * (k*pow(ec.L0/L,k+1)-1);
	if (c<EPSILON)
	{
		F = 0;
		dFdx = 0;
	}
	dlambda =  (F - ecVar.lambda1)/(dFdx*(w1+w2)+1);
	
	ecVar.lambda1 += dlambda;
	
	if (i==0) {
		printf("%f %f %f\n",F,L);
	}
	p2p1 = p2p1 / p2p1.mode(); //normalize
	deltap1 = p2p1 * dlambda * w1 *(-1);
	deltap2 = p2p1 * dlambda * w2;
	

	atomicadd_float3(data.deltaPos[pid1], deltap1);
	atomicadd_float3(data.deltaPos[pid2], deltap2);

}


__device__ void waterAbsorbCell(cint3 gridPos, int index, cfloat3 pos, SimData data) {
	uint gridHash = calcGridHash(gridPos);
	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];

	
	float vol =  data.mass[index]/dParam.cloth_density;
	float capacity = vol*dParam.max_saturation*dParam.restdensity; //water mass
	float cd  = dParam.collisionDistance;
	float waterMass = data.absorbBuf[index] + data.dripBuf[index];
	if (capacity - waterMass<EPSILON)
		return;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		uint endIndex = data.gridCellEnd[gridHash];
		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index && data.type[j]==TYPE_FLUID)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d = xij.mode();
		
				if (d >= cd)
					continue;

				float massj = data.mass[j];

				float dmass = (capacity-waterMass)*dParam.k_absorb*dParam.dt;
				//printf("%f\n", dmass);
				
				if (dmass>massj)
					dmass = massj;
				if(dmass<EPSILON)
					continue;
				
				atomicAdd(&data.mass[j], -dmass);
				data.absorbBuf[index] += dmass;
				waterMass = data.absorbBuf[index] + data.dripBuf[index];

				if(capacity - waterMass<EPSILON)
					break;
			}
		}
	}
}

__global__ void waterAbsorptionD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	if(data.type[index]!=TYPE_CLOTH) return;

	cfloat3 pos = data.pos[index];

	cint3 gridPos = calcGridPos(pos);
	

	float maxSaturation = dParam.max_saturation;

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				waterAbsorbCell(neighbourPos,
					index,
					pos,
					data);
			}
		}
	}

	float vol = data.mass[index] / dParam.cloth_density;

	float capacity = vol*dParam.max_saturation*dParam.restdensity;
	float waterMass = data.absorbBuf[index] + data.dripBuf[index];
	
	float poreMassLimit = vol*dParam.cloth_porosity*dParam.restdensity;

	if (data.absorbBuf[index] > poreMassLimit+EPSILON) {
		data.absorbBuf[index] = poreMassLimit;
		data.dripBuf[index] = waterMass - poreMassLimit;
	}
	//printf("%f %f\n", data.absorbBuf[index], data.dripBuf[index]);

	
}

__device__ void waterDiffusePredictCell(cint3 gridPos, int index, cfloat3 pos, SimData data) {
	uint gridHash = calcGridHash(gridPos);

	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];


	float vol = data.mass[index] / dParam.cloth_density;
	float poreMassLimit = vol*dParam.cloth_porosity*dParam.restdensity; //water mass
	float cd = dParam.diffuseDistance;
	float waterMass = data.absorbBuf[index] + data.dripBuf[index];
	float sat = data.absorbBuf[index] / poreMassLimit;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		uint endIndex = data.gridCellEnd[gridHash];
		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index && data.type[j] == TYPE_CLOTH)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d = xij.mode();

				if (d >= cd)
					continue;

				float volj = data.mass[j] / dParam.cloth_density * dParam.restdensity;

				float satj = data.absorbBuf[j] / (volj * dParam.cloth_porosity);
				

				float cosij = dot(xij, dParam.gravity)/d;
				//i -> j
				
				if (satj < 1 - EPSILON) {
					float absorbDiffuse = 0;
					if (sat > satj + EPSILON)
						absorbDiffuse += (sat - satj)  * dParam.k_diffuse;
					if (cosij < -EPSILON) //j is under i
						absorbDiffuse -= sat *cosij* dParam.k_diffuse_gravity;
					data.normalizeAbsorb[index] += absorbDiffuse * vol * dParam.restdensity;
				}		

				float dripDiffuse = 0;
				if (cosij < -EPSILON)
					dripDiffuse -= data.dripBuf[index]*cosij * dParam.k_dripBuf;
				data.normalizeDrip[index] += dripDiffuse;
			}
		}
	}
}

__global__ void waterDiffusionPredictD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	if (data.type[index] != TYPE_CLOTH) return;

	cfloat3 pos = data.pos[index];

	cint3 gridPos = calcGridPos(pos);

	float maxSaturation = dParam.max_saturation;

	data.normalizeAbsorb[index] = 0;
	data.normalizeDrip[index] = 0;
	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				waterDiffusePredictCell(neighbourPos,
					index,
					pos,
					data);
			}
		}
	}

	data.deltaAbsorb[index] = 0;
	data.deltaDrip[index] = 0;

	if (data.normalizeAbsorb[index] < data.absorbBuf[index]+EPSILON) {
		data.deltaAbsorb[index] = - data.normalizeAbsorb[index];
		data.normalizeAbsorb[index] = 1;
	}
	else {
		data.deltaAbsorb[index] = - data.absorbBuf[index];
		data.normalizeAbsorb[index] = data.absorbBuf[index] / data.normalizeAbsorb[index];
	}
		

	if (data.normalizeDrip[index] < data.dripBuf[index]+EPSILON) {
		data.deltaDrip[index] = - data.normalizeDrip[index];
		data.normalizeDrip[index] = 1;
	}
	else {
		data.deltaDrip[index] = - data.dripBuf[index];
		data.normalizeDrip[index] = data.dripBuf[index] / data.normalizeDrip[index];
	}
	
}



__device__ void waterDiffuseCell(cint3 gridPos, int index, cfloat3 pos, SimData data) {
	uint gridHash = calcGridHash(gridPos);
	    
	// get start of bucket for this cell
	uint startIndex = data.gridCellStart[gridHash];


	float vol = data.mass[index] / dParam.cloth_density;
	float poreMassLimit = vol*dParam.cloth_porosity*dParam.restdensity; //water mass
	float cd = dParam.diffuseDistance;
	float waterMass = data.absorbBuf[index] + data.dripBuf[index];
	float sat = data.absorbBuf[index] / poreMassLimit;

	if (startIndex != 0xffffffff)          // cell is not empty
	{
		uint endIndex = data.gridCellEnd[gridHash];
		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index && data.type[j] == TYPE_CLOTH)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d = xij.mode();

				if (d >= cd)
					continue;

				float volj = data.mass[j] / dParam.cloth_density * dParam.restdensity;

				float satj = data.absorbBuf[j] / (volj * dParam.cloth_porosity);
				float cosij = dot(xij, dParam.gravity) / d;
				
				//j -> i

				//printf("%f %f\n", satj, sat);
				if (sat < 1 - EPSILON) {
					float absorbDiffuseJ = 0;
					if (sat < satj - EPSILON) 
						absorbDiffuseJ += (satj - sat)  * dParam.k_diffuse;
					if (cosij > EPSILON) // j is above i
						absorbDiffuseJ += satj *cosij* dParam.k_diffuse_gravity;
					data.deltaAbsorb[index] += absorbDiffuseJ *volj * dParam.restdensity* data.normalizeAbsorb[j];
				}


				float dripDiffuseJ = 0;
				if (cosij > EPSILON)
					dripDiffuseJ += data.dripBuf[j] *cosij* dParam.k_dripBuf;
				data.deltaDrip[index] += dripDiffuseJ * data.normalizeDrip[j];
			}
		}
	}
}

__global__ void waterDiffusionD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	if (data.type[index] != TYPE_CLOTH) return;

	cfloat3 pos = data.pos[index];

	cint3 gridPos = calcGridPos(pos);

	for (int z = -1; z <= 1; z++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				cint3 neighbourPos = gridPos + cint3(x, y, z);
				waterDiffuseCell(neighbourPos,
					index,
					pos,
					data);
			}
		}
	}

	
}

__global__ void updateDiffusionD(SimData data, int numParticles) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= numParticles)
		return;
	if (data.type[index] != TYPE_CLOTH) return;

	data.absorbBuf[index] += data.deltaAbsorb[index] * dParam.dt;
	data.dripBuf[index] += data.deltaDrip[index] * dParam.dt;

	float waterMass = data.absorbBuf[index] + data.dripBuf[index];
	float vol = data.mass[index] / dParam.cloth_density;

	float poreMassLimit = vol*dParam.cloth_porosity*dParam.restdensity;
	if (data.absorbBuf[index] > poreMassLimit+EPSILON) {
		data.absorbBuf[index] = poreMassLimit;
		data.dripBuf[index] = waterMass - data.absorbBuf[index];
	}
	/*else if (data.absorbBuf[index] < poreMassLimit - EPSILON) {
		data.absorbBuf[index] = fmin(waterMass, poreMassLimit);
		data.dripBuf[index] = waterMass - data.absorbBuf[index];
	}*/


	//float sat = data.dripBuf[index] / 10;
	//if (sat>1 - EPSILON) {
	//	sat = 1 - EPSILON;
	//}

	float sat = data.absorbBuf[index] / poreMassLimit;
	if (sat>1 - EPSILON) {
		sat = 1 - EPSILON;
	}
	if(data.invMass[index]>EPSILON)
		data.color[index].Set( 0.8*(1 - sat), 0.8*(1 - sat), 0.4*(1 - sat), 1);
}

__global__ void waterEmissionD(SimData data, int numParticles) {

}

#endif
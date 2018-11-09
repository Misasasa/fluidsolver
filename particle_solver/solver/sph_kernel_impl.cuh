#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "sph_solver.cuh"
#include "cuda_common.cuh"

typedef unsigned int uint;
namespace sph
{


SimParam_SPH hParam;
__device__ SimParam_SPH dParam;



__device__ uint calcGridHash(cint3 gridPos)
{
	if(gridPos.x < 0) gridPos.x = 0;
	if(gridPos.y < 0) gridPos.y = 0;
	if(gridPos.z < 0) gridPos.z = 0;
	if (gridPos.x >= dParam.gridres.x) gridPos.x = dParam.gridres.x-1;
	if (gridPos.y >= dParam.gridres.y) gridPos.y = dParam.gridres.y-1;
	if (gridPos.z >= dParam.gridres.z) gridPos.z = dParam.gridres.z-1;
	
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

	if (x.x<-999)
		hash = GRID_UNDEF;
	else {
		cint3 gridPos = calcGridPos(x);
		hash = calcGridHash(gridPos);
	}

	ParticleHash[i] = hash;
	ParticleIndex[i] = i;
	
}



__global__ void reorderDataAndFindCellStartD(
	SimData_SPH data,
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
			if (hash!=GRID_UNDEF)
				data.gridCellStart[hash] = index;

			if (index > 0)
				data.gridCellEnd[sharedHash[threadIdx.x]] = index;
		}
		if (index == numParticles - 1)
		{
			if (hash!=GRID_UNDEF)
				data.gridCellEnd[hash] = index + 1;
		}
		

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = data.particleIndex[index];
		cfloat3 pos = data.pos[sortedIndex];       // macro does either global read or texture fetch
		cfloat3 vel = data.vel[sortedIndex];       // see particles_kernel.cuh
		
		data.sortedPos[index] = pos;
		data.sortedVel[index] = vel;
		data.sortedNormal[index] = data.normal[sortedIndex];
		data.sortedColor[index] =  data.color[sortedIndex];
		data.sortedMass[index] =   data.mass[sortedIndex];
		data.sortedType[index] =   data.type[sortedIndex];
		data.sortedGroup[index] =  data.group[sortedIndex];
		data.sortedUniqueId[index] = data.uniqueId[sortedIndex];
		data.indexTable[data.sortedUniqueId[index]] = index;
	}
}


__device__ void PressureCell(cint3 gridPos, int index, cfloat3 pos, float& density, SimData_SPH data) {
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];
		//printf("%d %d\n", startIndex, endIndex);
		
		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index )
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2)
					continue;

				float d = sqrt(d2);

				float c2 = sr2 - d2;
				if(data.type[j]==TYPE_FLUID)
					density += c2*c2*c2 * data.mass[j];
				//if (data.type[j]==TYPE_BOUNDARY)
				//	density += c2*c2*c2 * data.mass[index];
			}
		}
		
	}
}

__global__ void computeP(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index >= numP) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;

	
	for(int z=-1;z<=1;z++)
		for(int y=-1;y<=1;y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x,y,z);
				PressureCell(nPos, index, pos, density, data);
			}
	
	
	float sr2 = dParam.smoothradius * dParam.smoothradius;
	density += sr2*sr2*sr2*data.mass[index];
	density *= dParam.kpoly6;
	data.density[index] = density;
	data.pressure[index] = dParam.pressureK * (powf(density/dParam.restdensity, 7) - 1);
	//if(data.pressure[index]<0)
	//	data.pressure[index] = 0;
}


__device__ void ForceCell(cint3 gridPos, int index, cfloat3 pos, cfloat3& force, SimData_SPH data) {
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];

		for (uint j = startIndex; j < endIndex; j++)
		{
			cfloat3 pos2 = data.pos[j];
			cfloat3 xij = pos - pos2;
			float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
			float d = sqrt(d2);
			
			if (j != index && data.type[j]==TYPE_FLUID)
			{
				if (d2 >= sr2)
					continue;
				float c = sr - d;
				//pressure
				float nablaw = dParam.kspikydiff * c * c / d;
				float pc = nablaw *data.mass[j]* (data.pressure[j]/data.density[j]/data.density[j]
					+ data.pressure[index]/data.density[index]/data.density[index]);
				force += xij * pc * (-1);

				//viscosity?
				float vc = nablaw * d2 / (d2 + 0.01 * sr2)
					*data.mass[j]/data.density[j]*2* dParam.viscosity;
				force += (data.vel[index]-data.vel[j])*vc;

				//artificial visc
				/*
				cfloat3 vij = data.vel[index] - data.vel[j];
				float xv = dot(vij, xij);
				if (xv < 0) {
					float c = sr - d;
					float nablaw = dParam.kspikydiff * c * c / d;
					float visc = 2*dParam.bvisc*dParam.smoothradius * 88.5 /2000;
					float pi = -visc * xv /(d2 + 0.01*sr2);
					force += xij * pi * nablaw * data.mass[j] * (-1);
				}
				*/
			}

			if (data.type[j]==TYPE_BOUNDARY) {
				float B=1;
				//float y = abs(dot(xij, data.normal[j]));
				//float x = sqrt(d2 - y*y);
				//if(x<dParam.spacing)
				//	B = 1 - x/dParam.spacing;
				
				float q = d/sr;
				if (q<0.66666) {
					B *= 0.66666;
				}
				else if (q<1) {
					B *= 2*q - 1.5*q*q;
				}
				else if (q<2) {
					B *= 0.5 * (2-q)*(2-q);
				}
				else
					B = 0;

				B *= 0.02 * 88.5*88.5 /d;

				float magnitude = data.mass[j]/(data.mass[index]+data.mass[j]) * B;
				//force += data.normal[j]*magnitude;
				force += xij * magnitude;

				//artificial visc
				cfloat3 vij = data.vel[index] - data.vel[j];
				float xv = dot(vij, xij);
				if (xv < 0) {
					float c = sr - d;
					float nablaw = dParam.kspikydiff * c * c / d;
					float visc = 2*dParam.bvisc*dParam.smoothradius * 88.5 /2000;
					float pi = -visc * xv /(d2 + 0.01*sr2);
					force += xij * pi * nablaw * data.mass[j] * (-1);
				}
				
			}
		}
	}
}

__global__ void computeF(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0,0,0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				ForceCell(nPos, index, pos, force, data);
			}
	data.force[index] = force + cfloat3(0,-9.8,0);
}


__device__ void clampBoundary(int index, SimData_SPH data) {
	//boundary
	if (data.pos[index].y < dParam.gridxmin.y+EPSILON) {
		data.pos[index].y = dParam.gridxmin.y+EPSILON;
		data.vel[index].y = 0;
	}
	if (data.pos[index].y > dParam.gridxmax.y-EPSILON) {
		data.pos[index].y = dParam.gridxmax.y-EPSILON;
		data.vel[index].y = 0;
	}
	if (data.pos[index].x < dParam.gridxmin.x+EPSILON) {
		data.pos[index].x = dParam.gridxmin.x+EPSILON;
		data.vel[index].x = 0;
	}
	if (data.pos[index].x > dParam.gridxmax.x-EPSILON) {
		data.pos[index].x = dParam.gridxmax.x-EPSILON;
		data.vel[index].x = 0;
	}
	if (data.pos[index].z < dParam.gridxmin.z+EPSILON) {
		data.pos[index].z = dParam.gridxmin.z+EPSILON;
		data.vel[index].z = 0;
	}
	if (data.pos[index].z > dParam.gridxmax.z-EPSILON) {
		data.pos[index].z = dParam.gridxmax.z-EPSILON;
		data.vel[index].z = 0;
	}
	
}

__global__ void advectAndCollision(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]==TYPE_BOUNDARY) return;

	data.vel[index] += data.force[index] * dParam.dt;
	data.pos[index] += data.vel[index] * dParam.dt;

	clampBoundary(index, data);
	




}


};
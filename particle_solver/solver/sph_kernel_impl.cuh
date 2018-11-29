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
		//DFSPH
		data.sortedV_star[index] = data.v_star[sortedIndex];
		//Multiphase
		data.sortedRestDensity[index] = data.restDensity[sortedIndex];
		data.sortedMassFac[index] = data.massFac[sortedIndex];
		for(int t=0; t<dParam.maxtypenum; t++)
			data.sortedVFrac[index*dParam.maxtypenum+t] = data.vFrac[sortedIndex*dParam.maxtypenum+t];
	}
}


__device__ void DensityCell(cint3 gridPos, int index, cfloat3 pos, float& density, SimData_SPH data) {
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
				DensityCell(nPos, index, pos, density, data);
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










//====================================================
//
//                    DFSPH
//
//====================================================


__device__ void DensityAlphaCell(cint3 gridPos,
	int index,
	cfloat3 pos,
	float& density,
	float& mwij,
	cfloat3& mwij3,
	SimData_SPH data) 
{
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];
		//printf("%d %d\n", startIndex, endIndex);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2)
					continue;

				float d = sqrt(d2);
				float c2 = sr2 - d2;
				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d;
				
				if (data.type[j]==TYPE_FLUID){
					density += c2*c2*c2 * data.mass[j];
					cfloat3 aij = xij * nablaw * data.mass[j];
					mwij += dot(aij,aij);
					mwij3 += aij;
				}
			}
		}

	}
}

__global__ void computeDensityAlpha_kernel(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;
	float mwij = 0;
	cfloat3 mwij3(0,0,0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				DensityAlphaCell(nPos, 
					index, 
					pos, 
					density,
					mwij,
					mwij3,
					data);
			}


	float sr2 = dParam.smoothradius * dParam.smoothradius;
	density += sr2*sr2*sr2*data.mass[index];
	density *= dParam.kpoly6;
	data.density[index] = density;
	
	float denom = dot(mwij3, mwij3) + mwij;
	if(denom<0.000001)
		denom = 0.000001;//clamp for stability
	data.alpha[index] = data.density[index] / denom;

}


__device__ void computeNPFCell(cint3 gridPos,
	int index,
	cfloat3 pos,
	cfloat3& force,
	SimData_SPH data)
{
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
				//float pc = nablaw *data.mass[j]* (data.pressure[j]/data.density[j]/data.density[j]
				//	+ data.pressure[index]/data.density[index]/data.density[index]);
				//force += xij * pc * (-1);

				//viscosity
				float vc = nablaw * d2 / (d2 + 0.01 * sr2)
					*data.mass[j]/data.density[j]*2* dParam.viscosity;
				force += (data.vel[index]-data.vel[j])*vc;
			}

			if (data.type[j]==TYPE_BOUNDARY) {
				float B=1;
				
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

__global__ void computeNPF_kernel(SimData_SPH data, int numP)
{
	//viscosity, collision, gravity
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if(data.type[index]!=TYPE_FLUID) {
		data.v_star[index] = cfloat3(0,0,0);
		return;
	}

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				computeNPFCell(nPos,
					index,
					pos,
					force,
					data);
			}

	force += cfloat3(0,-9.8,0); //gravity
	//predict velocity
	data.v_star[index] = data.vel[index] + force*dParam.dt;
}

__device__ void DensityChangeCell(
	cint3 gridPos,
	int index,
	cfloat3 pos,
	float& densChange,
	SimData_SPH data
)
{
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];
		//printf("%d %d\n", startIndex, endIndex);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				cfloat3 vij = data.v_star[index] - data.v_star[j];
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2)
					continue;

				float d = sqrt(d2);
				float c2 = sr2 - d2;
				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d;

				if (data.type[j]==TYPE_FLUID) {
					
					cfloat3 nwij = xij * nablaw * data.mass[j];
					densChange += dot(vij, nwij);
				}
			}
		}

	}
}




__global__ void solveDivergenceStiff(SimData_SPH data, int numP) {
	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID){
		data.error[index] = 0;
		data.pstiff[index] = 0;
		return;
	}
	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float densChange = 0;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				DensityChangeCell(nPos,
					index,
					pos,
					densChange,
					data);
			}
	if(densChange < 0 ) densChange = 0;
	data.error[index] = densChange / data.restDensity[index];
	data.pstiff[index] = densChange/ dParam.dt * data.alpha[index];
}



__device__ void PredictDensityCell(
	cint3 gridPos,
	int index,
	cfloat3 pos,
	float& density,
	SimData_SPH data
)
{
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];
		//printf("%d %d\n", startIndex, endIndex);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				cfloat3 vij = data.v_star[index] - data.v_star[j];
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2)
					continue;

				float d = sqrt(d2);
				float c2 = sr2 - d2;
				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d;

				if (data.type[j]==TYPE_FLUID) {

					cfloat3 nwij = xij * nablaw * data.mass[j];
					density += dot(vij, nwij) * dParam.dt;
				}
			}
		}

	}
}


__global__ void solveDensityStiff(SimData_SPH data, int numP) {

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) {
		data.error[index] = 0;
		data.pstiff[index] = 0;
		return;
	}
	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = data.density[index];

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				PredictDensityCell(nPos,
					index,
					pos,
					density,
					data);
			}
	
	data.pstiff[index] = (density - data.restDensity[index])*data.alpha[index]
		/dParam.dt/dParam.dt;
	data.error[index] = density - data.restDensity[index];
	if(data.pstiff[index]<0){
		data.pstiff[index] = 0;
		data.error[index] = 0;
	}
}

__device__ void applyPressureCell(
	cint3 gridPos,
	int index,
	cfloat3 pos,
	cfloat3& force,
	SimData_SPH data
)
{
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];
		//printf("%d %d\n", startIndex, endIndex);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2)
					continue;

				float d = sqrt(d2);
				float c2 = sr2 - d2;
				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d;

				if (data.type[j]==TYPE_FLUID) {

					cfloat3 nwij = xij * nablaw * data.mass[j];
					force += nwij * (data.pstiff[index]/data.density[index]
						+ data.pstiff[j]/data.density[j]);
					//force += nwij * data.pstiff[index]/data.density[index];
				}
			}
		}

	}
}

__global__ void applyPStiff(SimData_SPH data, int numP) {
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
				applyPressureCell(nPos,
					index,
					pos,
					force,
					data);
			}
	
	data.v_star[index] += force * dParam.dt *(-1);
}

__global__ void updatePosition(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) return;

	data.pos[index] += data.v_star[index] * dParam.dt;
}


__global__ void updateVelocities(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) return;

	data.vel[index] = data.v_star[index];
}



/*****************************************

			Multiphase SPH

*****************************************/

__device__ void DFAlpha_MPH_Cell(cint3 gridPos,
	int index,
	cfloat3 pos,
	float& density,
	float& mwij,
	cfloat3& mwij3,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos);
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex != 0xffffffff) {
		uint endIndex = data.gridCellEnd[gridHash];
		//printf("%d %d\n", startIndex, endIndex);

		for (uint j = startIndex; j < endIndex; j++)
		{
			if (j != index)
			{
				cfloat3 pos2 = data.pos[j];
				cfloat3 xij = pos - pos2;
				float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

				if (d2 >= sr2)
					continue;

				float d = sqrt(d2);
				float c2 = sr2 - d2;
				float c = sr - d;
				float nablaw = dParam.kspikydiff * c * c / d;
				float densj = 0;

				if (data.type[j]==TYPE_FLUID) {
					density += c2*c2*c2 * data.mass[j];
					cfloat3 aij = xij * nablaw * data.mass[j];
					
					mwij += dot(aij, aij) / data.massFac[j]; //second term
					mwij3 += aij; //first term
				}
			}
		}

	}
}

__global__ void computeDFAlpha_MPH_kernel(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if(data.type[index]!=TYPE_FLUID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;
	float mwij = 0;
	cfloat3 mwij3(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				DFAlpha_MPH_Cell(nPos,
					index,
					pos,
					density,
					mwij,
					mwij3,
					data);
			}


	float sr2 = dParam.smoothradius * dParam.smoothradius;
	density += sr2*sr2*sr2*data.mass[index];
	density *= dParam.kpoly6;
	data.density[index] = density;

	float denom = dot(mwij3, mwij3) / data.massFac[index] + mwij;
	if (denom<0.000001)
		denom = 0.000001;//clamp for stability
	data.alpha[index] = data.density[index] / denom;

}



__device__ void NonPressureForce_MPH_Cell(cint3 gridPos,
	int index,
	cfloat3 pos,
	cfloat3& force,
	SimData_SPH data)
{
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
				float nablaw = dParam.kspikydiff * c * c / d;
				
				//pressure
				//float pc = nablaw *data.mass[j]* (data.pressure[j]/data.density[j]/data.density[j]
				//	+ data.pressure[index]/data.density[index]/data.density[index]);
				//force += xij * pc * (-1);

				//viscosity
				float vc = nablaw * d2 / (d2 + 0.01 * sr2)
					*data.mass[j]/data.density[j]*2* dParam.viscosity;
				force += (data.vel[index]-data.vel[j])*vc;

				//phase momentum diffusion
				// to be finished

			}


			//boundary collision
			if (data.type[j]==TYPE_BOUNDARY) {
				float B=1;

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

__global__ void computeNPF_MPH_kernel(SimData_SPH data, int numP) {
	//viscosity, collision, gravity,
	//phase momentum diffusion

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) {
		data.v_star[index] = cfloat3(0, 0, 0);
		return;
	}

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				NonPressureForce_MPH_Cell(nPos,
					index,
					pos,
					force,
					data);
			}

	force += cfloat3(0, -9.8, 0); //gravity
								  //predict velocity
	data.v_star[index] = data.vel[index] + force*dParam.dt;
}



__global__ void applyPStiff_MPH(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				applyPressureCell(nPos,
					index,
					pos,
					force,
					data);
			}

	data.v_star[index] += force * dParam.dt *(-1) / data.massFac[index];
}

__global__ void updateMassFac_kernel(SimData_SPH data, int numP) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= numP) return;
	if (data.type[index]!=TYPE_FLUID) return;

	float beta = 0;
	//compute mass fraction
	for (int k=0; k<dParam.maxtypenum; k++) {
		//mass fraction
		float alphak = data.vFrac[index*dParam.maxtypenum+k];
		float ck = alphak * dParam.densArr[k] / data.restDensity[index];
		if(ck<EPSILON)
			continue;
		beta += alphak * alphak / ck;
	}
	if(beta < EPSILON)
		printf("error value for beta.\n");
	data.massFac[index] = 1 / beta;

	//if(index %100==0)
	//	printf("%f\n", data.massFac[index]);
}


};
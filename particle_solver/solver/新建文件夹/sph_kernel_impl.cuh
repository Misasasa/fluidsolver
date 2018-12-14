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


/*********************************

SPH Kernel, Gradients, Laplacian

*********************************/


__inline__ __device__ cfloat3 KernelGradient_Spiky(float smooth_radius,
	cfloat3 xij) {
	float d = xij.mode();
	float c = smooth_radius - d;
	float factor = dParam.kspikydiff * c * c / d;
	return xij * factor;
}

__inline__ __device__ float Kernel_Cubic(float h, cfloat3 xij)
{
	float q = xij.mode()/h;
	float fq;
	if(q>=2) return 0;
	if(q>=1) fq = (2-q)*(2-q)*(2-q)*0.25;
	else fq = (0.666667 - q*q + 0.5*q*q*q)*1.5;
	return fq*dParam.kernel_cubic;
}

__inline__ __device__ cfloat3 KernelGradient_Cubic(float h, 
	cfloat3 xij) {
	float r = xij.mode();
	float q = r/h;
	if(q>=2 || q<EPSILON) return cfloat3(0,0,0);
	float df;
	if(q>=1)
		df = 0.5*(2-q)*(2-q)*(-1);
	else
		df = (-2*q + 1.5*q*q);
	return xij * df * dParam.kernel_cubic_gradient /r;
}



__device__ uint calcGridHash(cint3 cell_indices)
{
	/*
	if(gridPos.x < 0) gridPos.x = 0;
	if(gridPos.y < 0) gridPos.y = 0;
	if(gridPos.z < 0) gridPos.z = 0;
	if (gridPos.x >= dParam.gridres.x) gridPos.x = dParam.gridres.x-1;
	if (gridPos.y >= dParam.gridres.y) gridPos.y = dParam.gridres.y-1;
	if (gridPos.z >= dParam.gridres.z) gridPos.z = dParam.gridres.z-1;
	*/
	if(cell_indices.x<0 || cell_indices.x >= dParam.gridres.x ||
		cell_indices.y<0 || cell_indices.y >= dParam.gridres.y ||
		cell_indices.z<0 || cell_indices.z >= dParam.gridres.z)
		return GRID_UNDEF;

	return cell_indices.y * dParam.gridres.x* dParam.gridres.z 
		+ cell_indices.z*dParam.gridres.x 
		+ cell_indices.x;
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
	int num_particles) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= num_particles)	return;

	cfloat3 x = Pos[i];
	uint hash;

	if (x.x>dParam.gridxmax.x-EPSILON ||
		x.x<dParam.gridxmin.x+EPSILON ||
		x.y>dParam.gridxmax.y-EPSILON ||
		x.y<dParam.gridxmin.y+EPSILON ||
		x.z>dParam.gridxmax.z-EPSILON ||
		x.z<dParam.gridxmin.z+EPSILON) {
		hash = GRID_UNDEF;
		printf("bad particle position\n");
	}
	else {
		cint3 gridPos = calcGridPos(x);
		hash = calcGridHash(gridPos);
	}

	ParticleHash[i] = hash;
	ParticleIndex[i] = i;
	
}



__global__ void reorderDataAndFindCellStartD(
	SimData_SPH data,
	int num_particles
) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	extern __shared__ uint sharedHash[];
	uint hash;

	if (index < num_particles)
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


	if (index < num_particles)
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
		if (index == num_particles - 1)
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
		data.sorted_effective_mass[index] = data.effective_mass[sortedIndex];
		data.sorted_effective_density[index] = data.effective_density[sortedIndex];

		for(int t=0; t<dParam.maxtypenum; t++)
			data.sortedVFrac[index*dParam.maxtypenum+t] = data.vFrac[sortedIndex*dParam.maxtypenum+t];
	}
}


__device__ void DensityCell(cint3 gridPos, int index, cfloat3 pos, float& density, SimData_SPH data) {
	uint gridHash = calcGridHash(gridPos);
	if(gridHash==GRID_UNDEF) return;
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

__global__ void computeP(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if(index >= num_particles) return;

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
	if(gridHash==GRID_UNDEF) return;
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

__global__ void computeF(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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

__global__ void advectAndCollision(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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
	if(gridHash==GRID_UNDEF) return;
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

__global__ void computeDensityAlpha_kernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

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
	if(gridHash==GRID_UNDEF) return;
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

__global__ void computeNPF_kernel(SimData_SPH data, int num_particles)
{
	//viscosity, collision, gravity
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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
	if(gridHash==GRID_UNDEF) return;
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




__global__ void solveDivergenceStiff(SimData_SPH data, int num_particles) {
	
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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
	if(gridHash==GRID_UNDEF) return;
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


__global__ void solveDensityStiff(SimData_SPH data, int num_particles) {

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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
	if(gridHash==GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff)
		return;

	uint endIndex = data.gridCellEnd[gridHash];
	//printf("%d %d\n", startIndex, endIndex);

	for (uint j = startIndex; j < endIndex; j++)
	{
		if (j == index || data.type[j]!=TYPE_FLUID)	continue;
		
		cfloat3 pos2 = data.pos[j];
		cfloat3 xij = pos - pos2;
				
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2) continue;
		
		cfloat3 nwij = KernelGradient_Spiky(sr, xij) * data.mass[j];
		force += nwij * (data.pstiff[index]/data.density[index]
			+ data.pstiff[j]/data.density[j]);
	}
}

__global__ void applyPStiff(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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

__global__ void updatePosition(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	data.pos[index] += data.v_star[index] * dParam.dt;
}


__global__ void UpdateVelocities(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	cfloat3 drift_acceleration = data.v_star[index] - data.vel[index];
	drift_acceleration /= dParam.dt;
	drift_acceleration = dParam.gravity - drift_acceleration;
	//drift_acceleration = dParam.gravity;
	data.vel[index] = data.v_star[index];
	data.v_star[index] = drift_acceleration;
	//if(index %100==0)
	//	printf("%f %f %f\n", drift_acceleration.x, drift_acceleration.y, drift_acceleration.z);

}



/*****************************************

			Multiphase SPH

*****************************************/

__device__ void DFAlphaMultiphaseCell(cint3 gridPos,
	int i,
	cfloat3 xi,
	float& density,
	cfloat3& sum1,
	float& sum2,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos);
	if(gridHash==GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;
	
	uint endIndex = data.gridCellEnd[gridHash];
	
	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2) continue;

		float c2 = sr2 - d2;
		cfloat3 nablaw = KernelGradient_Cubic(sr/2, xij);
		float emass_j;

		switch (data.type[j]) {
		case TYPE_FLUID:
			emass_j = data.mass[i];
			nablaw *= emass_j;
			sum1 += nablaw;
			sum2 += dot(nablaw, nablaw) / data.effective_mass[j];
			density += Kernel_Cubic(sr/2, xij)*emass_j;
			break;
		case TYPE_RIGID:
			//emass_j = data.restDensity[j]*data.restDensity[i];
			emass_j = data.mass[i];
			nablaw *= emass_j;
			sum1 += nablaw;
			density += Kernel_Cubic(sr/2, xij)*emass_j;
			break;
		}



		if (!(sum2<1e20))
			printf("@ %f %f %f %f %f\n", sum2, nablaw.x,
				nablaw.y, nablaw.z, xij.mode());
	}
}

__global__ void DFAlphaKernel_Multiphase(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if(data.type[index]!=TYPE_FLUID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;
	float mass_i = data.mass[index];
	cfloat3 sum1(0, 0, 0);
	float sum2 = 0;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				DFAlphaMultiphaseCell(nPos,
					index,
					pos,
					density,
					sum1,
					sum2,
					data);
			}

	data.density[index] = density;

	float denom = (dot(sum1, sum1) / data.effective_mass[index] + sum2)
		* mass_i;

	if (denom<EPSILON)
		data.alpha[index] = 0;
	else {
		data.alpha[index] = data.density[index]/ denom;
		if (!(data.alpha[index]<10000000))
			printf("%f %f %f %f\n", data.alpha[index], denom,
				sum1.x, sum2);
	}
}



__device__ void NonPressureForceCell_Multiphase(cint3 gridPos,
	int index,
	cfloat3 pos,
	cfloat3& force,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos); 
	if(gridHash==GRID_UNDEF) return;
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
			if (data.type[j]==TYPE_RIGID && false) {
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

				//artificial viscosity
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

__global__ void NonPressureForceKernel_Multiphase(
	SimData_SPH data,
	int num_particles)
{
	//viscosity, collision, gravity,
	//phase momentum diffusion

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) {
		data.v_star[index] = cfloat3(0, 0, 0);
		return;
	}

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				NonPressureForceCell_Multiphase(nPos,
					index,
					pos,
					force,
					data);
			}

	force += cfloat3(0, -9.8, 0); //gravity
								  //predict velocity
	data.v_star[index] = data.vel[index] + force*dParam.dt;
}




__device__ void PredictDensityCell_Multiphase(
	cint3 gridPos,
	int i,
	cfloat3 pos,
	float& density,
	SimData_SPH data
)
{
	uint gridHash = calcGridHash(gridPos);
	if(gridHash==GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;
	uint endIndex = data.gridCellEnd[gridHash];
		
	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 pos2 = data.pos[j];
		cfloat3 xij = pos - pos2;
		cfloat3 vij = data.v_star[i] - data.v_star[j];
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2 || d2<EPSILON) continue;

		float emass_j;
		if(data.type[j]==TYPE_FLUID)
			emass_j = data.mass[i];
		else if(data.type[j]==TYPE_RIGID)
			//emass_j = data.restDensity[j]*data.restDensity[i];
			emass_j = data.mass[i];
		else 
			emass_j=0;
		
		cfloat3 nwij = KernelGradient_Cubic(sr/2, xij) * emass_j;
		density += dot(vij, nwij) * dParam.dt;
	}
}


__global__ void DensityStiff_Multiphase(SimData_SPH data, int num_particles) {

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
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
				PredictDensityCell_Multiphase(nPos,
					index,
					pos,
					density,
					data);
			}

	data.pstiff[index] = (density - data.restDensity[index])*data.alpha[index]
		/dParam.dt/dParam.dt;
	data.error[index] = density - data.restDensity[index];
	//if(data.error[index])
	//printf("%d %f %f\n",index, data.density[index], density);
	
	if (data.pstiff[index]<0) {
		data.pstiff[index] = 0;
		data.error[index] = 0;
	}
}

__device__ void DensityChangeCell_Multiphase(
	cint3 gridPos,
	int i,
	cfloat3 pos,
	float& densChange,
	SimData_SPH data
)
{
	uint gridHash = calcGridHash(gridPos);
	if(gridHash==GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;
	uint endIndex = data.gridCellEnd[gridHash];
	//printf("%d %d\n", startIndex, endIndex);

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 pos2 = data.pos[j];
		cfloat3 xij = pos - pos2;
		cfloat3 vij = data.v_star[i] - data.v_star[j];
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2 || d2<EPSILON) continue;

		float emass_j;
		if (data.type[j]==TYPE_FLUID) emass_j = data.mass[i];
		else if (data.type[j]==TYPE_RIGID) emass_j = data.restDensity[j]*data.restDensity[i];
		else emass_j=0;

		cfloat3 nwij = KernelGradient_Cubic(sr/2, xij) * emass_j;
		densChange += dot(vij, nwij);
	}
}

__global__ void DivergenceFreeStiff_Multiphase(SimData_SPH data, int num_particles) {

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) {
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
				DensityChangeCell_Multiphase(nPos,
					index,
					pos,
					densChange,
					data);
			}
	if (densChange < 0) densChange = 0;
	if (data.density[index]<data.restDensity[index]) densChange = 0;
	data.error[index] = densChange / data.restDensity[index];
	data.pstiff[index] = densChange/ dParam.dt * data.alpha[index];
}

__device__ void ApplyPressureCell_Multiphase(
	cint3 gridPos,
	int i,
	cfloat3 xi,
	float mass_i,
	SimData_SPH data,
	cfloat3& force
){
	uint gridHash = calcGridHash(gridPos); 
	if(gridHash==GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff)
		return;

	uint endIndex = data.gridCellEnd[gridHash];
	//printf("%d %d\n", startIndex, endIndex);

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;

		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2 || d2<EPSILON) continue;

		//cfloat3 nabla_w = KernelGradient_Spiky(sr, xij);
		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		
		switch (data.type[j]) {
		case TYPE_FLUID:
			force += nabla_w * (data.pstiff[i]*mass_i*mass_i/data.density[i]
				+ data.pstiff[j]*data.mass[j]*data.mass[j]/data.density[j]);
			break;
		case TYPE_RIGID:
			//float emass_j = data.restDensity[j]*data.restDensity[i];
			float emass_j = data.mass[i];
			force += nabla_w*mass_i*emass_j*data.pstiff[i]/data.density[i]*2;
			break;
		}
			
		
	}
}

__global__ void ApplyPressureKernel_Multiphase(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				ApplyPressureCell_Multiphase(
					nPos,
					index,
					pos,
					data.mass[index],
					data,
					force);
			}
	data.v_star[index] += force * dParam.dt *(-1) / data.effective_mass[index];
}

/* compute effective mass used in divergence-free solver,
the formula goes like:
\beta = \sum_k \alpha_k * \alpha_k / c_k,
m_e = m / \beta.
*/

__global__ void EffectiveMassKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	float beta = 0;
	float effective_density = 0;
	float rest_density = 0;
	float* vol_frac = &data.vFrac[index*dParam.maxtypenum];
	
	for (int k=0; k<dParam.maxtypenum; k++) {
		float mass_frac = vol_frac[k] * dParam.densArr[k] / data.restDensity[index];
		if(mass_frac < EPSILON)	continue;
		
		beta += vol_frac[k] * vol_frac[k] / mass_frac;
		rest_density += vol_frac[k] * dParam.densArr[k];
	}
	if(beta < EPSILON){
		printf("%f %f %f\n", vol_frac[0], vol_frac[1], vol_frac[2]);
		printf("error value for beta.\n");
	}

	//update mass
	data.mass[index] = rest_density * dParam.spacing*dParam.spacing*dParam.spacing;
	data.effective_mass[index] = data.mass[index] / beta;

	data.restDensity[index] = rest_density;
	data.color[index].Set(vol_frac[0], vol_frac[1], vol_frac[2], 1);
	if(vol_frac[0]<0.9 && vol_frac[1]<0.9)
		data.color[index].w = 1;
}


__device__ void VolumeFractionGradientCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cfloat3* vol_frac_gradient)
{
	uint grid_hash = calcGridHash(cell_index);
	if(grid_hash==GRID_UNDEF)
		return;
	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff) return;
	uint end_index = data.gridCellEnd[grid_hash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	
	for (uint j = start_index; j < end_index; j++)
	{
		if (j == i || data.type[j]!=TYPE_FLUID) continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2) continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		float vj = data.mass[j] / data.density[j];
		cfloat3 contribution;
		
		for (int k=0; k<dParam.maxtypenum; k++) {
			contribution = nabla_w * vj * 
				(data.vFrac[j*dParam.maxtypenum+k] - data.vFrac[i*dParam.maxtypenum+k]);
			vol_frac_gradient[k] += contribution;
		}
	}
}



__global__ void DriftVelocityKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	cfloat3* vol_frac_gradient = &data.vol_frac_gradient[index*dParam.maxtypenum];


	float turbulent_diffusion = dParam.drift_turbulent_diffusion;
	for(int k=0; k<dParam.maxtypenum; k++) vol_frac_gradient[k].Set(0,0,0); //initialization
	
	
	for (int z=-1; z<=1; z++) 
		for (int y=-1; y<=1; y++)	
			for (int x=-1; x<=1; x++) {
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				VolumeFractionGradientCell(
					neighbor_cell_index, 
					index,
					xi, 
					data, 
					vol_frac_gradient);
			}
	for (int k=0; k<dParam.maxtypenum; k++) 
		vol_frac_gradient[k] *= turbulent_diffusion;


	cfloat3 drift_acceleration = data.v_star[index];
	float dynamic_constant = dParam.drift_dynamic_diffusion; // 4*0.005/3/0.44

	float rest_density = data.restDensity[index];
	float accel_mode = drift_acceleration.mode();
	float* vol_frac = data.vFrac + index*dParam.maxtypenum;

	for (int k=0; k<dParam.maxtypenum; k++) {
		float density_k = dParam.densArr[k];
		float vol_frac_k = vol_frac[k];
		
		//if phase k doesnot exist£¬ continue
		if (vol_frac_k < EPSILON) { 
			drift_v[k].Set(0,0,0);	continue;
		}

		//dynamic term
		float density_factor = (density_k - rest_density) / rest_density;
		cfloat3 drift_vk = drift_acceleration * dynamic_constant * density_factor;
		drift_v[k] = drift_vk;

		//turbulent term
		//drift_v[k] += vol_frac_gradient[k] * turbulent_diffusion / data.vFrac[index*dParam.maxtypenum+k];
	}
}

__device__ void PredictPhaseDiffusionCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	float vol_i,
	SimData_SPH& data,
	float* phase_update) {

	uint grid_hash = calcGridHash(cell_index);
	if(grid_hash==GRID_UNDEF) return;
	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff) return;
	uint end_index = data.gridCellEnd[grid_hash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	int num_type = dParam.maxtypenum;

	for (uint j = start_index; j < end_index; j++)
	{
		if (j == i || data.type[j]!=TYPE_FLUID) continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2) continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		float vol_j = data.mass[j] / data.density[j];
		cfloat3 flux_k;

		for (int k=0; k<num_type; k++) {
			//dynamic flux
			flux_k = data.drift_v[i*num_type+k] * vol_i * data.vFrac[i*num_type+k]
				+ data.drift_v[j*num_type+k] * vol_j * data.vFrac[j*num_type+k];
			phase_update[k] += dot(flux_k, nabla_w) * (-1);

			//turbulent diffusion
			flux_k = data.vol_frac_gradient[i*num_type+k]*vol_i
				+ data.vol_frac_gradient[j*num_type+k]*vol_j;
			phase_update[k] += dot(flux_k, nabla_w) * (-1);
		}
	}

}

__global__ void PredictPhaseDiffusionKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	float vol_frac_change[10];
	float vol_i = data.mass[index] / data.density[index];

	for(int k=0; k<dParam.maxtypenum; k++) vol_frac_change[k]=0; //initialization
	for (int z=-1; z<=1; z++) for (int y=-1; y<=1; y++)	for (int x=-1; x<=1; x++) 
	{
		cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
		PredictPhaseDiffusionCell(
			neighbor_cell_index,
			index, 
			xi, 
			vol_i,
			data, 
			vol_frac_change);
	}

	//get flux multiplier: lambda_i for each particle
	
	float lambda = 1;
	for (int k=0; k<dParam.maxtypenum; k++) {
		if(vol_frac_change[k]>=0) continue;
		if (data.vFrac[index*dParam.maxtypenum+k]<EPSILON) {
			lambda=0; continue;
		}
		float lambda_k = data.vFrac[index*dParam.maxtypenum+k]/dParam.dt
			/abs(vol_frac_change[k]);
		if(lambda_k < lambda) lambda = lambda_k;
	}
	data.phase_diffusion_lambda[index] = lambda;
}


__device__ void PhaseDiffusionCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	float vol_i,
	float lambda_i,
	SimData_SPH& data,
	float* vol_frac_change) {

	uint grid_hash = calcGridHash(cell_index);
	if(grid_hash==GRID_UNDEF) return;
	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff) return;
	uint end_index = data.gridCellEnd[grid_hash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float lambda_ij, inc;

	int num_type = dParam.maxtypenum;
	for (uint j = start_index; j < end_index; j++)
	{
		if (j == i || data.type[j]!=TYPE_FLUID) continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2) continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		float vol_j = data.mass[j] / data.density[j];
		cfloat3 flux_k;
		if(lambda_i < data.phase_diffusion_lambda[j]) //pick up smaller lambda
			lambda_ij = lambda_i;
		else
			lambda_ij = data.phase_diffusion_lambda[j];

		for (int k=0; k<dParam.maxtypenum; k++) {
			flux_k = data.drift_v[i*num_type+k] * vol_i * data.vFrac[i*num_type+k]
				+ data.drift_v[j*num_type+k] * vol_j * data.vFrac[j*num_type+k];
			inc = dot(flux_k, nabla_w) * (-1) * lambda_ij;
			vol_frac_change[k] += inc;

			//turbulent diffusion
			flux_k = data.vol_frac_gradient[i*num_type+k]*vol_i
				+ data.vol_frac_gradient[j*num_type+k]*vol_j;
			inc = dot(flux_k, nabla_w) * (-1) * lambda_ij;
			vol_frac_change[k] += inc;
		}
	}
}

__global__ void PhaseDiffusionKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	
	
	if (data.type[index]!=TYPE_FLUID){
		for (int k=0; k<dParam.maxtypenum; k++)
			data.vol_frac_change[index*dParam.maxtypenum+k]=0;
		return;
	}
	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	float* vol_frac_change = &data.vol_frac_change[index*dParam.maxtypenum];
	float vol_i = data.mass[index] / data.density[index];
	float lambda_i = data.phase_diffusion_lambda[index];

	for (int k=0; k<dParam.maxtypenum; k++) vol_frac_change[k]=0; //initialization
	for (int z=-1; z<=1; z++) 
		for (int y=-1; y<=1; y++)	
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				PhaseDiffusionCell(
					neighbor_cell_index,
					index,
					xi,
					vol_i,
					lambda_i,
					data,
					vol_frac_change);
			}
	
	//float verify=0;
	//for (int k=0; k<dParam.maxtypenum; k++) verify+=vol_frac_change[k];
	//if( abs(verify) > EPSILON )
	//	printf("sum phase update: %d %f %f %f\n",  index,
	//		vol_frac_change[0], vol_frac_change[1], vol_frac_change[2]);
			
}

__global__ void UpdateVolumeFraction(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID) return;

	float* vol_frac_change = &data.vol_frac_change[index*dParam.maxtypenum];
	float* vol_frac = &data.vFrac[index*dParam.maxtypenum];
	bool debug = false;
	float normalize=0;
	for (int k=0; k<dParam.maxtypenum; k++){
		vol_frac[k] += vol_frac_change[k] * dParam.dt;
		if(data.vFrac[index*dParam.maxtypenum+k] < -0.0001)
			debug=true;
		if(vol_frac[k]<0) vol_frac[k] = 0;
		normalize += vol_frac[k];
	}
	for (int k=0; k<dParam.maxtypenum; k++)
		vol_frac[k] /= normalize;
}


__device__ void RigidParticleVolumeCell(cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	float& sum_wij) {

	uint grid_hash = calcGridHash(cell_index);
	if (grid_hash==GRID_UNDEF) return;
	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff) return;
	uint end_index = data.gridCellEnd[grid_hash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	
	for (uint j = start_index; j < end_index; j++)
	{
		if (data.type[j]!=TYPE_RIGID) continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2) continue;

		sum_wij += Kernel_Cubic(sr/2, xij);
	}

}

__global__ void RigidParticleVolumeKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_RIGID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	float sum_wij = 0;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				RigidParticleVolumeCell(
					neighbor_cell_index,
					index,
					xi,
					data,
					sum_wij);
			}
	data.restDensity[index] = 1 / sum_wij; //effective volume
}


};
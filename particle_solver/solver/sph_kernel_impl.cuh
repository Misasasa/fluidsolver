#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "sph_solver.cuh"
#include "cuda_common.cuh"
#include "custom_math.cuh"

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
	float d = xij.Norm();
	float c = smooth_radius - d;
	float factor = dParam.kspikydiff * c * c / d;
	return xij * factor;
}

__inline__ __device__ float Kernel_Cubic(float h, cfloat3 xij)
{
	float q = xij.Norm()/h;
	float fq;
	if(q>=2) return 0;
	if(q>=1) fq = (2-q)*(2-q)*(2-q)*0.25;
	else fq = (0.666667 - q*q + 0.5*q*q*q)*1.5;
	return fq*dParam.kernel_cubic;
}

__inline__ __device__ cfloat3 KernelGradient_Cubic(float h, 
	cfloat3 xij) {
	float r = xij.Norm();
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

	if (!(x.x<dParam.gridxmax.x-EPSILON &&
		x.x>dParam.gridxmin.x+EPSILON &&
		x.y<dParam.gridxmax.y-EPSILON &&
		x.y>dParam.gridxmin.y+EPSILON &&
		x.z<dParam.gridxmax.z-EPSILON &&
		x.z>dParam.gridxmin.z+EPSILON)) {
		hash = GRID_UNDEF;
		//printf("bad particle position\n");
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
		data.sorted_rho_stiff[index] = data.rho_stiff[sortedIndex];
		data.sorted_div_stiff[index] = data.div_stiff[sortedIndex];
		//Deformable Solid
		data.sorted_cauchy_stress[index] = data.cauchy_stress[sortedIndex];
		data.sorted_local_id[index] = data.local_id[sortedIndex];

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
	data.DF_factor[index] = data.density[index] / denom;

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
	data.pstiff[index] = densChange/ dParam.dt * data.DF_factor[index];
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
	
	data.pstiff[index] = (density - data.restDensity[index])*data.DF_factor[index]
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
	if (data.type[index]==TYPE_RIGID) return;

	data.pos[index] += data.v_star[index] * dParam.dt;
}


__global__ void UpdateVelocities(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]==TYPE_RIGID) 
		return;

	cfloat3 drift_acceleration = data.v_star[index] - data.vel[index];
	
	drift_acceleration /= dParam.dt;
	float magnitude = drift_acceleration.Norm();
	if(magnitude > dParam.acceleration_limit)
		drift_acceleration = drift_acceleration / magnitude * dParam.acceleration_limit;
	drift_acceleration = dParam.gravity - drift_acceleration;
	
	data.vel[index] = data.v_star[index];
	data.v_star[index] = drift_acceleration;
	

	//if(index %100==0)
	//	printf("%f %f %f\n", drift_acceleration.x, drift_acceleration.y, drift_acceleration.z);

}



/*****************************************

			Multiphase SPH

*****************************************/

__device__ void DFSPHFactorCell_Multiphase(cint3 gridPos,
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
		float mass_j;

		switch (data.type[j]) {
		case TYPE_FLUID:
			mass_j = data.mass[i];
			nablaw *= mass_j;
			sum1 += nablaw;
			sum2 += dot(nablaw, nablaw) / data.effective_mass[j];
			density += Kernel_Cubic(sr/2, xij)*mass_j;
			break;

		case TYPE_DEFORMABLE:
			mass_j = data.mass[i];
			nablaw *= mass_j;
			sum1 += nablaw;
			sum2 += dot(nablaw, nablaw) / data.effective_mass[j];
			density += Kernel_Cubic(sr/2, xij)*mass_j;
			break;

		case TYPE_RIGID:
			mass_j = data.restDensity[j]*data.restDensity[i];
			nablaw *= mass_j;
			sum1 += nablaw;
			//sum2 += dot(nablaw, nablaw) / mass_j;
			density += Kernel_Cubic(sr/2, xij)*mass_j;
			break;

		}
	}
}

__global__ void DFSPHFactorKernel_Multiphase(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if(data.type[index]==TYPE_RIGID) 
		return;

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
				DFSPHFactorCell_Multiphase(nPos,
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
		denom = 0.000001;
	
	data.DF_factor[index] = data.density[index]/ denom;
}



__device__ void NonPressureForceCell_Multiphase(cint3 gridPos,
	int i,
	cfloat3 xi,
	SimData_SPH data,
	cfloat3& force,
	cfloat3& debug)
{
	uint gridHash = calcGridHash(gridPos); 
	if(gridHash==GRID_UNDEF) 
		return;

	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	uint endIndex = data.gridCellEnd[gridHash];
	float h = dParam.smoothradius*0.5;
	float h2 = h*h;
	
	cfloat3 u1u_i(0,0,0), u2u_i(0,0,0), u3u_i(0,0,0);
	cfloat3 u1u_j, u2u_j, u3u_j;
	for (int k=0; k<dParam.maxtypenum; k++)
	{
		u1u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].x * data.vFrac[i*dParam.maxtypenum+k];
		u2u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].y * data.vFrac[i*dParam.maxtypenum+k];
		u3u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].z * data.vFrac[i*dParam.maxtypenum+k];
	}


	//surface tension
	float support_radius = h*2;
	float fac = 32.0f / 3.141593 / pow(support_radius, 9);
	float sf_kernel;

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= h2*4 || d2<EPSILON)
			continue;

		float d = sqrt(d2);
		cfloat3 nablaw = KernelGradient_Cubic(h, xij);
		cfloat3 vij =  data.vel[i]-data.vel[j];
		float volj =  data.mass[j]/data.density[j];

		if (data.type[i]==TYPE_FLUID && data.type[j]==TYPE_FLUID)
		{
			//float nablaw = dParam.kspikydiff * c * c / d;
				
			//pressure
			//float pc = nablaw *data.mass[j]* (data.pressure[j]/data.density[j]/data.density[j]
			//	+ data.pressure[index]/data.density[index]/data.density[index]);
			//force += xij * pc * (-1);

			//viscosity [becker 07]

			/*float vc = dot(nablaw,xij) / (d2 + 0.01 * h2)
				*data.mass[j]/data.density[j]*2* dParam.viscosity;
			force += (data.vel[i]-data.vel[j])*vc;*/

			//phase momentum diffusion
			cfloat3 nablaw = KernelGradient_Cubic(h, xij) * volj;
			cfloat3 tmp(0,0,0);
			u1u_j.Set(0,0,0);
			u2u_j.Set(0,0,0);
			u3u_j.Set(0,0,0);
			cfloat3 um_jk;
			for (int k=0; k<dParam.maxtypenum; k++) {
				float alphajk = data.vFrac[j*dParam.maxtypenum+k];
				um_jk = data.drift_v[j*dParam.maxtypenum+k];

				//if(dot(um_jk,um_jk)>100)
				//	printf("super drift %f %f %f %d\n",  um_jk.x, um_jk.y, um_jk.z, data.uniqueId[j] );

				u1u_j += data.drift_v[j*dParam.maxtypenum+k] * data.drift_v[j*dParam.maxtypenum+k].x * alphajk;
				u2u_j += data.drift_v[j*dParam.maxtypenum+k] * data.drift_v[j*dParam.maxtypenum+k].y * alphajk;
				u3u_j += data.drift_v[j*dParam.maxtypenum+k] * data.drift_v[j*dParam.maxtypenum+k].z * alphajk;
			}
			u1u_j += u1u_i;
			u2u_j += u2u_i;
			u3u_j += u3u_i; 

			tmp.x += dot(nablaw, u1u_j);
			tmp.y += dot(nablaw, u2u_j);
			tmp.z += dot(nablaw, u3u_j);
			force -= tmp;
			
			

			//xsph artificial viscosity [Schechter 13]

			force += vij * dParam.viscosity * volj *(-1) *Kernel_Cubic(h, xij) / dParam.dt;

			//surface tension
			if (data.group[i]==data.group[j])
			{
				
				if(d < h)
					sf_kernel = 2*pow((support_radius-d)*d,3) - pow(support_radius,6)/64.0;
				else if (d<support_radius)
					sf_kernel = pow((support_radius-d)*d, 3);
				else
					sf_kernel = 0;
				sf_kernel *= fac;

				cfloat3 sf_tension = xij * dParam.surface_tension * volj * sf_kernel / d *(-1);

				float kij = data.restDensity[i]/data.density[i] + data.restDensity[j]/data.density[j];
				force += sf_tension * kij;
			}
		}

		if (data.type[i]==TYPE_DEFORMABLE && data.type[j]==TYPE_DEFORMABLE) {
			
			force += vij * dParam.viscosity * volj *(-1) *Kernel_Cubic(h, xij) / dParam.dt;
		
		}

		if (data.type[j]==TYPE_RIGID) {
			/*float B=1, q = d/sr;
			if (q<0.66666) 
				B *= 0.66666;
			else if (q<1)
				B *= 2*q - 1.5*q*q;
			else if (q<2)
				B *= 0.5 * (2-q)*(2-q);
			else
				B = 0;
			B *= 0.02 * 88.5*88.5 /d;
			force += xij * data.mass[j]/(data.mass[index]+data.mass[j]) * B;
			*/

			//artificial viscosity
			float xv = dot(vij, xij);
			if (xv < 0) {
				float visc = dParam.bvisc*dParam.smoothradius * 88.5 / data.density[i] * 0.25;
				float pi = visc * xv /(d2 + 0.01*h2);
				cfloat3 f = nablaw * pi * data.restDensity[i]*data.restDensity[j];
				//cfloat3 n = data.normal[j];
				//cfloat3 fn = n *  dot(fn,n);
				//cfloat3 ft = f - fn;
				//force += fn;
				//force += ft;
				force += f;
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
	if (index >= num_particles) 
		return;
	if (data.type[index]==TYPE_RIGID) {
		data.v_star[index] = cfloat3(0, 0, 0);
		return;
	}

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);
	cfloat3 debug(0,0,0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				NonPressureForceCell_Multiphase(nPos,
					index,
					pos,
					data,
					force,
					debug);
			}
	force += debug;

	force += dParam.gravity; //gravity
	
	data.v_star[index] = data.vel[index] + force * dParam.dt;
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

		float emass_j = 0;

		switch (data.type[j])
		{
		case TYPE_FLUID:
		case TYPE_DEFORMABLE:
			emass_j = data.mass[i];
			break;
		case TYPE_RIGID:
			emass_j = data.restDensity[j]*data.restDensity[i];
			break;
		}
		
		cfloat3 nwij = KernelGradient_Cubic(sr/2, xij) * emass_j;
		density += dot(vij, nwij) * dParam.dt;
	}
}


__global__ void DensityStiff_Multiphase(
	SimData_SPH data, 
	int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) 
		return;

	if (data.type[index]==TYPE_RIGID) {
		data.error[index] = 0;
		data.pstiff[index] = 0;
		return;
	}

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float densityAdv = data.density[index];

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				PredictDensityCell_Multiphase(nPos,
					index,
					pos,
					densityAdv,
					data);
			}

	if(densityAdv < data.restDensity[index])
		densityAdv = data.restDensity[index];

	data.pstiff[index] = (densityAdv - data.restDensity[index])*data.DF_factor[index]
		/dParam.dt/dParam.dt;
	data.error[index] = densityAdv - data.restDensity[index];
	
	data.rho_stiff[index] += data.pstiff[index];
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

		float mass_j = 0;
		if (data.type[j]==TYPE_FLUID || data.type[j]==TYPE_DEFORMABLE)
			mass_j = data.mass[i];
		else if (data.type[j]==TYPE_RIGID)
			mass_j = data.restDensity[j]*data.restDensity[i];
		

		cfloat3 nwij = KernelGradient_Cubic(sr/2, xij) * mass_j;
		densChange += dot(vij, nwij);
	}
}

__global__ void DivergenceFreeStiff_Multiphase(
	SimData_SPH data, 
	int num_particles) 
{

	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) 
		return;
	if (data.type[index]==TYPE_RIGID) {
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
	if (densChange < 0) 
		densChange = 0;
	
	if (data.density[index]<data.restDensity[index]) 
		densChange = 0;
	
	data.error[index] = densChange / data.restDensity[index];
	data.pstiff[index] = densChange/ dParam.dt * data.DF_factor[index];

	data.div_stiff[index] += data.pstiff[index];

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
	if(gridHash==GRID_UNDEF) 
		return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff)
		return;

	uint endIndex = data.gridCellEnd[gridHash];
	
	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;

		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2<EPSILON) continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		
		switch (data.type[j]) {
		case TYPE_FLUID:
		case TYPE_DEFORMABLE:
			
			force += nabla_w * (data.pstiff[i]*mass_i*mass_i/data.density[i]
				+ data.pstiff[j]*data.mass[j]*data.mass[j]/data.density[j]);

			break;
		case TYPE_RIGID:
			float mass_j = data.restDensity[j]*data.restDensity[i];
			
			//force += nabla_w*data.pstiff[i]*mass_i*(mass_i+mass_j)/data.density[i];
			force += nabla_w*data.pstiff[i]*mass_i*mass_j/data.density[i];
			
			break;
		}
			
		
	}
}

__global__ void ApplyPressureKernel_Multiphase(
	SimData_SPH data, 
	int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

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



__device__ void ApplyPressureStiffCell(
	cint3 cell_index,
	int i,
	float* stiff,
	SimData_SPH& data,
	cfloat3& force
)
{
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF)
		return;
	
	uint startIndex = data.gridCellStart[gridHash];
	
	if (startIndex == 0xffffffff)
		return;
	uint endIndex = data.gridCellEnd[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float mass_i = data.mass[i];
	cfloat3 xi = data.pos[i];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;

		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2<EPSILON) 
			continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr*0.5, xij);

		switch (data.type[j]) {
		case TYPE_FLUID:
			force += nabla_w * (stiff[i]*mass_i*mass_i/data.density[i]
				+ stiff[j]*data.mass[j]*data.mass[j]/data.density[j]);
			break;
		case TYPE_DEFORMABLE:
			force += nabla_w * (stiff[i]*mass_i*mass_i/data.density[i]
				+ stiff[j]*data.mass[j]*data.mass[j]/data.density[j]);
			break;
		case TYPE_RIGID:
			float mass_j = data.restDensity[j]*data.restDensity[i];
			force += nabla_w*stiff[i]*mass_i*(mass_i+mass_j)/data.density[i];
			//force += nabla_w*data.pstiff[i]*mass_i*mass_j/data.density[i];
			break;
		}


	}
}


__global__ void EnforceDensityWarmStart(
	SimData_SPH data,
	int num_particles
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				ApplyPressureStiffCell(
					nPos,
					index,
					data.rho_stiff,
					data,
					force);
			}
	data.v_star[index] += force * dParam.dt *(-1) / data.effective_mass[index];
}

__global__ void EnforceDivergenceWarmStart(
	SimData_SPH data,
	int num_particles
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				ApplyPressureStiffCell(
					nPos,
					index,
					data.div_stiff,
					data,
					force);
			}
	data.v_star[index] += force * dParam.dt *(-1) / data.effective_mass[index] * 0.5;
}














//==========================================
//
//          Volume Fraction Part
//
//==========================================




/* compute effective mass used in divergence-free solver,
the formula goes like:
\beta = \sum_k \alpha_k * \alpha_k / c_k,
m_e = m / \beta.
*/

__global__ void EffectiveMassKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]==TYPE_RIGID)
		return;

	float beta = 0;
	float effective_density = 0;
	float rest_density = 0;
	float* vol_frac = &data.vFrac[index*dParam.maxtypenum];
	
	for (int k=0; k<dParam.maxtypenum; k++) {
		float mass_frac = vol_frac[k] * dParam.densArr[k] / data.restDensity[index];
		if(mass_frac < EPSILON)	
			continue;
		
		beta += vol_frac[k] * vol_frac[k] / mass_frac;
		rest_density += vol_frac[k] * dParam.densArr[k];
	}

	if(beta < EPSILON)
	{
		printf("%f %f %f\n", vol_frac[0], vol_frac[1], vol_frac[2]);
		printf("error value for beta.\n");
	}

	//update mass
	data.mass[index] = rest_density * dParam.spacing*dParam.spacing*dParam.spacing;
	data.effective_mass[index] = data.mass[index] ;/// beta;
	
	data.restDensity[index] = rest_density;
	data.color[index].Set(vol_frac[0], vol_frac[1], vol_frac[2], 1);
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
		if (j == i || data.type[j]==TYPE_RIGID) 
			continue;

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
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	cfloat3* vol_frac_gradient = &data.vol_frac_gradient[index*dParam.maxtypenum];


	//float turbulent_diffusion = dParam.drift_turbulent_diffusion;
	float turbulent_diffusion = data.vel[index].square() * dParam.drift_turbulent_diffusion + dParam.drift_thermal_diffusion;
	for(int k=0; k<dParam.maxtypenum; k++) 
		vol_frac_gradient[k].Set(0,0,0); //initialization
	
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
	float accel_mode = drift_acceleration.Norm();
	float* vol_frac = data.vFrac + index*dParam.maxtypenum;

	for (int k=0; k<dParam.maxtypenum; k++) {
		float density_k = dParam.densArr[k];
		float vol_frac_k = vol_frac[k];
		
		//if phase k doesnot exist�� continue
		if (vol_frac_k < EPSILON) { 
			drift_v[k].Set(0,0,0);	continue;
		}

		//dynamic term
		float density_factor = (density_k - rest_density) / rest_density;
		cfloat3 drift_vk = drift_acceleration * dynamic_constant * density_factor;
		drift_v[k] = drift_vk;

		if (dot(drift_v[k], drift_v[k])>100) {
			printf("super drift? %f %f %f from %f %f %f\n", drift_v[k].x, drift_v[k].y, drift_v[k].z, drift_acceleration.x, drift_acceleration.y, drift_acceleration.z);
		}
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
		if (j == i || data.type[j]==TYPE_RIGID) 
			continue;

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
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	float vol_frac_change[10];
	float vol_i = data.mass[index] / data.density[index];

	for(int k=0; k<dParam.maxtypenum; k++) 
		vol_frac_change[k]=0; //initialization

	for (int z=-1; z<=1; z++) 
	for (int y=-1; y<=1; y++)	
	for (int x=-1; x<=1; x++) 
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
	if(grid_hash==GRID_UNDEF) 
		return;
	
	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff)
		return;
	uint end_index = data.gridCellEnd[grid_hash];
	
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	int num_type = dParam.maxtypenum;
	float lambda_ij, inc;
	
	for (uint j = start_index; j < end_index; j++)
	{
		if (j == i || data.type[j]==TYPE_RIGID) continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2)
			continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr*0.5, xij);
		float vol_j = data.mass[j] / data.density[j];
		cfloat3 flux_k;

		lambda_ij = fmin(data.phase_diffusion_lambda[i], data.phase_diffusion_lambda[j]);

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
	
	if (data.type[index]==TYPE_RIGID){
		for (int k=0; k<dParam.maxtypenum; k++)
			data.vol_frac_change[index*dParam.maxtypenum+k]=0;
		return;
	}

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	float vol_i = data.mass[index] / data.density[index];
	float lambda_i = data.phase_diffusion_lambda[index];
	
	float* vol_frac_change = &data.vol_frac_change[index*dParam.maxtypenum];

	for (int k=0; k<dParam.maxtypenum; k++) 
		vol_frac_change[k]=0; //initialization

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
}

__global__ void UpdateVolumeFraction(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	float* vol_frac_change = &data.vol_frac_change[index*dParam.maxtypenum];
	float* vol_frac = &data.vFrac[index*dParam.maxtypenum];
	bool debug = false;
	float normalize=0;

	for (int k=0; k<dParam.maxtypenum; k++){
		vol_frac[k] += vol_frac_change[k] * dParam.dt;
		//if(!(data.vFrac[index*dParam.maxtypenum+k] > -EPSILON))
		//	debug=true;
		
		if( !(vol_frac[k]>0)) vol_frac[k] = 0;
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

__global__ void MoveConstraintBoxKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_RIGID || data.group[index]!=1) return;

	data.pos[index] = cfloat3(0.39,0.39,0.39);
}

__device__ void DetectDispersedCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	float& vol_frac,
	float& vol_sum,
	int& neighbor_count
)
{
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF) 
		return;
	
	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff) 
		return;

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	uint endIndex = data.gridCellEnd[gridHash];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xij = xi - data.pos[j];;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2) 
			continue;

		if (data.type[j]!=TYPE_RIGID && j!=i)
		{
			neighbor_count ++;
			float contrib = 1;
			vol_sum += contrib;

			if(data.group[i]==data.group[j]) 
				vol_frac += contrib;
		}
	}
}

__global__ void DetectDispersedParticlesKernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) 
		return;
	if (data.type[index]==TYPE_RIGID) 
		return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	float vol_frac = 0;
	float vol_sum = 0;
	int neighbor_count=0;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				DetectDispersedCell(
					neighbor_cell_index,
					index,
					xi,
					data,
					vol_frac,
					vol_sum,
					neighbor_count);
			}
	printf("neighbor count: %f\n", vol_sum);
	vol_frac /= vol_sum;
	data.spatial_status[index] = vol_frac;
}


__device__ void ComputeTension_Cell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cfloat3& tension
){
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF)
		return;

	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	uint endIndex = data.gridCellEnd[gridHash];
	float voli = data.mass[i] / data.density[i];
	cfloat3 nablaw_ij;
	cfloat3 tensionij;
	
	cmat3& sigma_i = data.cauchy_stress[i];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xij = xi - data.pos[j];;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2 < EPSILON || data.type[j]!=TYPE_DEFORMABLE)
			continue;

		cmat3& sigma_j = data.cauchy_stress[j];
		nablaw_ij = KernelGradient_Cubic(sr*0.5, xij);
		tensionij.x = nablaw_ij.x * sigma_j.data[0]
			+ nablaw_ij.y * sigma_j.data[3]
			+ nablaw_ij.z * sigma_j.data[6];
		tensionij.y = nablaw_ij.x * sigma_j.data[1]
			+ nablaw_ij.y * sigma_j.data[4]
			+ nablaw_ij.z * sigma_j.data[7];
		tensionij.z = nablaw_ij.x * sigma_j.data[2]
			+ nablaw_ij.y * sigma_j.data[5]
			+ nablaw_ij.z * sigma_j.data[8];
		tension += tensionij * data.mass[j] / data.density[j];

		tensionij.x = nablaw_ij.x * sigma_i.data[0]
			+ nablaw_ij.y * sigma_i.data[3]
			+ nablaw_ij.z * sigma_i.data[6];
		tensionij.y = nablaw_ij.x * sigma_i.data[1]
			+ nablaw_ij.y * sigma_i.data[4]
			+ nablaw_ij.z * sigma_i.data[7];
		tensionij.z = nablaw_ij.x * sigma_i.data[2]
			+ nablaw_ij.y * sigma_i.data[5]
			+ nablaw_ij.z * sigma_i.data[8];
		tension += tensionij * voli;

	}
}

__global__ void ComputeTension_Kernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3  tension(0,0,0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				ComputeTension_Cell(
					neighbor_cell_index,
					index,
					xi,
					data,
					tension);
			}
	data.v_star[index] += tension * dParam.dt / data.density[index];
}




__device__ void VelocityGradient_Cell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cmat3& nabla_v
) {
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF)
		return;

	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	uint endIndex = data.gridCellEnd[gridHash];
	float voli = data.mass[i] / data.density[i];
	cfloat3 nablaw_ij;
	cfloat3 tensionij;

	cmat3& sigma_i = data.cauchy_stress[i];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xij = xi - data.pos[j];;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2 < EPSILON || data.type[j]!=TYPE_DEFORMABLE)
			continue;

		cfloat3 vj = data.v_star[j] * data.mass[j] / data.density[j]
			- data.v_star[i] * voli;
		cfloat3 nablaw = KernelGradient_Cubic(sr*0.5, xij);

		nabla_v[0][0] += vj.x * nablaw.x;	nabla_v[0][1] += vj.x * nablaw.y;	nabla_v[0][2] += vj.x * nablaw.z;
		nabla_v[1][0] += vj.y * nablaw.x;	nabla_v[1][1] += vj.y * nablaw.y;	nabla_v[1][2] += vj.y * nablaw.z;
		nabla_v[2][0] += vj.z * nablaw.x;	nabla_v[2][1] += vj.z * nablaw.y;	nabla_v[2][2] += vj.z * nablaw.z;
	}
}


/*
Update Cauchy stress directly with SPH velocity gradient.
This approach is unstable, and requires large artificial
viscosity.
*/
__device__ void UpdateStressWithVGrad(cmat3& nabla_v, cmat3& stress, float dt)
{
	cmat3 nabla_vT;
	mat3transpose(nabla_v, nabla_vT);
	cmat3 strain_rate;
	mat3add(nabla_v, nabla_vT, strain_rate);
	cmat3 rotation_rate;
	mat3sub(nabla_v, nabla_vT, rotation_rate);

	float p = (strain_rate.data[0] + strain_rate.data[4] + strain_rate.data[8])/3;
	strain_rate.data[0] -= p;
	strain_rate.data[4] -= p;
	strain_rate.data[8] -= p;

	for (int k=0; k<9; k++) {
		strain_rate.data[k] *= dParam.solidG;
		rotation_rate.data[k] *= 0.5;
	}

	
	p *= 3*0.5*dParam.solidK;
	strain_rate.data[0] += p;
	strain_rate.data[4] += p;
	strain_rate.data[8] += p;

	cmat3 stress_rate = strain_rate;
	mat3prod(rotation_rate, stress, nabla_vT);
	mat3add(stress_rate, nabla_vT, stress_rate);
	mat3prod(stress, rotation_rate, nabla_vT);
	mat3sub(stress_rate, nabla_vT, stress_rate);

	for (int k=0; k<9; k++) {
		stress.data[k] += stress_rate.data[k] * dt;
	}
}

__global__ void UpdateSolidStateVGrad_Kernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cmat3  nabla_v;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				VelocityGradient_Cell(
					neighbor_cell_index,
					index,
					xi,
					data,
					nabla_v);
			}
	
	UpdateStressWithVGrad(nabla_v, data.cauchy_stress[index], dParam.dt);
}





/*
Compute deformation gradient F.
Extract the rotation part R.
Compute the strain \epsilon and 
the first Piola-Kirchhoff stress P accordingly.
*/





__device__ void DeformationGradient_Cell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cmat3& F
) {
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF)
		return;

	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	uint endIndex = data.gridCellEnd[gridHash];
	
	float volj = dParam.spacing*dParam.spacing*dParam.spacing;
	cfloat3 nablaw_ij;
	int localid_i = data.local_id[i];
	cmat3& L = data.correct_kernel[localid_i];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xij = xi - data.pos[j];;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2 < EPSILON || data.type[j]!=TYPE_DEFORMABLE)
			continue;

		cfloat3 xji = xij * (-1) * volj;
		cfloat3 nablaw = KernelGradient_Cubic(sr*0.5, xij);
		mvprod(L, nablaw, nablaw);
		F.Add(TensorProduct(xji, nablaw));
	}
}

__device__ void ExtractRotation(
	cmat3& A, cmat3& R, int maxIter
)
{
	cmat3 res, expw;
	float w;
	for (uint iter=0; iter<maxIter; iter++) {
		cfloat3 omega = cross(R.Col(0), A.Col(0))
			+ cross(R.Col(1), A.Col(1)) + cross(R.Col(2), A.Col(2));
		float denom = fabs(dot(R.Col(0), A.Col(0))
			+ dot(R.Col(1), A.Col(1)) + dot(R.Col(2), A.Col(2))) + 1e-9;
		omega *= 1.0/denom;
		w = omega.Norm();
		if (w < 1e-9)
			break;

		AxisAngle2Matrix(omega, w, expw);
		mat3prod(expw, R, res);
		R = res;
	}
}

__device__ void RotatedDeformationGradient_Cell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cmat3& rotated_F
) {
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF)
		return;

	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	uint endIndex = data.gridCellEnd[gridHash];

	float volj = dParam.spacing*dParam.spacing*dParam.spacing;
	cfloat3 nablaw_ij;
	int localid_i = data.local_id[i];
	cmat3& L = data.correct_kernel[localid_i];
	cmat3& R = data.rotation[localid_i];
	cmat3 RL;
	mat3prod(R,L,RL);

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xij = xi - data.pos[j];;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2 < EPSILON || data.type[j]!=TYPE_DEFORMABLE)
			continue;

		cfloat3 xji = xij * (-1) * volj;
		cfloat3 nablaw = KernelGradient_Cubic(sr*0.5, xij);
		mvprod(RL, nablaw, nablaw);
		rotated_F.Add(TensorProduct(xji, nablaw));
	}
}



__global__ void UpdateSolidStateF_Kernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cmat3 F;
	cmat3& R = data.rotation[data.local_id[index]];

	for (int z=-1; z<=1; z++)
	for (int y=-1; y<=1; y++)
	for (int x=-1; x<=1; x++)
	{
		cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
		DeformationGradient_Cell(
			neighbor_cell_index,
			index,
			xi,
			data,
			F);
	}
	
	//Extract rotation R from F.
	ExtractRotation(F, R, 10);

	//Rotated Deformation Gradient F_rotated.
	//Stored in F.
	for (int z=-1; z<=1; z++)
	for (int y=-1; y<=1; y++)
	for (int x=-1; x<=1; x++)
	{
		cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
		RotatedDeformationGradient_Cell(
			neighbor_cell_index,
			index,
			xi,
			data,
			F);
	}

	//Compute strain epsilon.
	cmat3 F_T;
	mat3transpose(F, F_T);
	cmat3 epsilon;
	epsilon = (F+F_T) * 0.5;
	epsilon[0][0] -= 1;
	epsilon[1][1] -= 1;
	epsilon[2][2] -= 1;

}



__global__ void PlasticProjection_Kernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	cmat3& stress = data.cauchy_stress[index];
	float mod = 0;
	for(int k=0;k<9;k++) mod += stress.data[k]*stress.data[k];
	if (mod > dParam.Yield)
	{
		float fac = sqrt(dParam.Yield/mod);
		for(int k=0;k<9;k++)
			stress.data[k] *= fac;
	}
}


__device__ void FindNeighborsCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	int& neighborcount,
	cmat3& kernel_tmp
)
{
	uint gridHash = calcGridHash(cell_index);
	if (gridHash==GRID_UNDEF)
		return;

	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	uint endIndex = data.gridCellEnd[gridHash];
	float volj = dParam.spacing*dParam.spacing*dParam.spacing;
	int localid_i = data.local_id[i];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xij = xi - data.pos[j];;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2 < EPSILON || data.type[j]!=TYPE_DEFORMABLE)
			continue;
		

		// build up neighbor list
		if (neighborcount < NUM_NEIGHBOR) {
			data.neighborlist[localid_i*NUM_NEIGHBOR + neighborcount] = data.uniqueId[j];
			neighborcount ++;
		}
		else {
			printf("too many neighbors error.\n");
		}
		
		// kernel correction matrix
		cfloat3 nablaw = KernelGradient_Cubic(sr*0.5, xij);
		xij *= (-1) * volj;
		kernel_tmp.Add(TensorProduct(nablaw, xij));
	}
}

__global__ void InitializeDeformable_Kernel(SimData_SPH data,
	int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	int localid_i = data.local_id[index];
	int neighborcount = 1;
	cmat3 kernel_tmp;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				FindNeighborsCell(
					neighbor_cell_index,
					index,
					xi,
					data,
					neighborcount,
					kernel_tmp);
			}
	data.neighborlist[localid_i*NUM_NEIGHBOR] = neighborcount - 1;
	
	//printf("%d %d\n", index, neighborcount - 1 );
	/*printf("%f %f %f\n%f %f %f\n%f %f %f\n", 
		kernel_tmp.data[0], kernel_tmp.data[1], kernel_tmp.data[2],
		kernel_tmp.data[3], kernel_tmp.data[4], kernel_tmp.data[5], 
		kernel_tmp.data[6], kernel_tmp.data[7], kernel_tmp.data[8]);*/

	data.correct_kernel[localid_i] = kernel_tmp.Inv();
	data.x0[localid_i] = xi;
	
	//Set rotation matrix R to identitiy matrix.
	cmat3& R = data.rotation[localid_i];
	R.Set(0.0);
	R[0][0] = R[1][1] = R[2][2] = 1;
}


};
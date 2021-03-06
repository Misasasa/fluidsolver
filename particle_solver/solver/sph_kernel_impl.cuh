﻿#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "sph_solver.cuh"
#include "cuda_common.cuh"
#include "custom_math.cuh"

typedef unsigned int uint;


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
	if(!(q<2 && q>EPSILON))
		return cfloat3(0,0,0);
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

	uint ret = cell_indices.y * dParam.gridres.x* dParam.gridres.z
		+ cell_indices.z*dParam.gridres.x
		+ cell_indices.x;

	if (ret > dParam.gridres.x * dParam.gridres.y * dParam.gridres.z)
		return GRID_UNDEF;
	return ret;
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
		data.sorted_temperature[index] = data.temperature[sortedIndex];
		data.sorted_heat_buffer[index] = data.heat_buffer[sortedIndex];
		//Deformable Solid
		data.sorted_cauchy_stress[index] = data.cauchy_stress[sortedIndex];
		data.sorted_gradient[index] = data.gradient[sortedIndex];
		data.sorted_local_id[index] = data.local_id[sortedIndex];
		data.sorted_adjacent_index[index] = data.adjacent_index[sortedIndex];
		//IISPH
		data.sorted_pressure[index] = data.pressure[sortedIndex];

		for(int t=0; t<dParam.maxtypenum; t++)
			data.sortedVFrac[index*dParam.maxtypenum+t] = data.vFrac[sortedIndex*dParam.maxtypenum+t];
	}
}


__device__ void DensityCell(
	cint3 gridPos, 
	int index, 
	cfloat3 pos, 
	float& density, 
	SimData_SPH data) 
{
	uint gridHash = calcGridHash(gridPos);
	if(gridHash==GRID_UNDEF) 
		return;
	
	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;
	uint endIndex = data.gridCellEnd[gridHash];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	
	for (uint j = startIndex; j < endIndex; j++)
	{
		
		cfloat3 pos2 = data.pos[j];
		cfloat3 xij = pos - pos2;
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2)
			continue;

		float d = sqrt(d2);
		float c2 = sr2 - d2;
		float wij = Kernel_Cubic(sr*0.5, xij);
		if(data.type[j]==TYPE_FLUID)
			density += data.mass[index] * wij;

		if (data.type[j]==TYPE_RIGID)
			density += data.restDensity[index]*data.restDensity[j] * wij;
		
	}
		
	
}

__global__ void ComputePressureKernel(SimData_SPH data, int num_particles) {
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
	
	data.density[index] = density;
	data.pressure[index] = dParam.pressureK * (powf(density/data.restDensity[index], 7) - 1);
	if(data.pressure[index]<0)
		data.pressure[index] = 0;
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
					float visc = 2*dParam.boundary_visc*dParam.smoothradius * 88.5 /2000;
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



__global__ void AdvectKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	data.vel[index] += data.force[index] * dParam.dt;
	data.pos[index] += data.vel[index] * dParam.dt;
	
	cfloat3 drift_acceleration = dParam.gravity - data.force[index];
	float magnitude = drift_acceleration.Norm();
	if (magnitude > dParam.acceleration_limit)
		drift_acceleration = drift_acceleration / magnitude * dParam.acceleration_limit;
	drift_acceleration = dParam.gravity - drift_acceleration;
	data.v_star[index] = drift_acceleration;
}


/*
Compare with Ren's method. #Ren#
*/

__device__ void ComputeForceMultiphaseCell(
	cint3 gridPos, 
	int i,
	cfloat3 pos, 
	cfloat3& force,
	SimData_SPH data) 
{

	uint gridHash = calcGridHash(gridPos);
	if (gridHash==GRID_UNDEF) 
		return;
	uint startIndex = data.gridCellStart[gridHash];
	if (startIndex == 0xffffffff)
		return;

	uint endIndex = data.gridCellEnd[gridHash];
	float h = dParam.smoothradius*0.5;
	float h2 = h*h;
	float support_radius = h*2;
	float fac = 32.0f / 3.141593 / pow(support_radius, 9);
	float sr6_d64 = pow(support_radius, 6)/64.0;
	

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 pos2 = data.pos[j];
		cfloat3 xij = pos - pos2;
		float d = xij.Norm();
		if (d >= h*2 || j==i)
			continue;
		
		cfloat3 vij = data.vel[i] - data.vel[j];
		cfloat3 nablawij = KernelGradient_Cubic(h, xij);

		cfloat3 u1u_i(0, 0, 0), u2u_i(0, 0, 0), u3u_i(0, 0, 0);
		cfloat3 u1u_j, u2u_j, u3u_j;
		for (int k=0; k<dParam.maxtypenum; k++)
		{
			float tmp = data.vFrac[i*dParam.maxtypenum+k] * dParam.densArr[k];
			u1u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].x * tmp;
			u2u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].y * tmp;
			u3u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].z * tmp;
		}

		if (data.type[j]==TYPE_FLUID)
		{
			
			//pressure
			//float pc = data.mass[j]* (data.pressure[j]/data.density[j]/data.density[j]
			//	+ data.pressure[i]/data.density[i]/data.density[i]);
			float pc = data.mass[j]* data.pressure[j]/data.density[j]
				+ data.pressure[i]*data.mass[i]/data.density[i];
			force += nablawij * pc * (-1) / data.density[i];

			//XSPH viscosity
			float volj = data.mass[j] / data.density[j];
			force += vij *(-1)* dParam.viscosity * volj *Kernel_Cubic(h, xij) 
				/ dParam.dt;


			if (data.group[i]==data.group[j])
			{
				float sf_kernel;
				/*if (d < h)
					sf_kernel = 2*pow((support_radius-d)*d, 3) - pow(support_radius, 6)/64.0;
				else if (d<support_radius)
					sf_kernel = pow((support_radius-d)*d, 3);
				else
					sf_kernel = 0;
				sf_kernel *= fac;

				cfloat3 sf_tension = xij * dParam.surface_tension * volj * sf_kernel / d *(-1);

				float kij = data.restDensity[i]/data.density[i] + data.restDensity[j]/data.density[j];
				force += sf_tension * kij;*/


				float c = (support_radius-d)*d;
				if (d < h)
					sf_kernel = 2*c*c*c - sr6_d64;
				else
					sf_kernel = c*c*c;

				sf_kernel *= fac * dParam.surface_tension * volj / d *(-1);

				float kij = data.restDensity[i]/data.density[i] + data.restDensity[j]/data.density[j];
				force += xij * (sf_kernel * kij);
			}


			/* Phase diffusion term. */
			cfloat3 nablaw = nablawij * volj;
			cfloat3 tmp(0, 0, 0);
			u1u_j.Set(0, 0, 0);
			u2u_j.Set(0, 0, 0);
			u3u_j.Set(0, 0, 0);
			
			for (int k=0; k<dParam.maxtypenum; k++) {
				float alphajk = data.vFrac[j*dParam.maxtypenum+k] * dParam.densArr[k];
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
			//force -= tmp / data.density[i];
		}

		if (data.type[j]==TYPE_RIGID) {
			/*float B=1;
				
			float q = d/h;
			if (q<0.66666)
				B *= 0.66666;
			else if (q<1)
				B *= 2*q - 1.5*q*q;
			else if (q<2)
				B *= 0.5 * (2-q)*(2-q);
			else
				B = 0;
			B *= 0.02 * 88.5*88.5 /d;

			float magnitude = data.mass[j]/(data.mass[i]+data.mass[j]) * B * dParam.boundstiff;
			force += xij * magnitude;*/

			float pc = data.pressure[i] * data.restDensity[j]
			+ data.pressure[i]*data.mass[i]/data.density[i];
			force += nablawij * pc * (-1) / data.density[i];

			//artificial viscosity
			cfloat3 xn = data.normal[j] * d;
			float xv = dot(vij, xn);

			if (xv < 0) {
				float visc = dParam.boundary_visc*dParam.smoothradius * 88.5 / data.density[i] * 0.25;
				float pi = visc * xv /(d*d + 0.01*h2);
				cfloat3 f = nablawij * pi * data.restDensity[i]*data.restDensity[j];
				force += f;
			}
		}
	}
}

__global__ void ComputeForceMultiphase_Kernel(SimData_SPH data, int nump)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= nump) 
		return;
	if (data.type[index]!=TYPE_FLUID) 
		return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	cfloat3 force(0, 0, 0);

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				ComputeForceMultiphaseCell(
					nPos,
					index,
					pos,
					force,
					data);
			}
	data.force[index] = force + cfloat3(0, -9.8, 0);
}




//====================================================
//
//                    #DFSPH#
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
					float visc = 2*dParam.boundary_visc*dParam.smoothradius * 88.5 /2000;
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
		case TYPE_CLOTH:
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
	
	/*cfloat3 u1u_i(0,0,0), u2u_i(0,0,0), u3u_i(0,0,0);
	cfloat3 u1u_j, u2u_j, u3u_j;
	for (int k=0; k<dParam.maxtypenum; k++)
	{
		u1u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].x * data.vFrac[i*dParam.maxtypenum+k];
		u2u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].y * data.vFrac[i*dParam.maxtypenum+k];
		u3u_i += data.drift_v[i*dParam.maxtypenum+k] * data.drift_v[i*dParam.maxtypenum+k].z * data.vFrac[i*dParam.maxtypenum+k];
	}*/


	//surface tension
	float support_radius = h*2;
	float fac = 32.0f / 3.141593 / pow(support_radius, 9);
	float sf_kernel;

	float sr6 = pow(dParam.smoothradius,6) / 64.0;

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
			//phase momentum diffusion
			//cfloat3 nablaw = KernelGradient_Cubic(h, xij) * volj;
			//cfloat3 tmp(0,0,0);
			//u1u_j.Set(0,0,0);
			//u2u_j.Set(0,0,0);
			//u3u_j.Set(0,0,0);
			//cfloat3 um_jk;
			//for (int k=0; k<dParam.maxtypenum; k++) {
			//	float alphajk = data.vFrac[j*dParam.maxtypenum+k];
			//	um_jk = data.drift_v[j*dParam.maxtypenum+k];

			//	//if(dot(um_jk,um_jk)>100)
			//	//	printf("super drift %f %f %f %d\n",  um_jk.x, um_jk.y, um_jk.z, data.uniqueId[j] );

			//	u1u_j += data.drift_v[j*dParam.maxtypenum+k] * data.drift_v[j*dParam.maxtypenum+k].x * alphajk;
			//	u2u_j += data.drift_v[j*dParam.maxtypenum+k] * data.drift_v[j*dParam.maxtypenum+k].y * alphajk;
			//	u3u_j += data.drift_v[j*dParam.maxtypenum+k] * data.drift_v[j*dParam.maxtypenum+k].z * alphajk;
			//}
			//u1u_j += u1u_i;
			//u2u_j += u2u_i;
			//u3u_j += u3u_i; 

			//tmp.x += dot(nablaw, u1u_j);
			//tmp.y += dot(nablaw, u2u_j);
			//tmp.z += dot(nablaw, u3u_j);
			//force -= tmp;
			
			//xsph artificial viscosity [Schechter 13]
			if(data.group[i]==0 || data.group[j]==0)
				force += vij * (dParam.viscosity * volj *(-1) *Kernel_Cubic(h, xij) / dParam.dt);
			else
				force += vij * (0.5 * volj *(-1) *Kernel_Cubic(h, xij) / dParam.dt);

			
		}

		//surface tension
		if (data.group[i]==data.group[j])
		{
			float c = support_radius - d;

			if (d < h)
				sf_kernel = 2* c*c*c*d*d2 - sr6;
			else if (d<support_radius)
				sf_kernel = c*c*c*d*d2;
			else
				sf_kernel = 0;
			sf_kernel *= fac;

			float tmp = dParam.surface_tension;
			if (data.group[i]==4)
				tmp *= 10;
			cfloat3 sf_tension = xij * (tmp * volj * sf_kernel / d *(-1));
			float kij = data.restDensity[i]/data.density[i] + data.restDensity[j]/data.density[j];
			force += sf_tension;
		}

		if (data.type[i]==TYPE_DEFORMABLE && data.type[j]==TYPE_DEFORMABLE) {
			
			force += vij * (dParam.solid_visc * volj *(-1) *Kernel_Cubic(h, xij) / dParam.dt);
		}

		if (data.type[j]==TYPE_RIGID && data.group[j]==0) {

			//artificial viscosity
			float xv = dot(vij, xij);
			cfloat3& n = data.normal[j];
			cfloat3 xn = n * d;
			cfloat3 vn = n * dot(n, vij);
			cfloat3 vt = vij - vn;
			//float xv = dot(vij, xn);

			if (xv < 0) {
				float visc = dParam.boundary_visc*dParam.smoothradius * 88.5 / data.density[i] * 0.25;
				float pi = visc * xv /(d2 + 0.01*h2);
				cfloat3 f = nablaw * pi * data.restDensity[i]*data.restDensity[j];
				
				cfloat3 fn = data.normal[j] *  dot(f,data.normal[j]);
				cfloat3 ft = vt * (fn.Norm() * (-1) * dParam.boundary_friction);
				force += fn;
				force += ft;
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

	if (data.type[index] == TYPE_CLOTH) { //temp
		data.v_star[index] = data.vel[index] + dParam.gravity * dParam.dt;
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
	//printf("%f %f %f\n", data.v_star[index].x, data.v_star[index].y, data.v_star[index].z);
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

	//if (data.type[index] == TYPE_CLOTH)
	//	printf("inter4: %f %f %f\n", data.pos[index].x, data.pos[index].y, data.pos[index].z);

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
		/*if (data.type[i] == TYPE_CLOTH && data.type[j] == TYPE_CLOTH)
		{
			bool same = false;
			for (int k = 0; k < 8; k++)
			{
				int neighbor = data.indexTable[data.adjacent_index[i][k]];
				if (neighbor == j)
				{
					same = true;
					break;
				}
			}
			if (same)
				continue;
		}*/

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;

		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;
		if (d2 >= sr2 || d2<EPSILON) continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		
		switch (data.type[j]) {
		case TYPE_FLUID:
			
			force += nabla_w * (data.pstiff[i] * mass_i*mass_i / data.density[i]
				+ data.pstiff[j] * data.mass[j] * data.mass[j] / data.density[j]);

			break;
		case TYPE_DEFORMABLE:
		case TYPE_CLOTH:
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
	int num_particles,
	float* stiffbuf
) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	//if (data.type[index] == TYPE_CLOTH) return;

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
	stiffbuf[index] += data.pstiff[index];
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

	if (data.type[index] == TYPE_CLOTH) return; //temp

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
	data.v_star[index] += force * dParam.dt *(-1) / data.effective_mass[index] * 0.5;
}

__global__ void EnforceDivergenceWarmStart(
	SimData_SPH data,
	int num_particles
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	if (data.type[index] == TYPE_CLOTH) return;

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



__global__ void DriftVelocityKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) 
		return;

	/*
	Drift velocity here only contains the first term, i.e. the inertial separating term.
	This term is only valid for fluid particles.
	*/

	if (data.type[index]!=TYPE_FLUID)
		return;

	cfloat3 xi = data.pos[index];
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	cfloat3* flux_buffer= &data.flux_buffer[index*dParam.maxtypenum];
	
	cfloat3 drift_acceleration = data.v_star[index];
	float dynamic_constant = dParam.drift_dynamic_diffusion; 

	float rest_density = data.restDensity[index];
	float* vol_frac = data.vFrac + index*dParam.maxtypenum;

	for (int k=0; k<dParam.maxtypenum; k++) {
		float density_k = dParam.densArr[k];
		float vol_frac_k = vol_frac[k];
		
		//if phase k doesnot exist， continue
		if (vol_frac_k < EPSILON) { 
			drift_v[k].Set(0,0,0);	continue;
		}

		//dynamic term
		float density_factor = (density_k - rest_density) / rest_density;
		cfloat3 drift_vk = drift_acceleration * dynamic_constant * density_factor;
		drift_v[k] = drift_vk;
		flux_buffer[k] = drift_vk * vol_frac_k; //alpha_k drift_vk
	}
}

__device__ void PredictPhaseDiffusionCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	float vol_i,
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
	float vol0 = dParam.spacing*dParam.spacing*dParam.spacing;
	int num_type = dParam.maxtypenum;
	
	float turbulenti = data.vel[i].square() * dParam.drift_turbulent_diffusion 
		+ dParam.drift_thermal_diffusion;

	
	for (uint j = start_index; j < end_index; j++)
	{
		if (data.type[j]!=TYPE_FLUID || j == i )
			continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr)
			continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		float vol_j = data.mass[j] / data.density[j];
		float fac = dot(xij, nabla_w)/(d2+0.01*sr2*0.25)*2;
		

		cfloat3 flux_k(0,0,0);

		/* The turbulent factor is averaged between particle i and j. */
		float turbulentj = (data.vel[j].square() * dParam.drift_turbulent_diffusion
			+ dParam.drift_thermal_diffusion + turbulenti)*0.5;

		for (int k=0; k<num_type; k++) {
			
			//flux_k = data.drift_v[i*num_type+k] * (vol_i * data.vFrac[i*num_type+k])
			//	+ data.drift_v[j*num_type+k] * (vol_j * data.vFrac[j*num_type+k]) ;
			flux_k = data.flux_buffer[i*num_type+k] + data.flux_buffer[j*num_type+k];
			vol_frac_change[k] -= dot(flux_k, nabla_w) * vol0;
				
			/* turbulent diffusion
			Evaluating nabla.Dot(nabla alpha) doesnot ensure positive
			volume fraction. Evaluate nabla^2 alpha instead. */

			float factmp = fac * (data.vFrac[i*num_type+k] - data.vFrac[j*num_type+k]);
			vol_frac_change[k] += factmp * turbulentj * vol0;
		}
		
	}

}

__global__ void PredictPhaseDiffusionKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]!=TYPE_FLUID){
		return;
	}

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

	//Get flux multiplier lambda for each particle.
	
	float lambda = 1;
	for (int k=0; k<dParam.maxtypenum; k++) {
		
		if(vol_frac_change[k]>=0) 
			continue;

		if (data.vFrac[index*dParam.maxtypenum+k]<EPSILON) {
			lambda=0;
			continue;
		}

		float lambda_k = data.vFrac[index*dParam.maxtypenum+k]/dParam.dt
			/abs(vol_frac_change[k]);
		
		// Always keep the smallest value.
		if(lambda_k < lambda)
			lambda = lambda_k;
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
	float vol0 = dParam.spacing*dParam.spacing*dParam.spacing;
	int num_type = dParam.maxtypenum;



	float turbulenti = data.vel[i].square() * dParam.drift_turbulent_diffusion
		+ dParam.drift_thermal_diffusion;

	float dissolution_factor = dParam.dissolution;

	for (uint j = start_index; j < end_index; j++)
	{
		if (j == i || data.type[j]==TYPE_RIGID) 
			continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2)
			continue;


		cfloat3 nabla_w = KernelGradient_Cubic(sr/2, xij);
		float vol_j = data.mass[j] / data.density[j];
		float fac = dot(xij, nabla_w)/(d2+0.01*sr2*0.25)*2;
		float lambda_ij = fmin(data.phase_diffusion_lambda[i], data.phase_diffusion_lambda[j]);
		
		/* Inertial separating flux only happens between fluid particles.*/
		if (data.type[i]==TYPE_FLUID && data.type[j]==TYPE_FLUID) {

			cfloat3 flux_k;

			/* The turbulent factor is averaged between particle i and j. */
			float turbulentj = (data.vel[j].square() * dParam.drift_turbulent_diffusion
				+ dParam.drift_thermal_diffusion + turbulenti)*0.5;
			 
			for (int k=0; k<num_type; k++) {
				/*flux_k = data.drift_v[i*num_type+k] * vol_i * data.vFrac[i*num_type+k]
					+ data.drift_v[j*num_type+k] * vol_j * data.vFrac[j*num_type+k];
				vol_frac_change[k] += dot(flux_k, nabla_w) * (-1) * lambda_ij;*/
				
				flux_k = data.flux_buffer[i*num_type+k] + data.flux_buffer[j*num_type+k];
				vol_frac_change[k] -= dot(flux_k, nabla_w) * vol0 * lambda_ij;

				float factmp = fac * (data.vFrac[i*num_type+k] - data.vFrac[j*num_type+k]);
				vol_frac_change[k] += factmp * turbulentj * vol0 * lambda_ij;
			}
		}
		else if( dParam.enable_dissolution && !(data.type[i]==TYPE_DEFORMABLE && data.type[j]==TYPE_DEFORMABLE) )
			/* i: fluid or deformable, j: fluid or deformable,
			this branch means (i,j) = (fluid, deformable) or (deformable, fluid)*/
		{
			bool dissolution = true;
			int fid;
			if(data.type[i]==TYPE_FLUID) fid = i;
			else fid = j;
			/* check if saturation is reached. This is not normalized with lambdaij, since
			no negative volume fraction appears in this process. 
			*/

			float vik, vjk;
			for (int k=0; k<num_type; k++) {
				if(data.vFrac[fid*num_type+k] >= dParam.max_alpha[k])
					dissolution = false;
			}

			/* Particles exchange the component only when on saturation happens.
			*/

			if (dissolution)
			{
				for (int k=0; k<num_type; k++) {
					float factmp = fac * (data.vFrac[i*num_type+k] - data.vFrac[j*num_type+k]);	
					vol_frac_change[k] += factmp * dissolution_factor * vol0;
				}
			}
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
		vol_frac_change[k]=0;

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
		
		if( !(vol_frac[k]>-EPSILON)) {
			//printf("%d %d %f %f\n",data.uniqueId[index], k, vol_frac[k], data.phase_diffusion_lambda[index]);
			vol_frac[k] = 0;
			
		}
		normalize += vol_frac[k];
	}
	for (int k=0; k<dParam.maxtypenum; k++)
		vol_frac[k] /= normalize;


	if (data.type[index]==TYPE_DEFORMABLE && data.vFrac[index*dParam.maxtypenum+0] < dParam.max_alpha[0])
	{
		data.type[index] = TYPE_FLUID;
	}
}




__device__ void HeatConductionCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	float& dT) {

	uint grid_hash = calcGridHash(cell_index);
	if (grid_hash==GRID_UNDEF)
		return;

	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff)
		return;
	uint end_index = data.gridCellEnd[grid_hash];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float vol0 = dParam.spacing*dParam.spacing*dParam.spacing;
	
	float heat_flow_rate = dParam.heat_flow_rate;

	for (uint j = start_index; j < end_index; j++)
	{
		if (j == i || data.type[j]==TYPE_RIGID)
			continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d2 = xij.square();
		if (d2 >= sr2)
			continue;

		cfloat3 nabla_w = KernelGradient_Cubic(sr*0.5, xij);
		float fac = dot(xij, nabla_w)/(d2+0.01*sr2*0.25)*2;
		fac = fac * (data.temperature[i] - data.temperature[j]);
		dT += fac * heat_flow_rate * vol0;
	}
}

__global__ void HeatConductionKernel(
	SimData_SPH data,
	int num_particles
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	
	float dT = 0;
	
	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				HeatConductionCell(
					neighbor_cell_index,
					index,
					xi,
					data,
					dT);
			}
	
	dT *= dParam.dt; // heat change

	float heat_capacity = 0;
	float vol = dParam.spacing * dParam.spacing * dParam.spacing;
	for(int k=0;k<dParam.maxtypenum;k++)
		heat_capacity += data.vFrac[index*dParam.maxtypenum+k] * dParam.heat_capacity[0];

	// test temperature
	float tmp = data.temperature[index] + dT / heat_capacity;
	
	if (data.type[index] == TYPE_DEFORMABLE && tmp > dParam.melt_point) {
		
		if (data.heat_buffer[index] < dParam.latent_heat - EPSILON) { //heat buffer not full
			
			float space = dParam.latent_heat - data.heat_buffer[index]; //empty space in heat buffer
			if ( space >= dT ) { //fill buffer, no temperature change
				data.heat_buffer[index] += dT;
			}
			else { //fill buffer, then update temperature
				data.heat_buffer[index] = dParam.latent_heat;
				dT -= space;
				data.temperature[index] += dT / heat_capacity;
			}
		}
		else { // heat buffer is full
			data.temperature[index] = tmp;
		}
	}
	else {
		data.temperature[index] = tmp;
	}

	//data.color[index].Set(data.temperature[index]/100, 0, 0, 1);
	if (data.type[index]==TYPE_DEFORMABLE && data.temperature[index]>dParam.melt_point + EPSILON) {
		data.type[index] = TYPE_FLUID;
	}
}


/*
The drift velocity and alpha_gradient for Ren's model is
actually the same as ours.
*/


__device__ void VolumeFractionGradientCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cfloat3* vol_frac_gradient)
{
	uint grid_hash = calcGridHash(cell_index);
	if (grid_hash==GRID_UNDEF)
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


__global__ void DriftVelRenKernel(SimData_SPH data, int nump)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= nump)
		return;
	if (data.type[index]!=TYPE_FLUID)
		return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	cfloat3* vol_frac_gradient = &data.vol_frac_gradient[index*dParam.maxtypenum];

	/*
	The volume fraction related term corresponds to turbulent diffusion.
	Storing alpha_gradient into a separate variable.
	*/
	float turbulent_diffusion = dParam.drift_thermal_diffusion;
	for (int k=0; k<dParam.maxtypenum; k++)
		vol_frac_gradient[k].Set(0, 0, 0); //initialization

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


	/*
	The acceleration related term corresponds to inertial separation.
	*/
	cfloat3 drift_acceleration = data.v_star[index];
	float dynamic_constant = dParam.drift_dynamic_diffusion;

	float rest_density = data.restDensity[index];
	float accel_mode = drift_acceleration.Norm();
	float* vol_frac = data.vFrac + index*dParam.maxtypenum;

	for (int k=0; k<dParam.maxtypenum; k++) {
		float density_k = dParam.densArr[k];
		float vol_frac_k = vol_frac[k];

		//if phase k doesnot exist, continue
		if (vol_frac_k < EPSILON) {
			drift_v[k].Set(0, 0, 0);	continue;
		}

		//dynamic term
		float density_factor = (density_k - rest_density) / rest_density;
		cfloat3 drift_vk = drift_acceleration * dynamic_constant * density_factor;
		drift_v[k] = drift_vk;

		/*if (dot(drift_v[k], drift_v[k])>100) {
			printf("super drift? %f %f %f from %f %f %f\n", drift_v[k].x, drift_v[k].y, drift_v[k].z, drift_acceleration.x, drift_acceleration.y, drift_acceleration.z);
		}*/
	}
}


__device__ void PhaseDiffusionRenCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	float vol_i,
	SimData_SPH& data,
	float* vol_frac_change) {

	uint grid_hash = calcGridHash(cell_index);
	if (grid_hash==GRID_UNDEF) return;
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
			vol_frac_change[k] += dot(flux_k, nabla_w) * (-1);

			//turbulent diffusion
			flux_k = data.vol_frac_gradient[i*num_type+k]*vol_i
				+ data.vol_frac_gradient[j*num_type+k]*vol_j;
			vol_frac_change[k] += dot(flux_k, nabla_w) * (-1);

			/*The first term in Ren's paper. alpha_k nabla.Dot(u_m)
			Appears here.*/
			flux_k = data.vel[j]*vol_j*data.vFrac[j*num_type+k]
				-  data.vel[i]*vol_i*data.vFrac[i*num_type+k];
			
			vol_frac_change[k] += dot(flux_k, nabla_w) * (-0.5) * dParam.drift_turbulent_diffusion;
		}
	}

}


__global__ void PhaseDiffusionRenKernel(SimData_SPH data, int num_p)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_p) return;
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);
	cfloat3* drift_v = &data.drift_v[index*dParam.maxtypenum];
	float* vol_frac_change = &data.vol_frac_change[index*dParam.maxtypenum];
	float vol_i = data.mass[index] / data.density[index];

	for (int k=0; k<dParam.maxtypenum; k++)
		vol_frac_change[k]=0; //initialization

	for (int z=-1; z<=1; z++)
	for (int y=-1; y<=1; y++)
	for (int x=-1; x<=1; x++)
	{
		cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
		PhaseDiffusionRenCell(
			neighbor_cell_index,
			index,
			xi,
			vol_i,
			data,
			vol_frac_change);
	}

	/*if (data.uniqueId[index]%10==0) {
		printf("%f %f %f\n", vol_frac_change[0], vol_frac_change[1], vol_frac_change[2]);
	}
*/
	/*if (data.uniqueId[index]==0) {
		printf("%f %f %f\n", vol_frac_change[0], vol_frac_change[1],
			vol_frac_change[2]);
	}*/

}










/*===================================

   #Rigid# Rigid fluid coupling.

====================================*/


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














/*=====================================

    #Deformable# Elastoplastic Model

======================================*/




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

__global__ void ComputeTensionCauchyStress_Kernel(SimData_SPH data, int num_particles)
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



/*
Compute tension with first Piola-Kirchhoff stress tensor P.
*/


__device__ void ComputeTensionWithP_Cell(
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cfloat3& tension
) {

	float h = dParam.smoothradius*0.5;


	cfloat3 nablawij;
	cfloat3 nablawi, nablawj;
	cfloat3 fi, fj;
	cmat3& P_i = data.cauchy_stress[i];
	int localid_i = data.local_id[i];
	cfloat3 x0i = data.x0[localid_i];
	cmat3 RL_i, RL_j;
	mat3prod(data.rotation[localid_i], data.correct_kernel[localid_i], RL_i);

	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	cfloat3* neighbordx = &data.neighbordx[localid_i*NUM_NEIGHBOR];
	int num_neighbor = neighborlist[0];

	for (uint niter = 1; niter <= num_neighbor; niter++)
	{

		/* Neighborlist stores the unique id of neighboring particles.
		We look up the index table to get the current index.
		*/

		int j = data.indexTable[neighborlist[niter]];
		int localid_j = data.local_id[j];

		cfloat3 x0ij = x0i - data.x0[data.local_id[j]];
		cmat3& sigma_j = data.cauchy_stress[j];
		nablawij = KernelGradient_Cubic(h, x0ij);
		mvprod(RL_i, nablawij, nablawi);
		mat3prod(data.rotation[localid_j], data.correct_kernel[localid_j], RL_j);
		mvprod(RL_j, nablawij, nablawj);

		mvprod(data.cauchy_stress[i], nablawi, fi);
		mvprod(data.cauchy_stress[j], nablawj, fj);
		tension += fi + fj;
	}
}

__device__ void ComputeTensionWithP_Cell_Plastic(
	int i,
	SimData_SPH& data,
	cfloat3& tension
) {

	float h = dParam.smoothradius*0.5;


	cfloat3 nablawij;
	cfloat3 nablawi, nablawj;
	cfloat3 fi, fj;
	cmat3& P_i = data.cauchy_stress[i];
	int localid_i = data.local_id[i];
	cmat3 RL_i, RL_j;
	mat3prod(data.rotation[localid_i], data.correct_kernel[localid_i], RL_i);

	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	cfloat3* neighbordx = &data.neighbordx[localid_i*NUM_NEIGHBOR];
	int num_neighbor = neighborlist[0];

	for (uint niter = 1; niter <= num_neighbor; niter++)
	{

		/* Neighborlist stores the unique id of neighboring particles.
		We look up the index table to get the current index.
		*/

		int j = data.indexTable[neighborlist[niter]];

		cfloat3 x0ij = data.neighbordx[localid_i*NUM_NEIGHBOR + niter];
		cmat3& sigma_j = data.cauchy_stress[j];

		// x0ij from particle i
		if (! (x0ij.Norm()<dParam.smoothradius))
			continue;
		nablawij = KernelGradient_Cubic(h, x0ij);
		mvprod(RL_i, nablawij, nablawi);

		// x0ij from particle j
		int localid_j = data.local_id[j];
		int* neighborlist_j = &data.neighborlist[localid_j*NUM_NEIGHBOR];
		cfloat3* neighbordx_j = &data.neighbordx[localid_j*NUM_NEIGHBOR];
		for (int k=1; k<=neighborlist_j[0]; k++) {
			if (neighborlist_j[k]==data.uniqueId[i])
				x0ij = neighbordx_j[k];
		}
		if (! (x0ij.Norm()<dParam.smoothradius))
			continue;
		x0ij *= (-1);
		nablawij = KernelGradient_Cubic(h, x0ij);
		mat3prod(data.rotation[localid_j], data.correct_kernel[localid_j], RL_j);
		mvprod(RL_j, nablawij, nablawj);

		mvprod(data.cauchy_stress[i], nablawi, fi);
		mvprod(data.cauchy_stress[j], nablawj, fj);
		tension += fi + fj;
	}
}

__global__ void ComputeTensionWithP_Kernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;

	if (data.type[index] == TYPE_DEFORMABLE)
	{

		cfloat3 x0i = data.x0[data.local_id[index]];

		float vol = dParam.spacing*dParam.spacing*dParam.spacing;
		cfloat3  tension(0, 0, 0);
		ComputeTensionWithP_Cell_Plastic(
			index,
			data,
			tension);

		tension *= vol*vol / data.mass[index];
		data.vel_right[index] = data.v_star[index] + tension * dParam.dt;

		data.v_star[index] = data.vel_right[index];
	}

	if (data.type[index] == TYPE_CLOTH)
	{
		
		float ladj = dParam.spacing;
		float ldiag = ladj * 1.4142;
		float lbend = ladj * 2;

		cfloat3 pos = data.pos[index] + data.v_star[index] * dParam.dt;
		cfloat3 tension;

		for (int iter = 0; iter < 10; iter++)
		{
			tension = cfloat3(0, 0, 0);
			for (int neighbor_num = 0; neighbor_num < 12; neighbor_num++)
			{
				int neighbor = data.adjacent_index[index][neighbor_num];
				if (neighbor == -1)
					continue;
				int j = data.indexTable[neighbor];
				cfloat3 xij = pos - data.pos[j];
				float norm = xij.Norm();
				if (neighbor_num < 4)
				{
					if (norm > ladj)
						tension -= xij / norm * dParam.kadj * (norm - ladj);
				}
				else if (neighbor_num < 8)
				{
					if (norm > ldiag)
						tension -= xij / norm * dParam.kdiag * (norm - ldiag);
				}
				else
				{
					if (norm > lbend)
						tension -= xij / norm * dParam.kbend * (norm - lbend);
				}
			}
			cfloat3 right = ((pos - data.pos[index]) / dParam.dt - data.v_star[index]) * data.mass[index] / dParam.dt;
			if ((tension - right).square() < 1e-10)
				break;
			pos = data.pos[index] + (data.v_star[index] + tension * dParam.dt / data.mass[index]) * dParam.dt;
		}
		data.v_star[index] += tension / data.mass[index] * dParam.dt;
		//if (data.v_star[index].Norm() > 5)
		//	printf("duang! tension1 = %f %f %f\n\ttension2 = ", tension.x, tension.y, tension.z);
	}
}



__device__ void DeformationGradient_Cell(
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	cmat3& F
) {

	float h = dParam.smoothradius * 0.5;
	float vol = dParam.spacing*dParam.spacing*dParam.spacing;
	int localid_i = data.local_id[i];
	cmat3& L = data.correct_kernel[localid_i];
	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	int num_neighbor = neighborlist[0];

	for (uint niter = 1; niter <= num_neighbor; niter++)
	{

		/* Neighborlist stores the unique id of neighboring particles.
		We look up the index table to get the current index.
		*/

		int j = data.indexTable[neighborlist[niter]];

		cfloat3 xji = data.pos[j] - xi;
		xji *= vol;

		cfloat3 x0ij = data.neighbordx[localid_i*NUM_NEIGHBOR + niter];

		cfloat3 nablaw = KernelGradient_Cubic(h, x0ij);

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
	int i,
	cfloat3 xi,
	cfloat3 x0i,
	SimData_SPH& data,
	cmat3& R,
	cmat3& rotated_F
) {

	float h = dParam.smoothradius * 0.5;
	float vol = dParam.spacing*dParam.spacing*dParam.spacing;
	int localid_i = data.local_id[i];
	cmat3& L = data.correct_kernel[localid_i];
	cmat3 RL;
	mat3prod(R,L,RL);

	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	int num_neighbor = neighborlist[0];

	for (uint niter = 1; niter <= num_neighbor; niter++)
	{

		/* Neighborlist stores the unique id of neighboring particles.
		We look up the index table to get the current index.
		*/
		int j = data.indexTable[neighborlist[niter]];

		cfloat3 xji = data.pos[j] - xi;

		cfloat3 x0ij = x0i - data.x0[data.local_id[j]];

		cfloat3 Rx0ji = x0ij*(-1);
		mvprod(R, Rx0ji, Rx0ji);

		xji = xji - Rx0ji;
		xji *= vol;

		cfloat3 x0ij_kernel = data.neighbordx[localid_i*NUM_NEIGHBOR + niter];
		cfloat3 nablaw = KernelGradient_Cubic(h, x0ij_kernel);

		mvprod(RL, nablaw, nablaw);

		rotated_F.Add(TensorProduct(xji, nablaw));
	}
}



#include "catpaw/svd3_cuda.h"

__host__ __device__ __forceinline__
void svd(cmat3& A, cmat3& U, cmat3& S, cmat3& V) {
	svd(A[0][0], A[0][1], A[0][2],
		A[1][0], A[1][1], A[1][2],
		A[2][0], A[2][1], A[2][2],
		U[0][0], U[0][1], U[0][2],
		U[1][0], U[1][1], U[1][2],
		U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2],
		S[1][0], S[1][1], S[1][2],
		S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2],
		V[1][0], V[1][1], V[1][2],
		V[2][0], V[2][1], V[2][2]
		);
}

__device__ void PlasticityVonMises(
	SimData_SPH& data,
	int i,
	float yieldi,
	float pn,
	cmat3& F,
	cmat3& Fp
)
{

	cmat3 U,S,V;
	svd(F, U,S,V);
	float flow = dParam.plastic_flow;

	float gamma = flow * (pn - yieldi) / pn;
	if(gamma > 1)
		gamma = 1;

	//clamp negative singular value
	//There's some bug inside the svd code

	if (S[0][0]<0) S[0][0]=1;
	if (S[1][1]<0) S[1][1]=1;
	if (S[2][2]<0) S[2][2]=1;

	cmat3 Fp_diag; //initialized as full-zero
	Fp_diag[0][0] = pow( S[0][0], gamma );
	Fp_diag[1][1] = pow( S[1][1], gamma );
	Fp_diag[2][2] = pow( S[2][2], gamma );
	
	/*Only need to get Fp, and update the reference shape
	with Fp.
	Fp = V Fp_diag VT
	Fe = U Fe_diag VT
	F = FeFp = U Fe_diag Fp_diag VT = USV
	*/
	

	cmat3 VT, tmp;
	mat3transpose(V,VT);
	mat3prod(V, Fp_diag, tmp);
	mat3prod(tmp, VT, Fp);
}

__device__ void UpdateX0_Elastoplastic(
	SimData_SPH& data,
	int i,
	cmat3& Fp
)
{
	//update x0_ij
	int localid_i = data.local_id[i];
	cfloat3* neighbordx = &data.neighbordx[localid_i*NUM_NEIGHBOR];
	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	int& num_neighbor = neighborlist[0];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float volj = dParam.spacing*dParam.spacing*dParam.spacing;

	
	
	for (int iter=1; iter<=num_neighbor; iter++)
	{
		cfloat3& x0ij = data.neighbordx[localid_i*NUM_NEIGHBOR+iter];
		mvprod(Fp, x0ij, x0ij);

		/*cfloat3 xij = data.pos[i] - data.pos[data.indexTable[
		neighborlist[iter]]];*/
	}

	//update kernel correction matrix
	/*cmat3 kernel_tmp;
	for (int iter=1; iter<=num_neighbor; iter++)
	{
		cfloat3 xij = neighbordx[iter];
		cfloat3 nablaw = KernelGradient_Cubic(sr*0.5, xij);
		xij *= (-1) * volj;
		kernel_tmp.Add(TensorProduct(nablaw, xij));
	}
	cmat3 kernel_corr = kernel_tmp.Inv();*/
	//cmat3 kernel_corr;
	//if (kernel_corr.Norm()<1e-10) {
	//	//printf("Degenerate correction matrix. Using Identity.\n");
	//	kernel_corr[0][0] = 1;
	//	kernel_corr[1][1] = 1;
	//	kernel_corr[2][2] = 1;
	//}
	//	
	//data.correct_kernel[localid_i] = kernel_corr;
}


__global__ void UpdateSolidStateF_Kernel(
	SimData_SPH data, 
	int num_particles,
	int projection_type
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	cfloat3 xi = data.pos[index];
	cfloat3 x0i = data.x0[data.local_id[index]];
	cint3 cell_index = calcGridPos(xi);
	cmat3 F;
	cmat3& R = data.rotation[data.local_id[index]];

	DeformationGradient_Cell(index, xi, data, F);
	
	if (!(F.Norm()>1e-10)) {
		int nn = data.neighborlist[ data.local_id[index]*NUM_NEIGHBOR ];
		printf("oops1 %d %f\n", nn ,data.vFrac[index*dParam.maxtypenum+0]);
		data.color[index].Set(0,1,0,1);
	}
	if (!(F.Det()>1e-10)) {
		F.Set(0.0);
		F[0][0]=1;
		F[1][1]=1;
		F[2][2]=1;
	}

	ExtractRotation(F, R, 10);

	F.Set(0.0);
	RotatedDeformationGradient_Cell(index, xi, x0i, data, R, F);
	F[0][0]+=1;
	F[1][1]+=1;
	F[2][2]+=1;
	
	data.gradient[index] = F;

	//Compute strain epsilon.
	cmat3 F_T;
	mat3transpose(F, F_T);
	cmat3 epsilon;
	epsilon = (F+F_T) * 0.5;
	epsilon[0][0] -= 1;
	epsilon[1][1] -= 1;
	epsilon[2][2] -= 1;

	
	//First Piola-Kirchhoff Stress
	cmat3 P;
	float mu = dParam.solidG;
	float lambda = dParam.solidK - mu*2.0/3.0; //Lame parameters.
	P = epsilon * 2*mu;
	float trace = epsilon[0][0] + epsilon[1][1] + epsilon[2][2];
	P[0][0] += lambda * trace;
	P[1][1] += lambda * trace;
	P[2][2] += lambda * trace;

	data.cauchy_stress[index] = P;
	if (! (P.Norm()<1e10))
	{
		P.Set(0.0);
	}
}

__device__ void SpatialColorFieldCell(
	cint3 cell_index,
	int i,
	cfloat3 xi,
	SimData_SPH& data,
	int& bcount,
	float& flag)
{
	uint grid_hash = calcGridHash(cell_index);
	if (grid_hash==GRID_UNDEF)
		return;

	uint start_index = data.gridCellStart[grid_hash];
	if (start_index == 0xffffffff)
		return;
	uint end_index = data.gridCellEnd[grid_hash];

	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	float mindis = 99;
	cfloat3 nearest_xij;

	for (uint j = start_index; j < end_index; j++)
	{
		if (data.type[j]!=TYPE_RIGID || data.group[j]!=2) //2 is scripted object
			continue;

		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		float d = xij.Norm();
		//if (d >= sr)
		//	continue;
		
		if (d < mindis) {
			mindis = d; 
			nearest_xij = xij / d;
			flag = dot(nearest_xij, data.normal[j]);
		}
		
		bcount ++;
	}
}

__global__ void SpatialColorFieldKernel(
	SimData_SPH data,
	int num_particles
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	if (data.type[index]==TYPE_RIGID) return;

	cfloat3 xi = data.pos[index];
	cint3 cell_index = calcGridPos(xi);

	int bcount=0;
	float flag = 0;

	for (int z=-1; z<=1; z++)
		for (int y=-1; y<=1; y++)
			for (int x=-1; x<=1; x++)
			{
				cint3 neighbor_cell_index = cell_index + cint3(x, y, z);
				SpatialColorFieldCell(
					neighbor_cell_index,
					index,
					xi,
					data,
					bcount,
					flag);
			}
	/*if (bcount >= 3) {
		data.spatial_color[index] = flag/abs(flag);
		if (flag > EPSILON)
			data.color[index].Set(0,0,1,1);
		else if(flag < -EPSILON)
			data.color[index].Set(0,1,0,1);
	}
	else {
		data.spatial_color[index] = 0;
	}*/

	if(abs(flag)>0.1)
		data.spatial_color[index] = flag;
	else
		data.spatial_color[index] = 0;

	/*if (flag > EPSILON)
		data.color[index].Set(0, 0, 1, 1);
	else if (flag < -EPSILON)
		data.color[index].Set(0, 1, 0, 1);*/
}

__global__ void Trim0(
	SimData_SPH data,
	int num_particles
)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;

	int localid_i = data.local_id[index];
	cfloat3 x0i = data.x0[localid_i];
	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	cfloat3* neighbordx = &data.neighbordx[localid_i*NUM_NEIGHBOR];
	float* len0_i = &data.length0[localid_i*NUM_NEIGHBOR];

	int* trim_tag = &data.trim_tag[localid_i*NUM_NEIGHBOR];
	int num_neighbor = neighborlist[0];


	for (uint niter = 1; niter <= num_neighbor; niter++)
	{

		int j = data.indexTable[neighborlist[niter]];
		int localid_j = data.local_id[j];
		int* neighborlist_j = &data.neighborlist[localid_j*NUM_NEIGHBOR];
		cfloat3* neighbordx_j = &data.neighbordx[localid_j*NUM_NEIGHBOR];
		float* len0_j = &data.length0[localid_j*NUM_NEIGHBOR];
		bool found = false;
		
		cfloat3 x0ji;
		float x0ji_len0;
		
		for (int k=1; k<=neighborlist_j[0]; k++) {
			if (neighborlist_j[k]==data.uniqueId[index]){
				x0ji = neighbordx_j[k];
				x0ji_len0 = len0_j[k];
				found = true;
			}
		}
		
		cfloat3 x0ij = neighbordx[niter];
		float x0ij_len0 = len0_i[niter];
		

		if ( x0ij.Norm()/x0ij_len0 > 1.5 || 
			x0ij.Norm() > dParam.smoothradius-EPSILON ||
			!(found) ||
			x0ji.Norm()/x0ji_len0 > 1.5 ||
			x0ji.Norm() > dParam.smoothradius-EPSILON ||

			data.type[j] != TYPE_DEFORMABLE //solid marked as dissolved or molten
			|| data.spatial_color[index]*data.spatial_color[j]<-EPSILON
			)
		{
			trim_tag[niter] = 1;
		}
	}

}


__global__ void Trim1(
	SimData_SPH data,
	int num_particles
) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_DEFORMABLE)
		return;
	

	int localid_i = data.local_id[index];
	
	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	cfloat3* neighbordx = &data.neighbordx[localid_i*NUM_NEIGHBOR];
	float* len0 = &data.length0[localid_i*NUM_NEIGHBOR];
	
	int* trim_tag = &data.trim_tag[localid_i*NUM_NEIGHBOR];
	int& num_neighbor = neighborlist[0];
	int trimed_id[NUM_NEIGHBOR];
	int validcount = 1;

	for (uint iter = 1; iter <= num_neighbor; iter++)
	{
		if (trim_tag[iter]==1)
			trimed_id[iter]=-1;
		else
			trimed_id[iter]=validcount++;
	}
	
	for (int iter=1; iter<=num_neighbor; iter++)
	{
		if (trimed_id[iter]!=-1) {
			int tmp = trimed_id[iter];
			neighborlist[tmp] = neighborlist[iter]; //substitute backward
			neighbordx[tmp] = neighbordx[iter];
			len0[tmp] = len0[iter];
		}
	}
	int tmp = num_neighbor;
	num_neighbor = validcount-1;

	if (num_neighbor < 1) {
		data.type[index] = TYPE_FLUID;
	}
	//if(num_neighbor < tmp)
	//	printf("%d %d -> %d\n", data.uniqueId[index], tmp, num_neighbor);
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
			
			data.neighbordx[localid_i*NUM_NEIGHBOR + neighborcount] = xij;
			data.length0[localid_i*NUM_NEIGHBOR + neighborcount] = xij.Norm();
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

extern HDFUNC cmat3 MooreInv(cmat3 A);

/*
Build the neighbor list.
*/
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

	if(kernel_tmp.Det() > EPSILON)
		data.correct_kernel[localid_i] = kernel_tmp.Inv();
	else
		data.correct_kernel[localid_i] = MooreInv(kernel_tmp);
	
	cmat3& ck = data.correct_kernel[localid_i];
	ck.Set(0.0);
	ck[0][0]=1; ck[1][1]=1; ck[2][2]=1;
	
	data.x0[localid_i] = xi;
	
	//Set rotation matrix R to identitiy matrix.
	cmat3& R = data.rotation[localid_i];
	R.Set(0.0);
	R[0][0] = R[1][1] = R[2][2] = 1;
}


__global__ void AdvectScriptObjectKernel(SimData_SPH data,
	int num_particles,
	cfloat3 vel)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index]!=TYPE_RIGID || data.group[index]!=2)
		return;

	//data.vel[index].Set(0, -0.5, 0);
	data.vel[index] = vel;
	data.pos[index] += data.vel[index] * dParam.dt;

}

__device__ void HourglassControl_Cell(
	int i,
	SimData_SPH& data,
	cfloat3& addition
) {
	float h = dParam.smoothradius * 0.5;
	float vol = dParam.spacing*dParam.spacing*dParam.spacing;
	int localid_i = data.local_id[i];
	cmat3& L = data.correct_kernel[localid_i];
	int* neighborlist = &data.neighborlist[localid_i*NUM_NEIGHBOR];
	int num_neighbor = neighborlist[0];

	for (uint niter = 1; niter <= num_neighbor; niter++)
	{

		/* Neighborlist stores the unique id of neighboring particles.
		We look up the index table to get the current index.
		*/

		int j = data.indexTable[neighborlist[niter]];

		cfloat3 xij = data.pos[i] - data.pos[j];
		cfloat3 xji = xij * (-1);
		cfloat3 x0ij = data.neighbordx[localid_i*NUM_NEIGHBOR + niter];
		cfloat3 x0ji = x0ij * (-1);

		float w = Kernel_Cubic(h, x0ij);

		cfloat3 xij_estimate;
		mvprod(data.gradient[i], x0ij, xij_estimate);
		cfloat3 eij = xij_estimate - xij;
		//printf("err is %f %f %f\n", eij.x, eij.y, eij.z);

		cfloat3 xji_estimate;
		mvprod(data.gradient[j], x0ji, xji_estimate);
		cfloat3 eji = xji_estimate - xji;

		float norm = xij.Norm();
		float dij = eij.dot(xij) / norm;
		float dji = eji.dot(xji) / norm;

		addition += xij / xij.Norm() * w / x0ij.square() * dParam.young * (dij + dji);
	}
}

__global__ void HourglassControl_Kernel(SimData_SPH data, int num_particles)
{
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles)
		return;
	if (data.type[index] != TYPE_DEFORMABLE)
		return;

	float vol = dParam.spacing*dParam.spacing*dParam.spacing;
	cfloat3  addition(0, 0, 0);
	HourglassControl_Cell(
		index,
		data,
		addition);

	addition *= -0.5 * 0.003 * vol * vol;
	
	if(data.gradient[index].Det() < EPSILON)
		data.v_star[index] += addition * dParam.dt / data.mass[index];

}

//IISPH

__device__ void IISPHFactorCell(cint3 gridPos,
	int i,
	cfloat3 xi,
	float& density,
	cfloat3& sum,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos);
	if (gridHash == GRID_UNDEF) return;
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
		cfloat3 nablaw = KernelGradient_Cubic(sr / 2, xij);
		float mass_j;

		switch (data.type[j]) {
		case TYPE_FLUID:
			mass_j = data.mass[i];
			sum += nablaw * mass_j;
			density += Kernel_Cubic(sr / 2, xij)*mass_j;
			break;

		case TYPE_DEFORMABLE:
		case TYPE_CLOTH:
			mass_j = data.mass[i];
			sum += nablaw * mass_j;
			density += Kernel_Cubic(sr / 2, xij)*mass_j;
			break;

		case TYPE_RIGID:
			mass_j = data.restDensity[j] * data.restDensity[i];
			sum += nablaw * mass_j;
			density += Kernel_Cubic(sr / 2, xij)*mass_j;
			break;

		}
	}
}

__global__ void IISPHFactorKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

	if (data.type[index] == TYPE_RIGID)
		return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	float density = 0;
	float mass_i = data.mass[index];
	cfloat3 sum(0, 0, 0);

	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				IISPHFactorCell(nPos,
					index,
					pos,
					density,
					sum,
					data);
			}

	//if (density < data.restDensity[index])
	//	density = data.restDensity[index];

	data.density[index] = density;

	sum *= dParam.dt * dParam.dt / density / density * -1;
	data.dii[index] = sum;
}

__device__ void IISPHPredictDensityCell(cint3 gridPos,
	int i,
	cfloat3 xi,
	float& density,
	float& aii,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos);
	if (gridHash == GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;

	uint endIndex = data.gridCellEnd[gridHash];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		cfloat3 vij = data.v_star[i] - data.v_star[j];
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2) continue;

		float c2 = sr2 - d2;
		cfloat3 nablaw = KernelGradient_Cubic(sr / 2, xij);
		float mass_j;
		cfloat3 dji;
		switch (data.type[j]) {
		case TYPE_FLUID:
			mass_j = data.mass[i];
			density += dParam.dt * mass_j * vij.dot(nablaw);
			dji = KernelGradient_Cubic(sr / 2, xij) * -1 * mass_j / data.density[i] / data.density[i] * dParam.dt * dParam.dt * -1;
			aii += mass_j * (data.dii[i] - dji).dot(nablaw);
			break;

		case TYPE_DEFORMABLE:
		case TYPE_CLOTH:
			mass_j = data.mass[i];
			density += dParam.dt * mass_j * vij.dot(nablaw);
			dji = KernelGradient_Cubic(sr / 2, xij) * -1 * mass_j / data.density[i] / data.density[i] * dParam.dt * dParam.dt * -1;
			aii += mass_j * (data.dii[i] - dji).dot(nablaw);
			break;

		case TYPE_RIGID:
			mass_j = data.restDensity[j] * data.restDensity[i];
			density += dParam.dt * mass_j * data.v_star[i].dot(nablaw);
			dji = KernelGradient_Cubic(sr / 2, xij) * -1 * mass_j / data.density[i] / data.density[i] * dParam.dt * dParam.dt * -1;
			aii += mass_j * (data.dii[i] - dji).dot(nablaw);
			break;

		}
	}
}

__global__ void IISPHPredictDensityKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;
	
	if (data.type[index] == TYPE_RIGID) return;

	data.pressure[index] /= 2;

	float density = data.density[index];

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);

	float aii = 0.0f;

	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				IISPHPredictDensityCell(nPos, index, pos, density, aii,  data);
			}

	data.density_star[index] = density;
	//printf("%f\n", aii);
	data.aii[index] = aii;

}

__device__ void CalcDIJPJLCell(cint3 gridPos,
	int i,
	cfloat3 xi,
	cfloat3& dijpjl,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos);
	if (gridHash == GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;

	uint endIndex = data.gridCellEnd[gridHash];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		cfloat3 xji = xj - xi;
		cfloat3 vij = data.v_star[i] - data.v_star[j];
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2) continue;

		float c2 = sr2 - d2;
		cfloat3 nablaw = KernelGradient_Cubic(sr / 2, xij);
		float mass_j;

		switch (data.type[j]) {
		case TYPE_FLUID:
			mass_j = data.mass[i];
			dijpjl += nablaw * mass_j / data.density[j] / data.density[j] * data.pressure[j] * -1;
			break;

		case TYPE_DEFORMABLE:
		case TYPE_CLOTH:
			mass_j = data.mass[i];
			dijpjl += nablaw * mass_j / data.density[j] / data.density[j] * data.pressure[j] * -1;
			break;

		case TYPE_RIGID:
			break;
		}
	}
}

__global__ void CalcDIJPJLKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

	if (data.type[index] == TYPE_RIGID) return;

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);

	cfloat3 dijpjl(0, 0, 0);

	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				CalcDIJPJLCell(nPos, index, pos, dijpjl, data);
			}
	dijpjl *= dParam.dt * dParam.dt;
	data.dijpjl[index] = dijpjl;
	//printf("%f %f %f\n", dijpjl.x, dijpjl.y, dijpjl.z);
}

__device__ void CalcNewPressureCell(cint3 gridPos,
	int i,
	cfloat3 xi,
	float& longlongthing,
	SimData_SPH data,
	int particleNum)
{
	uint gridHash = calcGridHash(gridPos);
	if (gridHash == GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;

	uint endIndex = data.gridCellEnd[gridHash];

	for (uint j = startIndex; j < endIndex; j++)
	{
		if (j >= particleNum)
			break;
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		cfloat3 vij = data.v_star[i] - data.v_star[j];
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2) continue;

		float c2 = sr2 - d2;
		cfloat3 nablaw = KernelGradient_Cubic(sr / 2, xij);
		float mass_j;
		cfloat3 dji;
		cfloat3 longthing;
		switch (data.type[j]) {
		case TYPE_FLUID:
			mass_j = data.mass[i];
			dji = KernelGradient_Cubic(sr / 2, xij) * -1 * mass_j / data.density[i] / data.density[i] * dParam.dt * dParam.dt * -1;
			longthing = data.dijpjl[i] - data.dii[j] * data.pressure[j] - data.dijpjl[j] + dji * data.pressure[i];
			longlongthing += mass_j * longthing.dot(nablaw);
			break;

		case TYPE_DEFORMABLE:
		case TYPE_CLOTH:
			mass_j = data.mass[i];
			dji = KernelGradient_Cubic(sr / 2, xij) * -1 * mass_j / data.density[i] / data.density[i] * dParam.dt * dParam.dt * -1;
			longthing = data.dijpjl[i] - data.dii[j] * data.pressure[j] - data.dijpjl[j] + dji * data.pressure[i];
			longlongthing += mass_j * longthing.dot(nablaw);
			break;

		case TYPE_RIGID:
			mass_j = data.restDensity[j] * data.restDensity[i];
			longlongthing += mass_j * data.dijpjl[i].dot(nablaw);
			break;

		}
	}
}

__global__ void CalcNewPressureKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

	if (data.type[index] == TYPE_RIGID)
	{
		data.error[index] = 0;
		return;
	}

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);

	float longlongthing = 0.0f;

	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				CalcNewPressureCell(nPos, index, pos, longlongthing, data, num_particles);
			}
	
	float prevPressure = data.pressure[index];
	float density_corr = data.density_star[index] + longlongthing;
	if (abs(data.aii[index]) > 1e-8)
	{
		data.pressure[index] = 0.5 * prevPressure + 0.5 * (data.restDensity[index] - density_corr) / data.aii[index];
		if (data.pressure[index] < 0)
			data.pressure[index] = 0;
	}
	else
		data.pressure[index] = 0;

	density_corr += prevPressure * data.aii[index];
	data.error[index] = density_corr;
}

__device__ void CalcPressureForceCell(cint3 gridPos,
	int i,
	cfloat3 xi,
	cfloat3& pressureForce,
	SimData_SPH data)
{
	uint gridHash = calcGridHash(gridPos);
	if (gridHash == GRID_UNDEF) return;
	uint startIndex = data.gridCellStart[gridHash];
	float sr = dParam.smoothradius;
	float sr2 = sr*sr;
	if (startIndex == 0xffffffff) return;

	uint endIndex = data.gridCellEnd[gridHash];

	for (uint j = startIndex; j < endIndex; j++)
	{
		cfloat3 xj = data.pos[j];
		cfloat3 xij = xi - xj;
		cfloat3 vij = data.v_star[i] - data.v_star[j];
		float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

		if (d2 >= sr2) continue;

		float c2 = sr2 - d2;
		cfloat3 nablaw = KernelGradient_Cubic(sr / 2, xij);
		float mass_j;
		cfloat3 dji;
		cfloat3 longthing;
		switch (data.type[j]) {
		case TYPE_FLUID:
			mass_j = data.mass[i];
			pressureForce += nablaw * -1 * mass_j * mass_j * (data.pressure[i] / data.density[i] / data.density[i] + data.pressure[j] / data.density[j] / data.density[j]);
			break;

		case TYPE_DEFORMABLE:
		case TYPE_CLOTH:
			mass_j = data.mass[i];
			pressureForce += nablaw * -1 * mass_j * mass_j * (data.pressure[i] / data.density[i] / data.density[i] + data.pressure[j] / data.density[j] / data.density[j]);
			break;

		case TYPE_RIGID:
			mass_j = data.restDensity[j] * data.restDensity[i];
			pressureForce += nablaw * -1 * data.mass[i] * mass_j * (data.pressure[i] / data.density[i] / data.density[i]);
			break;

		}
	}
}

__global__ void CalcPressureForceKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

	if (data.type[index] == TYPE_RIGID) return;

	cfloat3 pressureForce(0, 0, 0);

	cfloat3 pos = data.pos[index];
	cint3 gridPos = calcGridPos(pos);
	for (int z = -1; z <= 1; z++)
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++) {
				cint3 nPos = gridPos + cint3(x, y, z);
				CalcPressureForceCell(nPos, index, pos, pressureForce, data);
			}
	data.pressureForce[index] = pressureForce;
}

__global__ void IISPHUpdateKernel(SimData_SPH data, int num_particles) {
	uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	if (index >= num_particles) return;

	if (data.type[index] == TYPE_RIGID) return;

	data.vel[index] = data.v_star[index] + data.pressureForce[index] * dParam.dt / data.mass[index];
	data.pos[index] += data.vel[index] * dParam.dt;

}
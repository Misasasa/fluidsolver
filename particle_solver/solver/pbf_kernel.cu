
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <thrust\device_vector.h>
#include <thrust\sort.h>

#include "pbf_gpu.cuh"


#define GRID_UNDEF 999999999


namespace pbf {


	extern device_buf host_buf;
	extern __device__ device_buf dev_buf;

	
	
	//__global__ void kernel_pairwiseForce_iteration(SimData data, int numParticles) {
	//	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//	if (i >= numParticles)
	//		return;
	//	int gid = dev_buf.grid_index_p[i];
	//	if (gid==GRID_UNDEF)
	//		return;

	//	if (data.type[i]!=TYPE_FLUID)
	//		return;

	//	cint3 offset = getOffset(gid);
	//	int range = dev_buf.range;

	//	cfloat3 xi = data.pos[i];
	//	float restdens = dev_buf.restdensity;
	//	float sr = dev_buf.smoothradius * dev_buf.simscale;
	//	float kh = dev_buf.pairwise_k * sr;
	//	float sr2 = sr*sr;

	//	cfloat3 force(0, 0, 0);

	//	for (int nx= -range; nx<=range; nx++)
	//		for (int ny= -range; ny<=range; ny++)
	//			for (int nz= -range; nz<=range; nz++) {
	//				cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
	//				if (!insidegrid(noffset))
	//					continue;

	//				int nid = cellid(noffset);
	//				for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
	//					int pj = dev_buf.sorted_index_p[j];

	//					if (data.type[pj]!=TYPE_FLUID)
	//						continue;

	//					cfloat3 xj = data.pos[pj];
	//					cfloat3 xij = (xi - xj)*dev_buf.simscale;
	//					float d = xij.mode();

	//					if (d<kh && d>0) {
	//						float fij = cos(3*3.14159/2/kh*d);
	//						float forcefactor = dev_buf.pairwise_c / data.invMass[pj] *fij /d;
	//						force += xij * forcefactor;
	//					}
	//				}
	//			}
	//	//dev_buf.pdata[i].v += force * dev_buf.dt;
	//	data.pos[i] += force * dev_buf.dt * dev_buf.dt;
	//}

	
	/*
	
	__global__ void kernel_absorbfluid() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;

		if(dev_buf.pdata[i].type!=TYPE_CLOTH)
			return;

		int gid = dev_buf.grid_index_p[i];
		
		if (gid==GRID_UNDEF)
			return;

		cfloat3 xi = dev_buf.pdata[i].x;
		float r = dev_buf.collisionDistance * 1.2;
		
		float voli = dev_buf.pdata[i].m / dev_buf.cloth_density;
		float maxsat = dev_buf.max_saturation;
		float sat = (dev_buf.inner_massf[i]+dev_buf.excess_buf[i]) /dev_buf.restdensity / voli;
		
		float absorblimit = dev_buf.cloth_porosity*voli*dev_buf.restdensity;
		float excesslimit = maxsat*voli*dev_buf.restdensity - absorblimit;

		//case saturated
		if(dev_buf.inner_massf[i]+dev_buf.excess_buf[i] > voli*maxsat*dev_buf.restdensity -0.000001)
			return;

		//find neighbors
		cint3 offset = getOffset(gid);
		cint3 noffset;
		int nid;
		int pj;
		int range = dev_buf.range;		

		for (int nx= -range; nx<=range; nx++)
		for (int ny= -range; ny<=range; ny++)
		for (int nz= -range; nz<=range; nz++) {
			noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
			if (!insidegrid(noffset))
				continue;

			nid = cellid(noffset);
			for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
				pj = dev_buf.sorted_index_p[j];
						
				if (dev_buf.pdata[pj].type!=TYPE_FLUID)
					continue;
				if(dev_buf.pdata[pj].group==99)
					continue;

				if(dev_buf.pdata[pj].m<0.000001)
					continue;
						
				cfloat3 xj = dev_buf.pdata[pj].x;
				cfloat3 xij = xi - xj;
				float d = xij.mode();
						
				//absorb
				if (d<r && d>0) {
					float m_abs = dev_buf.k_absorb * (maxsat - sat) * voli * dev_buf.restdensity;
					//partially absorb
					if (m_abs < dev_buf.pdata[pj].m) {
						atomicAdd(&dev_buf.pdata[pj].m, -m_abs);
						dev_buf.inner_massf[i] += m_abs;
					}
					//fully absorb
					else {
						m_abs = dev_buf.pdata[pj].m;
						atomicAdd(&dev_buf.pdata[pj].m, -m_abs);
						atomicAdd(&dev_buf.delete_pnum, 1);
						dev_buf.inner_massf[i] += m_abs;
					}
					
					sat += m_abs /dev_buf.restdensity/ voli;
				}//end if d<0
			}//end cell-neighbor
		}//end loop

		//update inverse mass
		if (dev_buf.pdata[i].invm!=0)
			dev_buf.pdata[i].invm = 1 / (dev_buf.pdata[i].m + dev_buf.pdata[i].waterm + dev_buf.pdata[i].excess_buf);

		//convert buffer
		if (dev_buf.inner_massf[i] > absorblimit - 0.000001) {

			if (dev_buf.inner_massf[i] - absorblimit < excesslimit - dev_buf.excess_buf[i]) {
				dev_buf.excess_buf[i] += dev_buf.inner_massf[i] - absorblimit;
				dev_buf.inner_massf[i] = absorblimit;
			}
			else {
				dev_buf.inner_massf[i] -= excesslimit - dev_buf.excess_buf[i];
				dev_buf.excess_buf[i] = excesslimit;
			}
		}
	}
	

	__global__ void kernel_diffuse_predict() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;

		if(dev_buf.pdata[i].type!=TYPE_CLOTH)
			return;
		
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		//find neighbors
		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float voli = dev_buf.pdata[i].m / dev_buf.cloth_density;
		float sati = dev_buf.inner_massf[i]/dev_buf.restdensity / voli;
		if (sati<0.000001)  //empty
			return;

		float r = dev_buf.diffuseDistance;
		float dsat = 0;
		float dexcess = 0;

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {

						int pj = dev_buf.sorted_index_p[j];

						if(dev_buf.pdata[pj].type!=TYPE_CLOTH)
							continue;

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = xj - xi; // point from i to j
						float d = xij.mode();

						if (d<r && d>0) {
							float dsij=0; // i->j
							float cosij = dot(xij, cfloat3(0, -1, 0))/xij.mode();

							//diffuse
							float volj = dev_buf.pdata[pj].m / dev_buf.cloth_density;
							float satj = dev_buf.inner_massf[pj]/dev_buf.restdensity/volj;
							float exsj = dev_buf.excess_buf[pj]/dev_buf.restdensity/volj;

							
							dsij += dev_buf.k_diffuse*(sati-satj) + sati * dev_buf.k_diffuse_gravity*cosij;
							
							if(dsij < 0.000001 || satj>=dev_buf.cloth_porosity) //only consider i -> j
								dsij = 0;
							dsat += dsij;

							//excess water
							if (cosij>0 && exsj < dev_buf.max_saturation-dev_buf.cloth_porosity) {
								dexcess += dev_buf.excess_buf[i] * cosij * dev_buf.k_excessflow;
							}
						}
					}
				}
		if(dsat > sati && dsat > 0.000001 )
			dev_buf.norm_fac[i] = sati / dsat;
		else
			dev_buf.norm_fac[i] = 1.0;

		if(dexcess > dev_buf.excess_buf[i] && dexcess > 0.000001)
			dev_buf.norm_fac_ef[i] = dev_buf.excess_buf[i] / dexcess;
		else
			dev_buf.norm_fac_ef[i] = 1.0;
	}

	__global__ void kernel_diffuse() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;

		if (dev_buf.pdata[i].type!=TYPE_CLOTH)
			return;

		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		//find neighbors
		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float voli = dev_buf.pdata[i].m / dev_buf.cloth_density;
		float sati = (dev_buf.inner_massf[i]+dev_buf.excess_buf[i])/dev_buf.restdensity / voli;
		
		if (sati<0.000001)
			return;

		float r = dev_buf.diffuseDistance;
		float dsat = 0;
		float dexcess = 0;

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {

						int pj = dev_buf.sorted_index_p[j];

						if (dev_buf.pdata[pj].type!=TYPE_CLOTH)
							continue;

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = xj - xi; // point from i to j
						float d = xij.mode();
						

						if (d<r && d>0) {
							//diffuse
							float cosij = dot(xij, cfloat3(0, -1, 0))/xij.mode();
							float volj = dev_buf.pdata[pj].m / dev_buf.cloth_density;
							float satj = dev_buf.inner_massf[pj]/dev_buf.restdensity/volj;
							float exsj = dev_buf.excess_buf[pj]/dev_buf.restdensity/volj;

							float dsij = dev_buf.k_diffuse*(sati-satj) + sati * dev_buf.k_diffuse_gravity*cosij;
							
							if (dsij>0 && satj<dev_buf.cloth_porosity) //only consider i -> j
							{
								float mass_ij = dsij * voli * dev_buf.restdensity * dev_buf.norm_fac[i];
								atomicAdd(&dev_buf.inner_massf[pj], mass_ij);
								dsat += dsij;
							}

							//excess water
							if (cosij>0 && exsj < dev_buf.max_saturation-dev_buf.cloth_porosity) {
								float dexcess_ij = dev_buf.excess_buf[i] * cosij * dev_buf.k_excessflow*dev_buf.norm_fac_ef[i];
								atomicAdd(&dev_buf.excess_buf[pj], dexcess_ij);
								dexcess += dexcess_ij;
							}
						}
					}
				}
		//lose fluid in pi
		dev_buf.inner_massf[i] -= dsat * voli * dev_buf.restdensity;
		dev_buf.excess_buf[i] -= dexcess;
		
		float absorblimit = voli * dev_buf.cloth_porosity * dev_buf.restdensity;

		if (dev_buf.inner_massf[i] < absorblimit-EPSILON) {
			if (dev_buf.excess_buf[i] < absorblimit - dev_buf.inner_massf[i]) {
				dev_buf.inner_massf[i] += dev_buf.excess_buf[i];
				dev_buf.excess_buf[i] = 0;
			}
			else {
				dev_buf.excess_buf[i] -= (absorblimit - dev_buf.inner_massf[i]);
				dev_buf.inner_massf[i] = absorblimit;
			}
		}
		
		sati = dev_buf.inner_massf[i] / voli;
		float exsat = dev_buf.excess_buf[i]/dev_buf.restdensity/voli;
		float maxsat = dev_buf.cloth_porosity;
		float darken = sati/maxsat;
		darken = (darken<0.5)?darken*2:1;
		dev_buf.pdata[i].color.Set(1-darken, 1-darken, 0.7*(1-darken), 1);

		//set drip flag
		if (dev_buf.excess_buf[i]>dev_buf.emitThres) {
			atomicAdd(&dev_buf.insert_pnum, 1);
		}
	}



	__global__ void kernel_computeVorticity() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		if (dev_buf.pdata[i].type!=TYPE_FLUID)
			return;

		cint3 offset = getOffset(gid);
		int range = dev_buf.range;
		
		float restdens = dev_buf.pdata[i].restden;
		float sr = dev_buf.smoothradius * dev_buf.simscale;
		float sr2 = sr*sr;
		cfloat3 vorticity(0,0,0);

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;
					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];
						
						if (dev_buf.pdata[pj].type!=TYPE_FLUID)
							continue;

						cfloat3 xij = (dev_buf.pdata[i].x - dev_buf.pdata[pj].x)*dev_buf.simscale;
						float d = xij.mode();
						if (d<sr && d>0) {
							float c = sr - d;
							float nablaw = dev_buf.kspikydiff * c * c / d;
							cfloat3 nablawij = xij * nablaw;
							cfloat3 vij = dev_buf.pdata[pj].v - dev_buf.pdata[i].v;
							
							vorticity += cross(vij, nablawij);
						}
							 
					}
				}
		dev_buf.pdata[i].vorticity = vorticity;
	}

	__global__ void kernel_addVorticityConfinement() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;


		if (dev_buf.pdata[i].type!=TYPE_FLUID)
			return;

		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;


		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float restdens = dev_buf.pdata[i].restden;
		float sr = dev_buf.smoothradius * dev_buf.simscale;
		float sr2 = sr*sr;
		
		cfloat3 grad_vorticity(0, 0, 0);

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];

						if (dev_buf.pdata[pj].type!=TYPE_FLUID)
							continue;

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = (xi - xj)*dev_buf.simscale;
						float d = xij.mode();

						if (d<sr && d>0) {
							float c= sr - d;
							float nablaw = dev_buf.kspikydiff * c * c / d * dev_buf.pdata[pj].vorticity.mode();
							grad_vorticity += xij * nablaw;
						}
					}
				}
		float location = grad_vorticity.mode();
		if(location < 0.0001)
			return;


		cfloat3 f_vorticity = cross( grad_vorticity/location, dev_buf.pdata[i].vorticity) * dev_buf.vorticityfactor;
		dev_buf.pdata[i].v += f_vorticity / dev_buf.pdata[i].m * dev_buf.dt;
	}

	



//=======================================
//
//
//         Surface Tension
//
//
//=======================================

	__global__ void kernel_yangtao_model() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		if(dev_buf.pdata[i].type!=TYPE_FLUID)
			return;

		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float restdens = dev_buf.pdata[i].restden;
		float sr = dev_buf.smoothradius * dev_buf.simscale;
		float kh = dev_buf.pairwise_k * sr;
		float sr2 = sr*sr;
		
		cfloat3 force(0, 0, 0);
		
		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];

						if(dev_buf.pdata[pj].type!=TYPE_FLUID)
							continue;

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = (xi - xj)*dev_buf.simscale;
						float d = xij.mode();

						if (d<kh && d>0) {
							float fij = cos(3*3.14159/2/kh*d);
							float forcefactor = dev_buf.pairwise_c * dev_buf.pdata[pj].m *fij /d ;
							force += xij * forcefactor;
						}
					}
				}
		dev_buf.pdata[i].v += force * dev_buf.dt;
	}

	__global__ void kernel_computeNormal() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		if (dev_buf.pdata[i].type!=TYPE_FLUID)
			return;

		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float restdens = dev_buf.pdata[i].restden;
		float sr = dev_buf.smoothradius * dev_buf.simscale;
		float sr2 = sr*sr;
		
		cfloat3 normal(0, 0, 0);

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];
						if (dev_buf.pdata[pj].type!=TYPE_FLUID)
							continue;
						
						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = (xi - xj)*dev_buf.simscale;
						float d = xij.mode();

						if (d<sr && d>0) {
							float c= sr - d;
							float nablaw = dev_buf.kspikydiff * c*c/d;
							normal += xij * (dev_buf.pdata[pj].m / dev_buf.pdata[pj].density * nablaw)*sr;
							//normal += xij * (dev_buf.pdata[pj].m / 1 * nablaw)*sr;

						}
					}
				}
		
		dev_buf.pdata[i].normal = normal;
	}

	__global__ void kernel_computeCurvature() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;

		if (dev_buf.pdata[i].type!=TYPE_FLUID)
			return;
		
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float restdens = dev_buf.pdata[i].restden;
		float sr = dev_buf.smoothradius * dev_buf.simscale;
		float sr2 = sr*sr;
		
		cfloat3 vel(0,0,0);

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];
						if (dev_buf.pdata[pj].type!=TYPE_FLUID)
							continue;

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = (xi - xj)*dev_buf.simscale;
						cfloat3 nij = dev_buf.pdata[i].normal - dev_buf.pdata[pj].normal;
						float d = xij.mode();

						if (d<sr && d>0) {
							float c= sr - d;
							float nablaw = dev_buf.kspikydiff * c*c/d;
							vel += nij * (-1) *dev_buf.surface_stiff * dev_buf.dt;
						}
					}
				}

		dev_buf.pdata[i].v += vel;
	}



//=======================================
//
//
//         Convariance Matrix
//
//
//=======================================


	__global__ void kernel_computeAvgpos() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;

		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		float restdens = dev_buf.pdata[i].restden;
		float sr = dev_buf.smoothradius;
		float sr2 = sr*sr;

		cfloat3 posavg(0, 0, 0);

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = (xi - xj);
						float d2 = xij.x*xij.x + xij.y*xij.y + xij.z*xij.z;

						if (d2 <sr2) {
							float c2 = sr2 - d2;
							float w = dev_buf.kpoly6 * c2*c2*c2;
							posavg += dev_buf.pdata[pj].x * w * dev_buf.pdata[i].m / dev_buf.restdensity;
						}
					}
				}
		dev_buf.avgpos[i] = posavg;
	}


#include "../catpaw/svd3_cuda.h"

	__host__ __device__ __inline__ void svd3(cmat3 input, cmat3& u, cmat3& s, cmat3& v) {
		svd(input[0][0], input[0][1], input[0][2], input[1][0], input[1][1], input[1][2], input[2][0], input[2][1], input[2][2],
			u[0][0], u[0][1], u[0][2], u[1][0], u[1][1], u[1][2], u[2][0], u[2][1], u[2][2],
			s[0][0], s[0][1], s[0][2], s[1][0], s[1][1], s[1][2], s[2][0], s[2][1], s[2][2],
			v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2]
		);
	}

	__global__ void kernel_computeCovmat() {
		int i = blockIdx.x * blockDim.x + threadIdx.x;
		if (i >= dev_buf.pnum)
			return;
		int gid = dev_buf.grid_index_p[i];
		if (gid==GRID_UNDEF)
			return;


		cint3 offset = getOffset(gid);
		int range = dev_buf.range;

		cfloat3 xi = dev_buf.pdata[i].x;
		//cfloat3 xi = dev_buf.avgpos[i];
		float restdens = dev_buf.pdata[i].restden;
		float support = dev_buf.smoothradius * dev_buf.aniso_support;
		
		cmat3 covariance;
		covariance.Set(0.0f);
		float sumw=0;

		for (int nx= -range; nx<=range; nx++)
			for (int ny= -range; ny<=range; ny++)
				for (int nz= -range; nz<=range; nz++) {
					cint3 noffset = cint3(offset.x+nx, offset.y+ny, offset.z+nz);
					if (!insidegrid(noffset))
						continue;

					int nid = cellid(noffset);
					for (int j=dev_buf.offset_g[nid]; j<dev_buf.count_g[nid]; j++) {
						int pj = dev_buf.sorted_index_p[j];

						cfloat3 xj = dev_buf.pdata[pj].x;
						cfloat3 xij = xi - xj;
						float d = xij.mode();

						if (d<support && d>0) {
							float c = d/support;
							float w = 1 - c*c*c;
							cmat3 tmp =  tensor_prod(xij,xij);
							for(int i=0; i<9; i++)
								covariance.data[i] += tmp.data[i]*w;
							sumw += w;
						}
					}
				}

		if (sumw<0.0001) { //a separated particle
			return;
		}

		for (int i=0; i<9; i++) {
			covariance.data[i] /= sumw;
		}
			dev_buf.covmat[i] = covariance;
		cmat3 u,s,v;
		svd3(covariance, u,s,v);
		dev_buf.u[i] = u;
		dev_buf.s[i] = s;
		dev_buf.v[i] = v;

		float ratio1 = s[0][0]/s[2][2];
		float ratio2 = s[1][1]/s[2][2];

		if (ratio1>dev_buf.aniso_thres) {
			dev_buf.pdata[i].color.Set(1,1,1,1);
		}
		else {
			dev_buf.pdata[i].color.Set(1,0,0,1);
		}

		if (ratio2>dev_buf.aniso_thres_coplanar) {
			dev_buf.pdata[i].color.Set(0, 1, 0, 1);
		}

	}

	*/

}
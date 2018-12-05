
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "sph_solver.h"


namespace sph{

extern SimParam_SPH hParam;

void SPHSolver::Copy2Device() {
	num_particles = host_x.size();

	cudaMemcpy(device_data.pos,    host_x.data(),    num_particles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.color,  host_color.data(), num_particles * sizeof(cfloat4), cudaMemcpyHostToDevice);

	cudaMemcpy(device_data.vel,	 host_v.data(),    num_particles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.normal, host_normal.data(), num_particles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.type,   host_type.data(),   num_particles * sizeof(int),     cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.group,  host_group.data(),  num_particles * sizeof(int),     cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.mass,   host_mass.data(),   num_particles * sizeof(float),   cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.uniqueId, host_unique_id.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice);

	int num_particlesT = num_particles * hParam.maxtypenum;
	cudaMemcpy(device_data.vFrac, host_vol_frac.data(), num_particlesT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.restDensity, host_rest_density.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
	CopyParam2Device();
}

void SPHSolver::Copy2Device(int begin, int  end) {
	num_particles = host_x.size();
	int copy_length = end-begin; //end not included

	cudaMemcpy(device_data.pos+begin,		host_x.data()+begin,		copy_length * sizeof(cfloat3),  cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.color+begin,	host_color.data()+begin, copy_length * sizeof(cfloat4), cudaMemcpyHostToDevice);

	cudaMemcpy(device_data.vel+begin,		host_v.data()+begin,		copy_length * sizeof(cfloat3),  cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.normal+begin,	host_normal.data()+begin,	copy_length * sizeof(cfloat3),  cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.type+begin,	host_type.data()+begin,		copy_length * sizeof(int),		cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.group+begin,	host_group.data()+begin,	copy_length * sizeof(int),		cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.mass+begin,	host_mass.data()+begin,		copy_length * sizeof(float),	cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.uniqueId+begin,host_unique_id.data()+begin, copy_length * sizeof(int),		cudaMemcpyHostToDevice);

	int num_particlesT = num_particles * hParam.maxtypenum;
	int copy_length_type = copy_length * hParam.maxtypenum;

	cudaMemcpy(device_data.vFrac + begin*hParam.maxtypenum, 
		host_vol_frac.data() + begin*hParam.maxtypenum, 
		copy_length_type * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.restDensity + begin*hParam.maxtypenum,
		host_rest_density.data() + begin*hParam.maxtypenum,
		copy_length_type * sizeof(float),
		cudaMemcpyHostToDevice);
	CopyParam2Device();
}


void SPHSolver::CopyFromDevice() {
	cudaMemcpy(host_x.data(),		device_data.pos, sizeof(cfloat3)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_color.data(),	device_data.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
}

void SPHSolver::CopyFromDeviceFull() {
	cudaMemcpy(host_x.data(), device_data.pos, sizeof(cfloat3)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_color.data(), device_data.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToHost);

	cudaMemcpy(host_mass.data(),	device_data.mass, sizeof(float)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_v.data(),		device_data.vel, sizeof(cfloat3)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_unique_id.data(), device_data.uniqueId, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_id_table.data(), device_data.indexTable, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	//multiphase
	cudaMemcpy(host_vol_frac.data(), device_data.vFrac, sizeof(float)*num_particles*hParam.maxtypenum, cudaMemcpyDeviceToHost);

}


void SPHSolver::Sort() {
	calcHash(device_data, num_particles);

	sortParticle(device_data, num_particles);

	reorderDataAndFindCellStart(device_data, num_particles, num_grid_cells);
	
	cudaMemcpy(device_data.pos,		device_data.sortedPos,	num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.vel,		device_data.sortedVel,	num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.normal,	device_data.sortedNormal, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.color,		device_data.sortedColor,	num_particles * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.type,		device_data.sortedType,	num_particles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.group,		device_data.sortedGroup,	num_particles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.mass,		device_data.sortedMass,	num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.uniqueId,	device_data.sortedUniqueId, num_particles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.v_star,    device_data.sortedV_star, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.restDensity, device_data.sortedRestDensity, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.vFrac,		device_data.sortedVFrac,	num_particles * hParam.maxtypenum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.effective_mass,   device_data.sorted_effective_mass, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.effective_density, device_data.sorted_effective_density, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);

}

void SPHSolver::SolveSPH() {

	Sort();

	computePressure(device_data, num_particles);

	computeForce(device_data, num_particles);

	advect(device_data, num_particles);

	CopyFromDevice();
}



void SPHSolver::SetupDFSPH() {
	SetupFluidScene();
	Sort();
	computeDensityAlpha(device_data, num_particles);
}

void SPHSolver::SolveDFSPH() {
	//compute non-pressure force
	//predict velocities
	computeNonPForce(device_data, num_particles);

	//correct density
	correctDensityError(device_data, num_particles, 5, 1, false);

	//update neighbors
	Sort();

	//update rho and alpha
	computeDensityAlpha(device_data,num_particles);

	//correct divergence
	//update velocities
	correctDivergenceError(device_data, num_particles, 5, 1, false);

	CopyFromDevice();
}




/*
Still we solve multiphase model with divergence-free SPH.
V_m denotes the volume-averaged velocity of each particle.
*/

void SPHSolver::SetupMultiphaseSPH() {
	SetupFluidScene();
	Sort();
	DFAlpha_Multiphase(device_data, num_particles);
}

void SPHSolver::SolveMultiphaseSPH() {
	
	PhaseDiffusion(device_data, num_particles);
	
	EffectiveMass(device_data, num_particles);

	NonPressureForce_Multiphase(device_data, num_particles);
	
	//correct density + position update
	EnforceDensity_Multiphase(device_data, num_particles, 10, 0.1, false);

	Sort();
	
	DFAlpha_Multiphase(device_data, num_particles);

	//correct divergence + velocity update
	EnforceDivergenceFree_Multiphase(device_data, num_particles, 10, 0.1, true);

	DriftVelocity(device_data, num_particles);

	CopyFromDevice();

}





void SPHSolver::Step() {

	if (num_particles>0) {
		switch (run_mode) {
		case SPH:
			SolveSPH(); break;
		case DFSPH:
			SolveDFSPH(); break;
		case MSPH:
			SolveMultiphaseSPH(); break;
		}
		

	}

	//if(emit_particle_on)
	//	fluidSrcEmit();

	frame_count++;
	time += hParam.dt;
}

void SPHSolver::DumpSimulationDataText() {
	printf("Dumping simulation data text at frame %d\n", frame_count);
	char filepath[1000];
	sprintf(filepath, ".\\dump\\%03d.dat", frame_count);
	
	FILE* fp = fopen(filepath, "w+");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}

	CopyFromDeviceFull();

	// Particle Data
	fprintf(fp, "%d\n", num_particles);
	for (int i=0; i<num_particles; i++) {
		fprintf(fp, "%f %f %f ", host_x[i].x, host_x[i].y, host_x[i].z);
		fprintf(fp, "%f %f %f %f ", host_color[i].x, host_color[i].y, host_color[i].z, host_color[i].w);
		fprintf(fp, "%f %f %f ", host_v[i].x, host_v[i].y, host_v[i].z);
		fprintf(fp, "%d ", host_type[i]);
		fprintf(fp, "%d ", host_group[i]);
		fprintf(fp, "%f ", host_mass[i]);
		fprintf(fp, "%d ", host_unique_id[i]);
	}
	fclose(fp);
}

void SPHSolver::LoadSimulationDataText(char* filepath, cmat4& materialMat) {
	FILE* fp = fopen(filepath, "r");
	if (fp==NULL) {
		printf("error loading simulation data\n");
		return;
	}
}



void SPHSolver::HandleKeyEvent(char key) {
	switch (key) {
	case 'b':
		DumpSimulationDataText();
		break;
	//case 'r':
	//	dumpRenderingData();
	//	break;
	case 'e':
		emit_particle_on = !emit_particle_on;
		if (emit_particle_on)
			printf("Start to emit particles.\n");
		else
			printf("Stop emitting particles.\n");
		break;
	}
}



void SPHSolver::ParseParam(char* xmlpath) {
	printf("Parsing XML.\n");
	tinyxml2::XMLDocument doc;
	int result = doc.LoadFile(xmlpath);
	Tinyxml_Reader reader;

	XMLElement* fluidElement = doc.FirstChildElement("Fluid");
	XMLElement* sceneElement = doc.FirstChildElement("MultiScene");
	XMLElement* boundElement = doc.FirstChildElement("BoundInfo");

	if (!fluidElement || !sceneElement)
	{
		printf("missing fluid/scene xml node");
		return;
	}

	reader.Use(fluidElement);
	hParam.gravity =  reader.GetFloat3("Gravity");
	hParam.gridxmin = reader.GetFloat3("VolMin");
	hParam.gridxmax = reader.GetFloat3("VolMax");
	case_id = reader.GetInt("SceneId");

	
	//find corresponding scene node
	int tmp;
	while (true) {
		sceneElement->QueryIntAttribute("id", &tmp);
		if (tmp == case_id)
			break;
		else if (sceneElement->NextSiblingElement() != NULL) {
			sceneElement = sceneElement->NextSiblingElement();
		}
		else {
			printf("scene not found.\n");
			return;
		}
	}

	reader.Use(sceneElement);

	hParam.maxpnum = reader.GetInt("MaxPNum");
	hParam.maxtypenum = reader.GetInt("TypeNum");

	hParam.dt = reader.GetFloat("DT");
	hParam.spacing = reader.GetFloat("PSpacing");
	float smoothratio = reader.GetFloat("SmoothRatio");
	hParam.smoothradius = hParam.spacing * smoothratio;

	hParam.boundstiff = reader.GetFloat("BoundStiff");
	hParam.bounddamp = reader.GetFloat("BoundDamp");
	hParam.softminx = reader.GetFloat3("SoftBoundMin");
	hParam.softmaxx = reader.GetFloat3("SoftBoundMax");
	hParam.viscosity = reader.GetFloat("Viscosity");
	hParam.restdensity = reader.GetFloat("RestDensity");
	hParam.pressureK = reader.GetFloat("PressureK");
	reader.GetFloatN(hParam.densArr, hParam.maxtypenum, "DensityArray");
	reader.GetFloatN(hParam.viscArr, hParam.maxtypenum, "ViscosityArray");


	loadFluidVolume(sceneElement, hParam.maxtypenum, fluid_volumes);


	//=============================================
	//               Particle Boundary
	//=============================================
	reader.Use(boundElement);
	hParam.bRestdensity = reader.GetFloat("RestDensity");
	hParam.bvisc = reader.GetFloat("Viscosity");
}



void SPHSolver::LoadParam(char* xmlpath) {
	frame_count = 0;
	num_particles = 0;
	num_fluid_particles = 0;

	ParseParam(xmlpath);

	hParam.dx = hParam.smoothradius;
	float sr = hParam.smoothradius;

	//hParam.gridres = cint3(64, 64, 64);
	//hParam.gridxmin.x = 0 - hParam.gridres.x / 2 * hParam.dx;
	//hParam.gridxmin.z = 0 - hParam.gridres.z / 2 * hParam.dx;
	hParam.gridres.x = roundf((hParam.gridxmax.x - hParam.gridxmin.x)/hParam.dx);
	hParam.gridres.y = roundf((hParam.gridxmax.y - hParam.gridxmin.y)/hParam.dx);
	hParam.gridres.z = roundf((hParam.gridxmax.z - hParam.gridxmin.z)/hParam.dx);
	num_grid_cells = hParam.gridres.prod();

	domainMin = hParam.gridxmin;
	domainMax = hParam.gridxmax;
	//setup kernels

	hParam.kpoly6 = 315.0f / (64.0f * 3.141592 * pow(sr, 9.0f));
	hParam.kspiky =  15 / (3.141592 * pow(sr, 6.0f));
	hParam.kspikydiff = -45.0f / (3.141592 * pow(sr, 6.0f));
	hParam.klaplacian = 45.0f / (3.141592 * pow(sr, 6.0f));
	hParam.kspline = 1.0f/3.141593f/pow(sr, 3.0f);

	/*
	for (int k=0; k<hParam.maxtypenum; k++) {
		hParam.densArr[k] = hParam.restdensity * densratio[k];
		hParam.viscArr[k] = hParam.viscosity * viscratio[k];
	}
	*/
	//anisotropic kernels
	
	
	//run_mode = SPH;
	run_mode = DFSPH;
	run_mode = MSPH;
	emit_particle_on = false;
	dt = hParam.dt;
}

void SPHSolver::SetupHostBuffer() {
	int maxNP = hParam.maxpnum;
	/*
	host_x.resize(maxNP);
	host_color.resize(maxNP);
	host_v.resize(maxNP);
	host_type.resize(maxNP);
	host_group.resize(maxNP);
	host_mass.resize(maxNP);
	host_unique_id.resize(maxNP);
	host_id_table.resize(maxNP);
	*/
}

int SPHSolver::AddDefaultParticle() {
	host_x.push_back(cfloat3(0, 0, 0));
	host_color.push_back(cfloat4(1, 1, 1, 1));
	host_normal.push_back(cfloat3(0, 0, 0));
	host_unique_id.push_back(num_particles);
	host_v.push_back(cfloat3(0, 0, 0));
	host_type.push_back(TYPE_FLUID);
	host_mass.push_back(0);
	host_group.push_back(0);
	host_rest_density.push_back(0);

	for(int t=0; t<hParam.maxtypenum; t++)
		host_vol_frac.push_back(0);
	
	return host_x.size()-1;
}


void SPHSolver::Addfluidvolumes() {
	

	for (int i=0; i<fluid_volumes.size(); i++) {
		cfloat3 xmin = fluid_volumes[i].xmin;
		cfloat3 xmax = fluid_volumes[i].xmax;
		int addcount=0;

		//float* vf    = fluid_volumes[i].volfrac;
		float spacing = hParam.spacing;
		float pden = hParam.restdensity;
		float mp   = spacing*spacing*spacing* pden;
		float pvisc = hParam.viscosity;

		for (float x=xmin.x; x<xmax.x; x+=spacing)
			for (float y=xmin.y; y<xmax.y; y+=spacing)
				for (float z=xmin.z; z<xmax.z; z+=spacing) {
					int pid = AddDefaultParticle();
					host_x[pid] = cfloat3(x, y, z);
					host_color[pid]=cfloat4(0.7, 0.75, 0.95, 1);
					host_type[pid] = TYPE_FLUID;
					host_mass[pid] = mp;
					host_group[pid] = 0;
					addcount += 1;
				}

		printf("fluid block No. %d has %d particles.\n", i, addcount);
	}
}

void SPHSolver::AddMultiphaseFluidVolumes() {

	for (int i=0; i<fluid_volumes.size(); i++) {
		cfloat3 xmin = fluid_volumes[i].xmin;
		cfloat3 xmax = fluid_volumes[i].xmax;
		int addcount=0;

		float* vf    = fluid_volumes[i].volfrac;
		float spacing = hParam.spacing;
		float pden = 0; //hParam.restdensity;
		for(int t=0; t<hParam.maxtypenum; t++)
			pden += hParam.densArr[t] * vf[t];

		float mp   = spacing*spacing*spacing* pden;
		float pvisc = hParam.viscosity;

		for (float x=xmin.x; x<xmax.x; x+=spacing)
			for (float y=xmin.y; y<xmax.y; y+=spacing)
				for (float z=xmin.z; z<xmax.z; z+=spacing) {
					int pid = AddDefaultParticle();
					host_x[pid] = cfloat3(x, y, z);
					//host_color[pid]=cfloat4(0.7, 0.75, 0.95, 1);
					host_color[pid]=cfloat4(vf[0], vf[1], vf[2], 1);
					host_type[pid] = TYPE_FLUID;
					host_group[pid] = 0;
					
					host_mass[pid] = mp;
					host_rest_density[pid] = pden;
					for(int t=0;  t<hParam.maxtypenum; t++)
						host_vol_frac[pid*hParam.maxtypenum+t] = vf[t];

					addcount += 1;
				}

		printf("fluid block No. %d has %d particles.\n", i, addcount);
	}
}

void SPHSolver::LoadPO(ParticleObject* po) {
	float spacing = hParam.spacing;
	float pden = hParam.restdensity;
	float mp   = spacing*spacing*spacing* pden*2;

	for (int i=0; i<po->pos.size(); i++) {
		int pid = AddDefaultParticle();
		host_x[pid] = po->pos[i];
		host_color[pid] = cfloat4(1,1,1,0);
		host_type[pid] = TYPE_BOUNDARY;
		host_normal[pid] = po->normal[i];
		host_mass[pid] = mp;
		host_group[pid] = 0;
	}
}

void SPHSolver::SetupFluidScene() {
	LoadParam("config/sph_scene.xml");
	SetupHostBuffer();

	//addfluidvolumes();
	AddMultiphaseFluidVolumes();

	BoundaryGenerator bg;
	ParticleObject* boundary = bg.loadxml("script_object/box.xml");
	LoadPO(boundary);
	delete boundary;

	SetupDeviceBuffer();
	Copy2Device();
}

void SPHSolver::Setup() {
	//SetupFluidScene();

	//SetupDFSPH();

	SetupMultiphaseSPH();
}




void SPHSolver::SetupDeviceBuffer() {

	//particle
	int maxpnum = host_x.size();

	cudaMalloc(&device_data.pos, maxpnum * sizeof(float3));
	cudaMalloc(&device_data.vel, maxpnum * sizeof(float3));
	cudaMalloc(&device_data.color, maxpnum * sizeof(cfloat4));
	cudaMalloc(&device_data.type, maxpnum * sizeof(int));
	cudaMalloc(&device_data.group, maxpnum * sizeof(int));
	cudaMalloc(&device_data.uniqueId, maxpnum * sizeof(int));
	cudaMalloc(&device_data.mass, maxpnum * sizeof(float));
	cudaMalloc(&device_data.density, maxpnum * sizeof(float));
	cudaMalloc(&device_data.pressure, maxpnum * sizeof(float));
	cudaMalloc(&device_data.force, maxpnum*sizeof(cfloat3));
	cudaMalloc(&device_data.normal, maxpnum*sizeof(cfloat3));


	cudaMalloc(&device_data.sortedPos, maxpnum * sizeof(float3));
	cudaMalloc(&device_data.sortedVel, maxpnum * sizeof(float3));
	cudaMalloc(&device_data.sortedColor, maxpnum * sizeof(cfloat4));
	cudaMalloc(&device_data.sortedMass, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sortedType, maxpnum * sizeof(int));
	cudaMalloc(&device_data.sortedGroup, maxpnum * sizeof(int));
	cudaMalloc(&device_data.sortedNormal, maxpnum*sizeof(cfloat3));
	cudaMalloc(&device_data.sortedUniqueId, maxpnum * sizeof(int));
	cudaMalloc(&device_data.indexTable, maxpnum * sizeof(int));
	
	//DFSPH
	cudaMalloc(&device_data.alpha, maxpnum * sizeof(float));
	cudaMalloc(&device_data.v_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.x_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.pstiff, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sortedV_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.error, maxpnum * sizeof(float));
	cudaMalloc(&device_data.pstiff_sum, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sortedPstiff_sum, maxpnum * sizeof(float));

	//Multiphase
	int ptnum = maxpnum*hParam.maxtypenum;
	cudaMalloc(&device_data.vFrac,		ptnum*sizeof(float));
	cudaMalloc(&device_data.restDensity,	maxpnum*sizeof(float));
	cudaMalloc(&device_data.driftV,		ptnum*sizeof(cfloat3));
	cudaMalloc(&device_data.sortedVFrac,	ptnum*sizeof(float));
	cudaMalloc(&device_data.sortedRestDensity,	maxpnum*sizeof(float));
	cudaMalloc(&device_data.effective_mass, maxpnum*sizeof(float));
	cudaMalloc(&device_data.sorted_effective_mass, maxpnum*sizeof(float));
	cudaMalloc(&device_data.effective_density, maxpnum*sizeof(float));
	cudaMalloc(&device_data.sorted_effective_density, maxpnum*sizeof(float));

	int glen = hParam.gridres.prod();

	cudaMalloc(&device_data.particleHash, maxpnum * sizeof(int));
	cudaMalloc(&device_data.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&device_data.gridCellStart, glen * sizeof(int));
	cudaMalloc(&device_data.gridCellEnd, glen * sizeof(int));
}

};
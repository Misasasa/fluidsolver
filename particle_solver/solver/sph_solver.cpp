
#include "cuda.h"
#include "cuda_runtime.h"

#include "sph_solver.h"


extern SimParam_SPH hParam;

void SPHSolver::CopyParticleDataToDevice() {
	numParticles = pos.size();
	//Single phase properties.
	cudaMemcpy(device_data.pos,    pos.data(),    numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.color,  color.data(), numParticles * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.vel,	 vel.data(),    numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.normal, normal.data(), numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.type,   type.data(),   numParticles * sizeof(int),     cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.uniqueId, unique_id.data(), numParticles * sizeof(int), cudaMemcpyHostToDevice);
}

void SPHSolver::CopyParticleDataToDevice(int begin, int  end) {
	numParticles = pos.size();
	int copy_length = end-begin; //end not included

	cudaMemcpy(device_data.pos+begin, pos.data()+begin, copy_length * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.color+begin,	color.data()+begin, copy_length * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.vel+begin, vel.data()+begin, copy_length * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.normal+begin, normal.data()+begin, copy_length * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.type+begin, type.data()+begin, copy_length * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.uniqueId+begin, unique_id.data()+begin, copy_length * sizeof(int), cudaMemcpyHostToDevice);

}


void SPHSolver::CopyPosColorFromDevice() {
	cudaMemcpy(pos.data(),		device_data.pos, numParticles * sizeof(cfloat3), cudaMemcpyDeviceToHost);
	cudaMemcpy(color.data(),	device_data.color, numParticles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
}

void SPHSolver::CopySimulationDataFromDevice() {
	cudaMemcpy(pos.data(), device_data.pos, sizeof(cfloat3)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(color.data(), device_data.color, numParticles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
	cudaMemcpy(vel.data(),		device_data.vel, sizeof(cfloat3)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(type.data(), device_data.type, sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(unique_id.data(), device_data.uniqueId, sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(id_table.data(), device_data.indexTable, sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
}


void SPHSolver::SortParticles() {
	calcHash(device_data, numParticles);

	sortParticle(device_data, numParticles);

	reorderDataAndFindCellStart(device_data, numParticles, num_grid_cells);
	
	//Single Phase
	cudaMemcpy(device_data.pos,		device_data.sortedPos,	numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.vel,		device_data.sortedVel,	numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.normal,	device_data.sortedNormal, numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.color,		device_data.sortedColor,	numParticles * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.type,		device_data.sortedType,	numParticles * sizeof(int), cudaMemcpyDeviceToDevice);
	
	//Multi Phase
	cudaMemcpy(device_data.group,		device_data.sortedGroup,	numParticles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.mass,		device_data.sortedMass,	numParticles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.uniqueId,	device_data.sortedUniqueId, numParticles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.v_star,    device_data.sortedV_star, numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.restDensity, device_data.sortedRestDensity, numParticles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.vFrac,		device_data.sortedVFrac,	numParticles * hParam.maxtypenum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.effective_mass,   device_data.sorted_effective_mass, numParticles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.rho_stiff, device_data.sorted_rho_stiff, numParticles*sizeof(float), cudaMemcpyDeviceToDevice );
	cudaMemcpy(device_data.div_stiff, device_data.sorted_div_stiff, numParticles*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.temperature, device_data.sorted_temperature, numParticles*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.heat_buffer, device_data.sorted_heat_buffer, numParticles*sizeof(float), cudaMemcpyDeviceToDevice);

	//Deformable Solid
	cudaMemcpy(device_data.cauchy_stress, device_data.sorted_cauchy_stress, numParticles*sizeof(cmat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.gradient, device_data.sorted_gradient, numParticles * sizeof(cmat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.local_id, device_data.sorted_local_id, numParticles*sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.adjacent_index, device_data.sorted_adjacent_index, numParticles * sizeof(adjacent), cudaMemcpyDeviceToDevice);

	//IISPH
	cudaMemcpy(device_data.pressure, device_data.sorted_pressure, numParticles * sizeof(float), cudaMemcpyDeviceToDevice);
}

void SPHSolver::SolveSPH() {

	SortParticles();

	ComputePressure(device_data, numParticles);

	computeForce(device_data, numParticles);

	Advect(device_data, numParticles);

	CopyPosColorFromDevice();
}



void SPHSolver::SetupDFSPH() {
	SetupFluidScene();
	SortParticles();
	computeDensityAlpha(device_data, numParticles);
}

void SPHSolver::SolveDFSPH() {
	
	computeNonPForce(device_data, numParticles);

	correctDensityError(device_data, numParticles, 5, 1, false);
	
	SortParticles();
	computeDensityAlpha(device_data,numParticles);

	correctDivergenceError(device_data, numParticles, 5, 1, false);

	CopyPosColorFromDevice();
}

void SPHSolver::Eval(const char* expression) {
	if (strcmp(expression, "DumpRenderData")==0) {
		DumpRenderData();
	}
}


void SPHSolver::Step() {

	if (numParticles>0) {
		switch (run_mode) {
		case SPH:
			SolveSPH(); break;
		case DFSPH:
			SolveDFSPH(); break;
		}
	}

	frame_count++;
	time += hParam.dt;
}

void SPHSolver::DumpSimulationDataText() {
	printf("Dumping simulation data at frame %d\n", frame_count);
	char filepath[1000];
	sprintf(filepath, ".\\dump\\%03d.txt", frame_count);
	
	FILE* fp = fopen(filepath, "w+");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}

	CopySimulationDataFromDevice();

	// Particle Data
	fprintf(fp, "%d\n", numParticles);
	for (int i=0; i<numParticles; i++) {
		fprintf(fp, "%f %f %f ", pos[i].x, pos[i].y, pos[i].z);
		fprintf(fp, "%f %f %f ", vel[i].x, vel[i].y, vel[i].z);
		fprintf(fp, "%d ", type[i]);
		fprintf(fp, "%d ", unique_id[i]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void SPHSolver::DumpRenderData() {
	
	char filepath[1000];
	sprintf(filepath, "..\\particle_data\\%03d.txt", dump_count++);

	FILE* fp = fopen(filepath, "w+");
	if (fp == NULL) {
		printf("error opening file\n"); 
		return;
	}

	CopySimulationDataFromDevice();

	
	fprintf(fp, "frame %d\n", frame_count);
	int output_count=0;
	
	for (int i=0; i<numParticles; i++) {
		fprintf(fp, "%d %f %f %f ", output_count++, pos[i].x, pos[i].y, pos[i].z);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void SPHSolver::LoadSimulationDataText(char* filepath, cmat4& materialMat) {
	
	printf("Loading simulation data text from %s", filepath);	
	FILE* fp = fopen(filepath, "r");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}
	// Particle Data
	fscanf(fp, "%d\n", &numParticles);
	for (int pi=0; pi<numParticles; pi++) {

		int i = AddDefaultParticle();

		fscanf(fp, "%f %f %f ", &pos[i].x, &pos[i].y, &pos[i].z);
		fscanf(fp, "%f %f %f ", &vel[i].x, &vel[i].y, &vel[i].z);
		fscanf(fp, "%d ", &type[i]);
		fscanf(fp, "%d ", &unique_id[i]);
		fscanf(fp,"\n");
	}
	fclose(fp);
}



void SPHSolver::HandleKeyEvent(char key) {
	switch (key) {
	case 'b':
		DumpSimulationDataText();
		break;
	case 'r':
		DumpRenderData();
		break;
	case 'e':
		emit_particle_on = !emit_particle_on;
		if (emit_particle_on)
			printf("Start to emit particles.\n");
		else
			printf("Stop emitting particles.\n");
		break;
	
	case 'm':
		MoveConstraintBoxAway(device_data, numParticles);
		RigidParticleVolume(device_data, numParticles);
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

	hParam.viscosity = reader.GetFloat("Viscosity");
	hParam.restdensity = reader.GetFloat("RestDensity");
	hParam.pressureK = reader.GetFloat("PressureK");
	reader.GetFloatN(hParam.densArr, hParam.maxtypenum, "DensityArray");
	reader.GetFloatN(hParam.viscArr, hParam.maxtypenum, "ViscosityArray");
	
	//Multiphase
	hParam.drift_dynamic_diffusion = reader.GetFloat("DriftDynamicDiffusion");
	hParam.drift_turbulent_diffusion = reader.GetFloat("DriftTurbulentDiffusion");
	hParam.drift_thermal_diffusion = reader.GetFloat("DriftThermalDiffusion");
	hParam.surface_tension = reader.GetFloat("SurfaceTension");
	hParam.acceleration_limit = reader.GetFloat("AccelerationLimit");

	/* Solid parameters. 
	E: Young's modulus
	v: Poisson's ratio
	G: shear modulus
	K: bulk modulus
	*/

	float E = reader.GetFloat("YoungsModulus");
	float v = reader.GetFloat("PoissonsRatio");
	hParam.young = E;
	hParam.Yield = reader.GetFloat("Yield");
	hParam.plastic_flow = reader.GetFloat("PlasticFlow");
	hParam.solidG = E/2/(1+v);
	hParam.solidK = E/3/(1-2*v);
	hParam.solid_visc = reader.GetFloat("SolidViscosity");
	hParam.dissolution = reader.GetFloat("Dissolution");
	printf("G,K,solidvisc: %f %f %f\n", hParam.solidG, hParam.solidK, hParam.solid_visc);
	reader.GetFloatN(hParam.max_alpha, hParam.maxtypenum, "MaxVFraction");
	hParam.heat_flow_rate = reader.GetFloat("HeatFlowRate");
	reader.GetFloatN(hParam.heat_capacity, hParam.maxtypenum, "HeatCapacity");

	loadFluidVolume(sceneElement, hParam.maxtypenum, fluid_volumes);


	//=============================================
	//               Particle Boundary
	//=============================================
	reader.Use(boundElement);
	hParam.boundary_visc = reader.GetFloat("Viscosity");
	hParam.boundary_friction = reader.GetFloat("Friction");
}



void SPHSolver::LoadParam(char* xmlpath) {

	//Parameter initialization.
	frame_count = 0;
	dump_count = 0;
	numParticles = 0;
	num_fluid_particles = 0;
	hParam.num_fluid_p = 0;
	for(int k=0; k<100; k++) 
		localid_counter[k]=0;
	
	emit_particle_on = false;
	advect_scriptobject_on = true;
	hParam.enable_dissolution = false;


	ParseParam(xmlpath);

	hParam.dx = hParam.smoothradius;
	hParam.gridres.x = roundf((hParam.gridxmax.x - hParam.gridxmin.x)/hParam.dx);
	hParam.gridres.y = roundf((hParam.gridxmax.y - hParam.gridxmin.y)/hParam.dx);
	hParam.gridres.z = roundf((hParam.gridxmax.z - hParam.gridxmin.z)/hParam.dx);
	num_grid_cells = hParam.gridres.prod();

	domainMin = hParam.gridxmin;
	domainMax = hParam.gridxmax;
	dt = hParam.dt;

	//Precompute kernel constant factors.
	float sr = hParam.smoothradius;
	hParam.kpoly6 = 315.0f / (64.0f * 3.141592 * pow(sr, 9.0f));
	hParam.kspiky =  15 / (3.141592 * pow(sr, 6.0f));
	hParam.kspikydiff = -45.0f / (3.141592 * pow(sr, 6.0f));
	hParam.klaplacian = 45.0f / (3.141592 * pow(sr, 6.0f));
	hParam.kernel_cubic = 1/3.141593/pow(sr/2,3);
	hParam.kernel_cubic_gradient = 1.5/3.141593f/pow(sr/2, 4);


	hParam.melt_point = 60;
	hParam.latent_heat = 10;
}

void SPHSolver::SetupHostBuffer() {
	int maxNP = hParam.maxpnum;
}


int SPHSolver::AddDefaultParticle() {

	//Single-phase properties.
	pos.push_back(cfloat3(0, 0, 0));
	color.push_back(cfloat4(1, 1, 1, 1));
	normal.push_back(cfloat3(0, 0, 0));
	unique_id.push_back(pos.size()-1);
	vel.push_back(cfloat3(0, 0, 0));
	type.push_back(TYPE_FLUID);
	return pos.size()-1;
}

void SPHSolver::LoadPO(ParticleObject* po) {
	float spacing = hParam.spacing;
	float pden = hParam.restdensity;
	float mp   = spacing*spacing*spacing* pden*2;

	for (int i=0; i<po->pos.size(); i++) {
		int pid = AddDefaultParticle();
		pos[pid] = po->pos[i];
		if(po->id[i]==2)
			color[pid] = cfloat4(1,1,1,0.5);
		else
			color[pid] = cfloat4(1, 1, 1, 0.0);

		type[pid] = TYPE_RIGID;
		normal[pid] = po->normal[i];
	}
}

void SPHSolver::LoadPO(ParticleObject* po, int objectType)
{
	
	float vf[3] = {1,0,0};
	int addcount = 0;
	 
	float spacing = hParam.spacing;
	float density = 0;
	for (int t=0; t<hParam.maxtypenum; t++)
		density += hParam.densArr[t] * vf[t];
	
	float mp   = spacing*spacing*spacing* density;

	float relax = 1.65;
	for (int i=0; i<po->pos.size(); i++) {
		int pid = AddDefaultParticle();
		pos[pid] = po->pos[i];
		pos[pid] *= 0.005 * relax;
		pos[pid].y += 0.16;

		type[pid] = objectType;
		addcount ++;
	}

	if (objectType==TYPE_FLUID) {
		printf("PO type: fluid, particle num: %d\n", addcount);
		hParam.num_fluid_p += addcount;
	}
	else if (objectType==TYPE_DEFORMABLE) {
		printf("PO type: deformable, particle num: %d\n", addcount);
		hParam.num_deformable_p += addcount;
	}
	else if (objectType==TYPE_GRANULAR) {
		printf("PO type: granular, particle num: %d\n", addcount);
	}
}

void SPHSolver::SetupFluidScene() {
	LoadParam("config/sph_scene.xml");
	SetupHostBuffer();

	BoundaryGenerator bg;
	ParticleObject* boundary = bg.loadxml("config/big box.xml");
	LoadPO(boundary);
	delete boundary;

	ObjParser objparser;
	ParticleObject* duck = objparser.loadPoints("config/ducky.obj");
	printf("duck has %d points\n", duck->pos.size());
	LoadPO(duck, TYPE_DEFORMABLE);
	delete duck;

	SetupDeviceBuffer();
	CopyParticleDataToDevice();
	CopyParam2Device();
}


void SPHSolver::Setup() {
	
	//SetupFluidScene(); run_mode = SPH;

	SetupDFSPH(); run_mode = DFSPH;
}




void SPHSolver::SetupDeviceBuffer() {

	//particle
	int maxpnum = pos.size();
	id_table.resize(maxpnum);

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
	cudaMalloc(&device_data.DF_factor, maxpnum * sizeof(float));
	cudaMalloc(&device_data.v_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.pstiff, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sortedV_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.error, maxpnum * sizeof(float));
	cudaMalloc(&device_data.rho_stiff, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sorted_rho_stiff, maxpnum * sizeof(float));
	cudaMalloc(&device_data.div_stiff, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sorted_div_stiff, maxpnum * sizeof(float));

	//Multiphase
	int num_pt = maxpnum*hParam.maxtypenum;
	cudaMalloc(&device_data.vFrac, num_pt*sizeof(float));
	cudaMalloc(&device_data.restDensity,	maxpnum*sizeof(float));
	cudaMalloc(&device_data.drift_v, num_pt*sizeof(cfloat3));
	cudaMalloc(&device_data.vol_frac_gradient, num_pt*sizeof(cfloat3));
	cudaMalloc(&device_data.effective_mass, maxpnum*sizeof(float));
	cudaMalloc(&device_data.phase_diffusion_lambda, maxpnum*sizeof(float));
	cudaMalloc(&device_data.vol_frac_change, num_pt*sizeof(float));
	cudaMalloc(&device_data.spatial_status, maxpnum*sizeof(float));
	cudaMalloc(&device_data.temperature, maxpnum*sizeof(float));
	cudaMalloc(&device_data.heat_buffer, maxpnum*sizeof(float));

	cudaMalloc(&device_data.flux_buffer, num_pt*sizeof(cfloat3));
	
	cudaMalloc(&device_data.sortedVFrac, num_pt*sizeof(float));
	cudaMalloc(&device_data.sortedRestDensity,	maxpnum*sizeof(float));
	cudaMalloc(&device_data.sorted_effective_mass, maxpnum*sizeof(float));
	cudaMalloc(&device_data.sorted_temperature, maxpnum*sizeof(float));
	cudaMalloc(&device_data.sorted_heat_buffer, maxpnum*sizeof(float));

	// Deformable Solid
	cudaMalloc(&device_data.strain_rate, maxpnum*sizeof(cmat3));
	cudaMalloc(&device_data.cauchy_stress, maxpnum*sizeof(cmat3));
	cudaMalloc(&device_data.gradient, maxpnum * sizeof(cmat3));
	cudaMalloc(&device_data.vel_right, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.local_id, maxpnum*sizeof(int));
	cudaMalloc(&device_data.neighborlist, hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(int));
	cudaMalloc(&device_data.neighbordx, hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(cfloat3));
	cudaMalloc(&device_data.correct_kernel, hParam.num_deformable_p*sizeof(cmat3));
	cudaMalloc(&device_data.x0, hParam.num_deformable_p*sizeof(cfloat3));
	cudaMalloc(&device_data.rotation, hParam.num_deformable_p*sizeof(cmat3));
	cudaMalloc(&device_data.trim_tag, hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(int));
	cudaMalloc(&device_data.length0,  hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(float));
	cudaMalloc(&device_data.spatial_color, maxpnum*sizeof(float));

	cudaMalloc(&device_data.sorted_cauchy_stress, maxpnum*sizeof(cmat3));
	cudaMalloc(&device_data.sorted_gradient, maxpnum * sizeof(cmat3));
	cudaMalloc(&device_data.sorted_local_id, maxpnum*sizeof(int));
	
	cudaMalloc(&device_data.adjacent_index, maxpnum * sizeof(adjacent));
	cudaMalloc(&device_data.sorted_adjacent_index, maxpnum * sizeof(adjacent));

	int glen = hParam.gridres.prod();

	cudaMalloc(&device_data.particleHash, maxpnum * sizeof(int));
	cudaMalloc(&device_data.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&device_data.gridCellStart, glen * sizeof(int));
	cudaMalloc(&device_data.gridCellEnd, glen * sizeof(int));

	//IISPH
	cudaMalloc(&device_data.dii, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.dijpjl, maxpnum * sizeof(cfloat3));
	cudaMalloc(&device_data.aii, maxpnum * sizeof(float));
	cudaMalloc(&device_data.density_star, maxpnum * sizeof(float));
	cudaMalloc(&device_data.pressure, maxpnum * sizeof(float));
	cudaMalloc(&device_data.sorted_pressure, maxpnum * sizeof(float));
}
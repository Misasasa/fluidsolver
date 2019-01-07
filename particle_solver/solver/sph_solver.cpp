
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "sph_solver.h"


namespace sph{

extern SimParam_SPH hParam;

void SPHSolver::Copy2Device() {
	num_particles = host_x.size();

	//Single phase properties.
	cudaMemcpy(device_data.pos,    host_x.data(),    num_particles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.color,  host_color.data(), num_particles * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.vel,	 host_v.data(),    num_particles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.normal, host_normal.data(), num_particles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.type,   host_type.data(),   num_particles * sizeof(int),     cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.uniqueId, host_unique_id.data(), num_particles * sizeof(int), cudaMemcpyHostToDevice);


	//Multi phase properties.
	cudaMemcpy(device_data.group,  host_group.data(),  num_particles * sizeof(int),     cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.mass,   host_mass.data(),   num_particles * sizeof(float),   cudaMemcpyHostToDevice);
	int num_particlesT = num_particles * hParam.maxtypenum;
	cudaMemcpy(device_data.vFrac, host_vol_frac.data(), num_particlesT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_data.restDensity, host_rest_density.data(), num_particles * sizeof(float), cudaMemcpyHostToDevice);
	
	//Deformable Solid properties.
	cudaMemcpy(device_data.local_id, host_localid.data(),num_particles*sizeof(int), cudaMemcpyHostToDevice);

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



	// multiphase data
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
	cudaMemcpy(host_x.data(),		device_data.pos, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToHost);
	cudaMemcpy(host_color.data(),	device_data.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
}

void SPHSolver::CopyFromDeviceFull() {
	cudaMemcpy(host_x.data(), device_data.pos, sizeof(cfloat3)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_color.data(), device_data.color, num_particles * sizeof(cfloat4), cudaMemcpyDeviceToHost);
	

	cudaMemcpy(host_v.data(),		device_data.vel, sizeof(cfloat3)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_type.data(), device_data.type, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_group.data(), device_data.group, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_mass.data(),	device_data.mass, sizeof(float)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_unique_id.data(), device_data.uniqueId, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_id_table.data(), device_data.indexTable, sizeof(int)*num_particles, cudaMemcpyDeviceToHost);
	
	cudaMemcpy(host_vol_frac.data(), device_data.vFrac, sizeof(float)*num_particles*hParam.maxtypenum, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_v_star.data(), device_data.v_star, sizeof(cfloat3)*num_particles, cudaMemcpyDeviceToHost);
}


void SPHSolver::Sort() {
	calcHash(device_data, num_particles);

	sortParticle(device_data, num_particles);

	reorderDataAndFindCellStart(device_data, num_particles, num_grid_cells);
	
	//Single Phase
	cudaMemcpy(device_data.pos,		device_data.sortedPos,	num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.vel,		device_data.sortedVel,	num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.normal,	device_data.sortedNormal, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.color,		device_data.sortedColor,	num_particles * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.type,		device_data.sortedType,	num_particles * sizeof(int), cudaMemcpyDeviceToDevice);
	
	//Multi Phase
	cudaMemcpy(device_data.group,		device_data.sortedGroup,	num_particles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.mass,		device_data.sortedMass,	num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.uniqueId,	device_data.sortedUniqueId, num_particles * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.v_star,    device_data.sortedV_star, num_particles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.restDensity, device_data.sortedRestDensity, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.vFrac,		device_data.sortedVFrac,	num_particles * hParam.maxtypenum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.effective_mass,   device_data.sorted_effective_mass, num_particles * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.rho_stiff, device_data.sorted_rho_stiff, num_particles*sizeof(float), cudaMemcpyDeviceToDevice );
	cudaMemcpy(device_data.div_stiff, device_data.sorted_div_stiff, num_particles*sizeof(float), cudaMemcpyDeviceToDevice);
	
	//Deformable Solid
	cudaMemcpy(device_data.cauchy_stress, device_data.sorted_cauchy_stress, num_particles*sizeof(cmat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(device_data.local_id, device_data.sorted_local_id, num_particles*sizeof(int), cudaMemcpyDeviceToDevice);
}

void SPHSolver::SolveSPH() {

	Sort();

	ComputePressure(device_data, num_particles);

	computeForce(device_data, num_particles);

	Advect(device_data, num_particles);

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
	//computeNonPForce(device_data, num_particles);

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






void SPHSolver::PhaseDiffusion_Host() {

	DriftVelocity(device_data, num_particles);

	PhaseDiffusion(device_data, num_particles, NULL, frame_count);
	//PhaseDiffusion(device_data, num_particles);
	
	EffectiveMass(device_data, num_particles);

	DFSPHFactor_Multiphase(device_data, num_particles);

}



void SPHSolver::SetupMultiphaseSPH() {
	SetupFluidScene();
	run_mode = MSPH;

	Sort();
	EffectiveMass(device_data, num_particles);
	DFSPHFactor_Multiphase(device_data, num_particles);
	RigidParticleVolume(device_data, num_particles);
	InitializeDeformable(device_data, num_particles);
}

void SPHSolver::SolveMultiphaseSPH() {
	
	
	catpaw::cTime clock; clock.tick();
	
	//DetectDispersedParticles(device_data, num_particles);
	NonPressureForce_Multiphase(device_data, num_particles);
	//printf("predict %f\n",clock.tack()*1000); clock.tick();
	
	//compute tension in solids
	ComputeTension(device_data, num_particles);

	//clock.tick();
	EnforceDensity_Multiphase(device_data, num_particles, 30, 0.5, 2, false,  true);
	//clock.tack("density solve");
	
	if (advect_scriptobject_on) {
		
		if(frame_count<524)
			AdvectScriptObject(device_data, num_particles, cfloat3(0,-0.5,0));
		else
			AdvectScriptObject(device_data, num_particles, cfloat3(0, 0.5, 0));
		//RigidParticleVolume(device_data, num_particles);
	}
		

	Sort();
	
	DFSPHFactor_Multiphase(device_data, num_particles);
	

	//clock.tick();
	EnforceDivergenceFree_Multiphase(device_data, num_particles, 5, 0.5, false, true);
	//clock.tack("divergence solve"); clock.tick();


	//update deformation gradient F
	UpdateSolidState(device_data, num_particles, VON_MISES);
	
	UpdateSolidTopology(device_data, num_particles);

	PhaseDiffusion_Host();


	CopyFromDevice();
}

void SPHSolver::SolveMultiphaseSPHRen() {
	
	ComputePressure(device_data, num_particles);
	ComputeForceMultiphase(device_data, num_particles);
	Advect(device_data, num_particles);
	
	Sort();
	
	ComputePressure(device_data, num_particles);
	DriftVel_Ren(device_data, num_particles);
	PhaseDiffusion_Ren(device_data, num_particles);
	EffectiveMass(device_data, num_particles);
	
	CopyFromDevice();
}



void SPHSolver::Eval(const char* expression) {
	if (strcmp(expression, "DumpRenderData")==0) {
		DumpRenderData();
	}
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
		case MSPH_REN:
			SolveMultiphaseSPHRen(); break;
		}
		

	}

	//if(emit_particle_on)
	//	fluidSrcEmit();


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

	CopyFromDeviceFull();

	// Particle Data
	fprintf(fp, "%d\n", num_particles);
	for (int i=0; i<num_particles; i++) {
		fprintf(fp, "%f %f %f ", host_x[i].x, host_x[i].y, host_x[i].z);
		fprintf(fp, "%f %f %f ", host_v[i].x, host_v[i].y, host_v[i].z);
		fprintf(fp, "%d ", host_type[i]);
		fprintf(fp, "%d ", host_group[i]);
		fprintf(fp, "%f ", host_mass[i]);
		fprintf(fp, "%d ", host_unique_id[i]);
		fprintf(fp, "%f %f %f ",host_v_star[i].x, host_v_star[i].y, host_v_star[i].z);
		
		for(int k=0;k<hParam.maxtypenum;k++)
			fprintf(fp, "%f ", host_vol_frac[i*hParam.maxtypenum+k]);
		fprintf(fp, "\n");
	}
	fclose(fp);
}

void SPHSolver::DumpRenderData() {
	//printf("Dumping rendering data at frame %d\n", frame_count);
	char filepath[1000];
	sprintf(filepath, "..\\particle_data\\%03d.txt", dump_count++);

	FILE* fp = fopen(filepath, "w+");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}

	CopyFromDeviceFull();

	// Particle Data
	fprintf(fp, "frame %d\n", frame_count);
	int output_count=0;
	//fprintf(fp, "%d\n", num_particles);
	for (int i=0; i<num_particles; i++) {

		if(host_type[i]!=TYPE_FLUID)
			continue;

		fprintf(fp, "%d %f %f %f ", output_count++, host_x[i].x, host_x[i].y, host_x[i].z);
		//fprintf(fp, "%f %f %f ",host_color[i].x, host_color[i].y, host_color[i].z);
		fprintf(fp, "%f %f %f ", host_vol_frac[i*hParam.maxtypenum], host_vol_frac[i*hParam.maxtypenum+1], host_vol_frac[i*hParam.maxtypenum+2]);
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
	fscanf(fp, "%d\n", &num_particles);
	for (int pi=0; pi<num_particles; pi++) {

		int i = AddDefaultParticle();

		fscanf(fp, "%f %f %f ", &host_x[i].x, &host_x[i].y, &host_x[i].z);
		fscanf(fp, "%f %f %f ", &host_v[i].x, &host_v[i].y, &host_v[i].z);
		fscanf(fp, "%d ", &host_type[i]);
		fscanf(fp, "%d ", &host_group[i]);
		fscanf(fp, "%f ", &host_mass[i]);
		fscanf(fp, "%d ", &host_unique_id[i]);
		fscanf(fp, "%f %f %f ", &host_v_star[i].x, &host_v_star[i].y, &host_v_star[i].z);

		for (int k=0; k<hParam.maxtypenum; k++)
			fscanf(fp, "%f ", &host_vol_frac[i*hParam.maxtypenum+k]);
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
		MoveConstraintBoxAway(device_data, num_particles);
		RigidParticleVolume(device_data, num_particles);
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
	hParam.Yield = reader.GetFloat("Yield");
	hParam.plastic_flow = reader.GetFloat("PlasticFlow");
	hParam.solidG = E/2/(1+v);
	hParam.solidK = E/3/(1-2*v);
	hParam.solid_visc = reader.GetFloat("SolidViscosity");
	hParam.dissolution = reader.GetFloat("Dissolution");
	printf("G,K,solidvisc: %f %f %f\n", hParam.solidG, hParam.solidK, hParam.solid_visc);
	reader.GetFloatN(hParam.max_alpha, hParam.maxtypenum, "MaxVFraction");
	/*hParam.max_alpha[0] = 1;
	hParam.max_alpha[1] = 1;
	hParam.max_alpha[2] = 1;*/

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
	num_particles = 0;
	num_fluid_particles = 0;
	hParam.num_fluid_p = 0;
	for(int k=0; k<100; k++) 
		localid_counter[k]=0;
	
	emit_particle_on = false;
	advect_scriptobject_on = true;
	hParam.enable_dissolution = true;


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
	
}

void SPHSolver::SetupHostBuffer() {
	int maxNP = hParam.maxpnum;
}

//zero by default
cmat3 zero_mat3;

int SPHSolver::AddDefaultParticle() {

	//Single-phase properties.
	host_x.push_back(cfloat3(0, 0, 0));
	host_color.push_back(cfloat4(1, 1, 1, 1));
	host_normal.push_back(cfloat3(0, 0, 0));
	host_unique_id.push_back(host_x.size()-1);
	host_v.push_back(cfloat3(0, 0, 0));
	host_type.push_back(TYPE_FLUID);
	
	
	//Multi-phase properties.
	host_mass.push_back(0);
	host_group.push_back(0);
	host_rest_density.push_back(0);
	host_v_star.push_back(cfloat3(0,0,0));
	host_localid.push_back(0);
	for (int t=0; t<hParam.maxtypenum; t++)
		host_vol_frac.push_back(0);

	
	//Solid material properties.
	host_cauchy_stress.push_back(zero_mat3);

	
	
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

void SPHSolver::AddTestVolume() {
	cfloat3 xmin( -0.12, 0.011, -0.12);
	cfloat3 xmax( 0.12, 0.241, 0.12 );
	int addcount=0;

	float spacing = hParam.spacing;
	float density1 = 0, density2 = 0;
	
	float vf1[3]={1,0,0};
	float vf2[3]={0,1,0};
	for (int t=0; t<hParam.maxtypenum; t++){
		density1 += hParam.densArr[t] * vf1[t];
		density2 += hParam.densArr[t] * vf2[t];
	}

	float vol   = spacing*spacing*spacing;
	int type;
	float pad = 0.03;
	for (float x=xmin.x; x<xmax.x; x+=spacing)
		for (float y=xmin.y; y<xmax.y; y+=spacing)
			for (float z=xmin.z; z<xmax.z; z+=spacing) {
				int pid = AddDefaultParticle();
				
				host_x[pid] = cfloat3(x, y, z);
				
				if (x>xmin.x+pad && x<xmax.x-pad 
					&& y>xmin.y+pad && y<xmax.y-pad 
					&& z>xmin.z+pad && z<xmax.z-pad)
				{
					host_color[pid]=cfloat4(vf2[0], vf2[1], vf2[2], 1);

					type = TYPE_FLUID;
					host_type[pid] = type;

					host_group[pid] = 0;
					host_mass[pid] = vol * density2;
					host_rest_density[pid] = density2;
					host_localid[pid] = localid_counter[type]++;

					for (int t=0; t<hParam.maxtypenum; t++)
						host_vol_frac[pid*hParam.maxtypenum+t] = vf2[t];
					hParam.num_fluid_p += 1;

				}
				else {
					host_color[pid]=cfloat4(vf1[0], vf1[1], vf1[2], 1);

					type = TYPE_DEFORMABLE;
					host_type[pid] = type;

					host_group[pid] = 0;
					host_mass[pid] = vol * density1;
					host_rest_density[pid] = density1;
					host_localid[pid] = localid_counter[type]++;

					for (int t=0; t<hParam.maxtypenum; t++)
						host_vol_frac[pid*hParam.maxtypenum+t] = vf1[t];
					hParam.num_deformable_p += 1;
				}
				

				addcount += 1;
			}
	
	printf("test block No. %d has %d particles.\n", 0, addcount);
}

void SPHSolver::AddMultiphaseFluidVolumes() {

	for (int i=0; i<fluid_volumes.size(); i++) {
		cfloat3 xmin = fluid_volumes[i].xmin;
		cfloat3 xmax = fluid_volumes[i].xmax;
		int addcount=0;

		float* vf    = fluid_volumes[i].volfrac;
		int type = fluid_volumes[i].type;
		if (!(type==TYPE_FLUID || type==TYPE_DEFORMABLE || type==TYPE_GRANULAR))
		{
			printf("Error: wrong volume type.\n"); 
			continue;
		}

		int group = fluid_volumes[i].group;
		float spacing = hParam.spacing;
		float density = 0;
		for (int t=0; t<hParam.maxtypenum; t++)
			density += hParam.densArr[t] * vf[t];

		float mp   = spacing*spacing*spacing* density;
		float pvisc = hParam.viscosity;

		for (float x=xmin.x; x<xmax.x; x+=spacing)
			for (float y=xmin.y; y<xmax.y; y+=spacing)
				for (float z=xmin.z; z<xmax.z; z+=spacing) {
					int pid = AddDefaultParticle();
					host_x[pid] = cfloat3(x, y, z);
					host_color[pid]=cfloat4(vf[0], vf[1], vf[2], 1);
					host_type[pid] = type;
					
					host_group[pid] = group;
					host_mass[pid] = mp;
					host_rest_density[pid] = density;
					host_localid[pid] = localid_counter[type]++;
					
					for (int t=0; t<hParam.maxtypenum; t++)
						host_vol_frac[pid*hParam.maxtypenum+t] = vf[t];

					addcount += 1;
				}
		if (type==TYPE_FLUID) {
			printf("Block No. %d, type: fluid, particle num: %d\n", i, addcount);
			hParam.num_fluid_p += addcount;
		}
		else if(type==TYPE_DEFORMABLE){
			printf("Block No. %d, type: deformable, particle num: %d\n", i, addcount);
			hParam.num_deformable_p += addcount;
		}
		else if(type==TYPE_GRANULAR){
			printf("Block No. %d, type: granular, particle num: %d\n", i, addcount);
		}
	}
}

void SPHSolver::LoadPO(ParticleObject* po) {
	float spacing = hParam.spacing;
	float pden = hParam.restdensity;
	float mp   = spacing*spacing*spacing* pden*2;

	for (int i=0; i<po->pos.size(); i++) {
		int pid = AddDefaultParticle();
		host_x[pid] = po->pos[i];
		host_color[pid] = cfloat4(1,1,1,0.5);
		host_type[pid] = TYPE_RIGID;
		host_normal[pid] = po->normal[i];
		host_mass[pid] = mp;
		host_group[pid] = po->id[i];
	}
}

void SPHSolver::SetupFluidScene() {
	LoadParam("config/sph_scene.xml");
	SetupHostBuffer();

	AddMultiphaseFluidVolumes();
	//AddTestVolume();

	BoundaryGenerator bg;
	ParticleObject* boundary = bg.loadxml("config/water tank.xml");
	LoadPO(boundary);
	delete boundary;

	SetupDeviceBuffer();
	Copy2Device();
}



void SPHSolver::SetupMultiphaseSPHRen() {
	SetupFluidScene();
	run_mode = MSPH_REN;

	Sort();
	EffectiveMass(device_data, num_particles);
	//DFSPHFactor_Multiphase(device_data, num_particles);
	RigidParticleVolume(device_data, num_particles);
	//InitializeDeformable(device_data, num_particles);
}

void SPHSolver::Setup() {
	
	//SetupFluidScene(); run_mode = SPH;

	//SetupDFSPH(); run_mode = DFSPH;

	SetupMultiphaseSPH();

	//SetupMultiphaseSPHRen();

}




void SPHSolver::SetupDeviceBuffer() {

	//particle
	int maxpnum = host_x.size();
	host_id_table.resize(maxpnum);

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
	
	cudaMalloc(&device_data.sortedVFrac, num_pt*sizeof(float));
	cudaMalloc(&device_data.sortedRestDensity,	maxpnum*sizeof(float));
	cudaMalloc(&device_data.sorted_effective_mass, maxpnum*sizeof(float));
	

	// Deformable Solid
	cudaMalloc(&device_data.strain_rate, maxpnum*sizeof(cmat3));
	cudaMalloc(&device_data.cauchy_stress, maxpnum*sizeof(cmat3));
	cudaMalloc(&device_data.local_id, maxpnum*sizeof(int));
	cudaMalloc(&device_data.neighborlist, hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(int));
	cudaMalloc(&device_data.neighbordx, hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(cfloat3));
	cudaMalloc(&device_data.correct_kernel, hParam.num_deformable_p*sizeof(cmat3));
	cudaMalloc(&device_data.x0, hParam.num_deformable_p*sizeof(cfloat3));
	cudaMalloc(&device_data.rotation, hParam.num_deformable_p*sizeof(cmat3));
	cudaMalloc(&device_data.trim_tag, hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(int));
	cudaMalloc(&device_data.length0,  hParam.num_deformable_p*NUM_NEIGHBOR*sizeof(float));

	cudaMalloc(&device_data.sorted_cauchy_stress, maxpnum*sizeof(cmat3));
	cudaMalloc(&device_data.sorted_local_id, maxpnum*sizeof(int));
	


	int glen = hParam.gridres.prod();

	cudaMalloc(&device_data.particleHash, maxpnum * sizeof(int));
	cudaMalloc(&device_data.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&device_data.gridCellStart, glen * sizeof(int));
	cudaMalloc(&device_data.gridCellEnd, glen * sizeof(int));
}

};
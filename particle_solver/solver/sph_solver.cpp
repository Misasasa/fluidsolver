
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "sph_solver.h"


namespace sph{

extern SimParam_SPH hParam;

void SPHSolver::copy2Device() {
	numP = hPos.size();

	cudaMemcpy(dData.pos,	hPos.data(), numP * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.vel,	hVel.data(), numP * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.normal, hNormal.data(), numP * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color, hColor.data(), numP * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.type,  hType.data(), numP * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.group, hGroup.data(), numP * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.mass, hMass.data(), numP * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.uniqueId, hUniqueId.data(), numP * sizeof(int), cudaMemcpyHostToDevice);

	int numPT = numP * hParam.maxtypenum;
	cudaMemcpy(dData.vFrac, hVFrac.data(), numPT * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.restDensity, hDensity.data(), numP * sizeof(float), cudaMemcpyHostToDevice);
	copyDeviceBuffer();
}


void SPHSolver::copy2Host() {

	//cudaMemcpy(hMass.data(),	dData.mass, sizeof(float)*numP, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos.data(),		dData.pos, sizeof(cfloat3)*numP, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hVel.data(),		dData.vel, sizeof(cfloat3)*numP, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hUniqueId.data(), dData.uniqueId, sizeof(int)*numP, cudaMemcpyDeviceToHost);
	//cudaMemcpy(hIndexTable.data(), dData.indexTable, hParam.maxpnum * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hColor.data(),	dData.color, numP * sizeof(cfloat4), cudaMemcpyDeviceToHost);
}


void SPHSolver::sort() {
	calcHash(dData, numP);

	sortParticle(dData, numP);

	//cudaMemset(dData.gridCellCollisionFlag, 0, sizeof(char)*numGC);

	reorderDataAndFindCellStart(dData, numP, numGC);
	
	cudaMemcpy(dData.pos,		dData.sortedPos,	numP * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.vel,		dData.sortedVel,	numP * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.normal,	dData.sortedNormal, numP * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.color,		dData.sortedColor,	numP * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.type,		dData.sortedType,	numP * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.group,		dData.sortedGroup,	numP * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.mass,		dData.sortedMass,	numP * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.uniqueId,	dData.sortedUniqueId, numP * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.v_star,    dData.sortedV_star, numP * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.restDensity, dData.sortedRestDensity, numP * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.vFrac,		dData.sortedVFrac,	numP * hParam.maxtypenum * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.massFac,   dData.sortedMassFac, numP * sizeof(float), cudaMemcpyDeviceToDevice);

}

void SPHSolver::solveSPH() {

	sort();

	computePressure(dData, numP);

	computeForce(dData, numP);

	advect(dData, numP);

	copy2Host();
}



void SPHSolver::setupDFSPH() {
	setupFluidScene();
	sort();
	computeDensityAlpha(dData, numP);
}

void SPHSolver::solveDFSPH() {
	//compute non-pressure force
	//predict velocities
	computeNonPForce(dData, numP);

	//correct density
	correctDensityError(dData, numP, 5, 1, false);

	//update neighbors
	sort();

	//update rho and alpha
	computeDensityAlpha(dData,numP);

	//correct divergence
	//update velocities
	correctDivergenceError(dData, numP, 5, 1, false);

	copy2Host();
}




/*
Still we solve multiphase model with divergence-free SPH.
V_m denotes the volume-averaged velocity of each particle.
*/

void SPHSolver::setupMultiphaseSPH() {
	setupFluidScene();
	sort();
	computeDFAlpha_MPH(dData, numP);
}

void SPHSolver::solveMultiphaseSPH() {

	//compute non-pressure force
	//predict velocity
	computeNonPForce_MPH(dData, numP);


	updateMassFac(dData, numP);

	//correct density
	correctDensity_MPH(dData, numP, 5, 0.1, true);

	//update neighbors
	sort();
	
	//updateMassFac(dData, numP);

	//update rho and alpha
	computeDFAlpha_MPH(dData, numP);
	//computeDensityAlpha(dData, numP);

	//correct divergence
	//update velocity
	correctDivergence_MPH(dData, numP, 5, 0.1, false);

	copy2Host();

}





void SPHSolver::step() {

	if (numP>0) {
		switch (runmode) {
		case SPH:
			solveSPH(); break;
		case DFSPH:
			solveDFSPH(); break;
		case MSPH:
			solveMultiphaseSPH(); break;
		}
		

	}

	//if(bEmitParticle)
	//	fluidSrcEmit();

	frameNo++;
	time += hParam.dt;
}

void SPHSolver::dumpSimulationDataText() {
	printf("Dumping simulation data text at frame %d\n", frameNo);
	char filepath[1000];
	sprintf(filepath, ".\\dump\\%03d.dat", frameNo);
	
	FILE* fp = fopen(filepath, "w+");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}

	copy2host_full();

	// Particle Data
	fprintf(fp, "%d\n", numP);
	for (int i=0; i<numP; i++) {
		fprintf(fp, "%f %f %f ", hPos[i].x, hPos[i].y, hPos[i].z);
		fprintf(fp, "%f %f %f %f ", hColor[i].x, hColor[i].y, hColor[i].z, hColor[i].w);
		fprintf(fp, "%f %f %f ", hVel[i].x, hVel[i].y, hVel[i].z);
		fprintf(fp, "%d ", hType[i]);
		fprintf(fp, "%d ", hGroup[i]);
		fprintf(fp, "%f ", hMass[i]);
		fprintf(fp, "%d ", hUniqueId[i]);
	}
	fclose(fp);
}



void SPHSolver::HandleKeyEvent(char key) {
	switch (key) {
	case 'b':
		dumpSimulationDataText();
		break;
	//case 'r':
	//	dumpRenderingData();
	//	break;
	case 'e':
		bEmitParticle = !bEmitParticle;
		if (bEmitParticle)
			printf("Start to emit particles.\n");
		else
			printf("Stop emitting particles.\n");
		break;
	}
}

void SPHSolver::copy2host_full() {
	cudaMemcpy(hMass.data(),	dData.mass, sizeof(float)*numP, cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos.data(), dData.pos, sizeof(cfloat3)*numP, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel.data(),		dData.vel, sizeof(cfloat3)*numP, cudaMemcpyDeviceToHost);
	cudaMemcpy(hUniqueId.data(), dData.uniqueId, sizeof(int)*numP, cudaMemcpyDeviceToHost);
	cudaMemcpy(hIndexTable.data(), dData.indexTable, hParam.maxpnum * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hColor.data(), dData.color, numP * sizeof(cfloat4), cudaMemcpyDeviceToHost);
}

void SPHSolver::parseParam(char* xmlpath) {
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
	caseid = reader.GetInt("SceneId");

	
	//find corresponding scene node
	int tmp;
	while (true) {
		sceneElement->QueryIntAttribute("id", &tmp);
		if (tmp == caseid)
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


	loadFluidVolume(sceneElement, hParam.maxtypenum, fvs);


	//=============================================
	//               Particle Boundary
	//=============================================
	reader.Use(boundElement);
	hParam.bRestdensity = reader.GetFloat("RestDensity");
	hParam.bvisc = reader.GetFloat("Viscosity");
}



void SPHSolver::loadParam(char* xmlpath) {
	frameNo = 0;
	numP = 0;
	numFluidP = 0;

	parseParam(xmlpath);

	hParam.dx = hParam.smoothradius;
	float sr = hParam.smoothradius;

	//hParam.gridres = cint3(64, 64, 64);
	//hParam.gridxmin.x = 0 - hParam.gridres.x / 2 * hParam.dx;
	//hParam.gridxmin.z = 0 - hParam.gridres.z / 2 * hParam.dx;
	hParam.gridres.x = roundf((hParam.gridxmax.x - hParam.gridxmin.x)/hParam.dx);
	hParam.gridres.y = roundf((hParam.gridxmax.y - hParam.gridxmin.y)/hParam.dx);
	hParam.gridres.z = roundf((hParam.gridxmax.z - hParam.gridxmin.z)/hParam.dx);
	numGC = hParam.gridres.prod();

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
	
	
	//runmode = SPH;
	runmode = DFSPH;
	runmode = MSPH;
	bEmitParticle = false;
	dt = hParam.dt;
}

void SPHSolver::setupHostBuffer() {
	int maxNP = hParam.maxpnum;
	/*
	hPos.resize(maxNP);
	hColor.resize(maxNP);
	hVel.resize(maxNP);
	hType.resize(maxNP);
	hGroup.resize(maxNP);
	hMass.resize(maxNP);
	hUniqueId.resize(maxNP);
	hIndexTable.resize(maxNP);
	*/
}

int SPHSolver::addDefaultParticle() {
	hPos.push_back(cfloat3(0, 0, 0));
	hColor.push_back(cfloat4(1, 1, 1, 1));
	hNormal.push_back(cfloat3(0, 0, 0));
	hUniqueId.push_back(numP);
	hVel.push_back(cfloat3(0, 0, 0));
	hType.push_back(TYPE_FLUID);
	hMass.push_back(0);
	hGroup.push_back(0);
	hDensity.push_back(0);

	for(int t=0; t<hParam.maxtypenum; t++)
		hVFrac.push_back(0);
	
	return hPos.size()-1;
}


void SPHSolver::addfluidvolumes() {
	

	for (int i=0; i<fvs.size(); i++) {
		cfloat3 xmin = fvs[i].xmin;
		cfloat3 xmax = fvs[i].xmax;
		int addcount=0;

		//float* vf    = fvs[i].volfrac;
		float spacing = hParam.spacing;
		float pden = hParam.restdensity;
		float mp   = spacing*spacing*spacing* pden;
		float pvisc = hParam.viscosity;

		for (float x=xmin.x; x<xmax.x; x+=spacing)
			for (float y=xmin.y; y<xmax.y; y+=spacing)
				for (float z=xmin.z; z<xmax.z; z+=spacing) {
					int pid = addDefaultParticle();
					hPos[pid] = cfloat3(x, y, z);
					hColor[pid]=cfloat4(0.7, 0.75, 0.95, 1);
					hType[pid] = TYPE_FLUID;
					hMass[pid] = mp;
					hGroup[pid] = 0;
					addcount += 1;
				}

		printf("fluid block No. %d has %d particles.\n", i, addcount);
	}
}

void SPHSolver::addMultiphaseFluidVolumes() {

	for (int i=0; i<fvs.size(); i++) {
		cfloat3 xmin = fvs[i].xmin;
		cfloat3 xmax = fvs[i].xmax;
		int addcount=0;

		float* vf    = fvs[i].volfrac;
		float spacing = hParam.spacing;
		float pden = 0; //hParam.restdensity;
		for(int t=0; t<hParam.maxtypenum; t++)
			pden += hParam.densArr[t] * vf[t];

		float mp   = spacing*spacing*spacing* pden;
		float pvisc = hParam.viscosity;

		for (float x=xmin.x; x<xmax.x; x+=spacing)
			for (float y=xmin.y; y<xmax.y; y+=spacing)
				for (float z=xmin.z; z<xmax.z; z+=spacing) {
					int pid = addDefaultParticle();
					hPos[pid] = cfloat3(x, y, z);
					//hColor[pid]=cfloat4(0.7, 0.75, 0.95, 1);
					hColor[pid]=cfloat4(vf[0], vf[1], vf[2], 1);
					hType[pid] = TYPE_FLUID;
					hGroup[pid] = 0;
					
					hMass[pid] = mp;
					hDensity[pid] = pden;
					for(int t=0;  t<hParam.maxtypenum; t++)
						hVFrac[pid*hParam.maxtypenum+t] = vf[t];

					addcount += 1;
				}

		printf("fluid block No. %d has %d particles.\n", i, addcount);
	}
}

void SPHSolver::loadPO(ParticleObject* po) {
	float spacing = hParam.spacing;
	float pden = hParam.restdensity;
	float mp   = spacing*spacing*spacing* pden*2;

	for (int i=0; i<po->pos.size(); i++) {
		int pid = addDefaultParticle();
		hPos[pid] = po->pos[i];
		hColor[pid] = cfloat4(1,1,1,0.2);
		hType[pid] = TYPE_BOUNDARY;
		hNormal[pid] = po->normal[i];
		hMass[pid] = mp;
		hGroup[pid] = 0;
	}
}

void SPHSolver::setupFluidScene() {
	loadParam("config/sph_scene.xml");
	setupHostBuffer();

	//addfluidvolumes();
	addMultiphaseFluidVolumes();

	BoundaryGenerator bg;
	ParticleObject* boundary = bg.loadxml("script_object/box.xml");
	loadPO(boundary);
	delete boundary;

	setupDeviceBuffer();
	copy2Device();
}

void SPHSolver::setup() {
	//setupFluidScene();

	//setupDFSPH();

	setupMultiphaseSPH();
}




void SPHSolver::setupDeviceBuffer() {

	//particle
	int maxpnum = hPos.size();

	cudaMalloc(&dData.pos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.vel, maxpnum * sizeof(float3));
	cudaMalloc(&dData.color, maxpnum * sizeof(cfloat4));
	cudaMalloc(&dData.type, maxpnum * sizeof(int));
	cudaMalloc(&dData.group, maxpnum * sizeof(int));
	cudaMalloc(&dData.uniqueId, maxpnum * sizeof(int));
	cudaMalloc(&dData.mass, maxpnum * sizeof(float));
	cudaMalloc(&dData.density, maxpnum * sizeof(float));
	cudaMalloc(&dData.pressure, maxpnum * sizeof(float));
	cudaMalloc(&dData.force, maxpnum*sizeof(cfloat3));
	cudaMalloc(&dData.normal, maxpnum*sizeof(cfloat3));


	cudaMalloc(&dData.sortedPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedVel, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedColor, maxpnum * sizeof(cfloat4));
	cudaMalloc(&dData.sortedMass, maxpnum * sizeof(float));
	cudaMalloc(&dData.sortedType, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedGroup, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedNormal, maxpnum*sizeof(cfloat3));
	cudaMalloc(&dData.sortedUniqueId, maxpnum * sizeof(int));
	cudaMalloc(&dData.indexTable, maxpnum * sizeof(int));
	
	//DFSPH
	cudaMalloc(&dData.alpha, maxpnum * sizeof(float));
	cudaMalloc(&dData.v_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&dData.x_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&dData.pstiff, maxpnum * sizeof(float));
	cudaMalloc(&dData.sortedV_star, maxpnum * sizeof(cfloat3));
	cudaMalloc(&dData.error, maxpnum * sizeof(float));

	//Multiphase
	int ptnum = maxpnum*hParam.maxtypenum;
	cudaMalloc(&dData.vFrac,		ptnum*sizeof(float));
	cudaMalloc(&dData.restDensity,	maxpnum*sizeof(float));
	cudaMalloc(&dData.driftV,		ptnum*sizeof(cfloat3));
	cudaMalloc(&dData.sortedVFrac,	ptnum*sizeof(float));
	cudaMalloc(&dData.sortedRestDensity,	maxpnum*sizeof(float));
	cudaMalloc(&dData.massFac, maxpnum*sizeof(float));
	cudaMalloc(&dData.sortedMassFac, maxpnum*sizeof(float));

	int glen = hParam.gridres.prod();

	cudaMalloc(&dData.particleHash, maxpnum * sizeof(int));
	cudaMalloc(&dData.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&dData.gridCellStart, glen * sizeof(int));
	cudaMalloc(&dData.gridCellEnd, glen * sizeof(int));
}

};
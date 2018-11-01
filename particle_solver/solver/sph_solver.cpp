
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "sph_solver.h"

namespace sph{

extern SimParam_SPH hParam;

void SPHSolver::copy2Device() {

	cudaMemcpy(dData.pos,	hPos.data(), numP * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.vel,	hVel.data(), numP * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color, hColor.data(), numP * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.type,  hType.data(), numP * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.group, hGroup.data(), numP * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.mass, hMass.data(), numP * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.uniqueId, hUniqueId.data(), numP * sizeof(int), cudaMemcpyHostToDevice);

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
	
	cudaMemcpy(dData.pos,		dData.sortedPos, numP * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.vel,		dData.sortedVel, numP * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.color,		dData.sortedColor, numP * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.type,		dData.sortedType, numP * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.group,		dData.sortedGroup, numP * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.mass,		dData.sortedMass, numP * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.uniqueId,	dData.sortedUniqueId, numP * sizeof(int), cudaMemcpyDeviceToDevice);
	
}

void SPHSolver::solveSPH() {

	//computePressure
	computePressure(dData, numP);

	//computeForce
	computeForce(dData, numP);

	//advect
	advect(dData, numP);


}

void SPHSolver::step() {

	if (numP>0) {
		sort();
		solveSPH();
		copy2Host();
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
	
	
	runmode = 0;
	bEmitParticle = false;
}

void SPHSolver::setupHostBuffer() {
	int maxNP = hParam.maxpnum;
	hPos.resize(maxNP);
	hColor.resize(maxNP);
	hVel.resize(maxNP);
	hType.resize(maxNP);
	hGroup.resize(maxNP);
	hMass.resize(maxNP);
	hUniqueId.resize(maxNP);
	hIndexTable.resize(maxNP);
}

int SPHSolver::addDefaultParticle() {
	if (numP ==hParam.maxpnum)
		return -1;
	else {
		hMass[numP] = 0;
		hVel[numP].Set(0, 0, 0);
		hUniqueId[numP] = numP;

		numP++;
		return numP-1;
	}
}


void SPHSolver::addfluidvolumes() {
	int addcount=0;

	for (int i=0; i<fvs.size(); i++) {
		cfloat3 xmin = fvs[i].xmin;
		cfloat3 xmax = fvs[i].xmax;
		//float* vf    = fvs[i].volfrac;
		float spacing = hParam.spacing;
		float pden = hParam.restdensity;
		float mp   = spacing*spacing*spacing* pden;
		float pvisc = hParam.viscosity;

		for (float x=xmin.x; x<xmax.x; x+=spacing)
			for (float y=xmin.y; y<xmax.y; y+=spacing)
				for (float z=xmin.z; z<xmax.z; z+=spacing) {

					int pid = addDefaultParticle();
					if (pid==-1)
						goto done_add;
					addcount += 1;

					hPos[pid] = cfloat3(x, y, z);
					hColor[pid] = cfloat4(0.7, 0.75, 0.95, 1);
					hType[pid] = TYPE_FLUID;
					hMass[pid] = mp;
					hGroup[pid] = 0;
				}

		printf("fluid block No. %d has %d particles.\n", i+1, addcount);
	}
done_add:
	printf("%d particles added in total.\n", addcount);
}

void SPHSolver::setupFluidScene() {
	loadParam("config/sph_scene.xml");
	setupHostBuffer();

	addfluidvolumes();

	setupDeviceBuffer();
	copy2Device();
}

void SPHSolver::setup() {
	setupFluidScene();
}

void SPHSolver::setupDeviceBuffer() {

	//particle
	int maxpnum = hParam.maxpnum;

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
	
	cudaMalloc(&dData.sortedPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedVel, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedColor, maxpnum * sizeof(cfloat4));
	cudaMalloc(&dData.sortedMass, maxpnum * sizeof(float));
	cudaMalloc(&dData.sortedType, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedGroup, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedUniqueId, maxpnum * sizeof(int));
	cudaMalloc(&dData.indexTable, maxpnum * sizeof(int));

	int glen = hParam.gridres.prod();

	cudaMalloc(&dData.particleHash, maxpnum * sizeof(int));
	cudaMalloc(&dData.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&dData.gridCellStart, glen * sizeof(int));
	cudaMalloc(&dData.gridCellEnd, glen * sizeof(int));
}

};
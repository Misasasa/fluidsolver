
#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "pbfsolver.h"




extern SimParam hParam;



void PBFSolver::parseParam(char* xmlpath) {
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

	//some general parameter
	//pele = fluidElement;
	reader.Use(fluidElement);
	hParam.gravity =  reader.GetFloat3("Gravity");
	hParam.gridxmin = reader.GetFloat3("VolMin");
	hParam.gridxmax = reader.GetFloat3("VolMax");
	caseid = reader.GetInt("SceneId");


	int tmp;
	//find corresponding scene node
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

	//=============================================
	//            Scene Paramters
	//=============================================
	reader.Use(sceneElement);

	hParam.maxpnum = reader.GetInt("MaxPNum");
	hParam.maxtypenum = reader.GetInt("MaxTypeNum");
	hParam.maxobjnum = reader.GetInt("MaxObjNum");

	hParam.dt = reader.GetFloat("DT");
	hParam.spacing = reader.GetFloat("PSpacing");
	hParam.smoothradius = reader.GetFloat("SmoothRadius");
	
	hParam.boundstiff = reader.GetFloat("BoundStiff");
	hParam.bounddamp = reader.GetFloat("BoundDamp");
	hParam.softminx = reader.GetFloat3("SoftBoundMin");
	hParam.softmaxx = reader.GetFloat3("SoftBoundMax");

	loadFluidVols(sceneElement);

	
	//=============================================
	//                     PBF
	//=============================================
	hParam.viscosity = reader.GetFloat("Viscosity");
	hParam.restdensity = reader.GetFloat("RestDensity");
	hParam.pbfepsilon = reader.GetFloat("PBFepsilon");
	hParam.qfactor = reader.GetFloat("QFactor");
	hParam.k_ti = reader.GetFloat("k_TensileInstability");
	hParam.n_ti = reader.GetFloat("n_TensileInstability");
	hParam.vorticityfactor = reader.GetFloat("VorticityFactor");
	reader.GetFloatN(densratio, hParam.maxtypenum, "DensityRatio");
	reader.GetFloatN(viscratio, hParam.maxtypenum, "ViscRatio");
	//============================================
	//                  Porosity
	//============================================
	hParam.k_absorb = reader.GetFloat("k_absorb");
	hParam.cloth_porosity = reader.GetFloat("cloth_porosity");
	hParam.diffuseDistance = reader.GetFloat("diffuseDistance");
	hParam.k_diffuse = reader.GetFloat("k_diffuse");
	hParam.k_diffuse_gravity = reader.GetFloat("k_diffuse_gravity");
	hParam.k_dripBuf = reader.GetFloat("k_dripBuf");
	hParam.max_saturation = reader.GetFloat("max_saturation");
	hParam.emitThres = reader.GetFloat("EmitThres");

	//=============================================
	//               Unified Particle
	//=============================================
	hParam.collisionDistance = reader.GetFloat("collisionDistance");
	hParam.overlap = reader.GetFloat("overlap");
	hParam.selfcollisionDistance = reader.GetFloat("selfcollisionDistance");
	hParam.global_relaxation = reader.GetFloat("global_relaxation");

	//=============================================
	//                    Cloth
	//=============================================
	hParam.stretchStiff = reader.GetFloat("StretchStiff");
	hParam.compressStiff = reader.GetFloat("CompressStiff");
	hParam.bendingstiff = reader.GetFloat("BendingStiff");
	hParam.k_damping = reader.GetFloat("K_damping");
	hParam.maxconstraintnum = reader.GetInt("MaxConstraintNum");
	hParam.maxtrianglenum = reader.GetInt("MaxTriangleNum");
	hParam.cloth_density = reader.GetFloat("cloth_density");
	hParam.restvol = reader.GetFloat("restvol");
	hParam.clothThickness = reader.GetFloat("ClothThickness");
	hParam.cloth_selfcd = reader.GetFloat("ClothSelfColDistance");
	hParam.stretchComp = reader.GetFloat("StretchCompliance");
	hParam.compressComp = reader.GetFloat("CompressCompliance");
	hParam.pE = reader.GetFloat("PressureEnvironment");
	hParam.resistance = reader.GetFloat("Resistance");
	hParam.collisionStiff = reader.GetFloat("CollisionStiff");
	//=============================================
	//               Surface Tension
	//=============================================
	hParam.pairwise_k = reader.GetFloat("Pairwise_k");
	hParam.pairwise_c = reader.GetFloat("Pairwise_c");
	hParam.surface_threshold = reader.GetFloat("Surface_Threshold");
	hParam.surface_stiff = reader.GetFloat("Surface_Stiff");
	hParam.dens_thres = reader.GetFloat("Density_Threshold");

	//=============================================
	//               Anisotropic Kernel
	//=============================================
	hParam.aniso_support = reader.GetFloat("Anisosupport");
	hParam.aniso_thres = reader.GetFloat("Anisothres");
	hParam.aniso_thres_coplanar = reader.GetFloat("Anisothres_coplanar");//1.1;

	//=============================================
	//               Von Mises Model
	//=============================================
	hParam.solidK = reader.GetFloat("SolidK");
	hParam.solidG = reader.GetFloat("SolidG");
	hParam.Yield = reader.GetFloat("Yield");

	hParam.volumeStiff = reader.GetFloat("VolumeStiff");

	//=============================================
	//               Particle Boundary
	//=============================================
	reader.Use(boundElement);
	hParam.bRestdensity = reader.GetFloat("RestDensity");
	hParam.bvisc = reader.GetFloat("Viscosity");
}

void PBFSolver::loadParam(char* xmlpath) {
	frameNo = 0;

	parseParam(xmlpath);

	hParam.dx = hParam.smoothradius;
	float sr = hParam.smoothradius;

	hParam.gridres = cint3(64, 64, 64);
	hParam.gridxmin.x = 0 - hParam.gridres.x / 2 * hParam.dx;
	hParam.gridxmin.y = -10;
	hParam.gridxmin.z = 0 - hParam.gridres.z / 2 * hParam.dx;

	//setup kernels

	hParam.kpoly6 = 315.0f / (64.0f * 3.141592 * pow(sr, 9.0f));
	hParam.kspiky =  15 / (3.141592 * pow(sr, 6.0f));
	hParam.kspikydiff = -45.0f / (3.141592 * pow(sr, 6.0f));
	hParam.klaplacian = 45.0f / (3.141592 * pow(sr, 6.0f));
	hParam.kspline = 1.0f/3.141593f/pow(sr, 3.0f);

	for (int k=0; k<hParam.maxtypenum; k++) {
		hParam.densArr[k] = hParam.restdensity * densratio[k];
		hParam.viscArr[k] = hParam.viscosity * viscratio[k];
	}

	//anisotropic kernels
	hParam.avgpos_k = 0.95;
	numGridCells = hParam.gridres.prod();
	
	runmode = RUN_GPU;
	bEmitParticle = false;
	bVolumeCorr = false;
	bJetGas = false;
	bInitVol = false;
}






void PBFSolver::setupBasicScene() {
	loadParam("config/pbf_scene.xml");
	setupHostBuffer();
	
	addfluidvolumes();

	setupDeviceBuffer();
	copy2Device();
}

// CASE 1
void PBFSolver::setupBalloonScene() {
	
	loadParam("config/pbf_scene.xml");
	setupHostBuffer();

	//cloth
	cmat4 materialMat;
	materialMat = IDENTITY_MAT;
	float scale = 20;
	//ScaleMatrix(materialMat, cfloat3(scale, scale, scale));
	TranslateMatrix(materialMat, cfloat3(0, 35, 0));

	//addBalloon(materialMat);
	//addParticleFluidSource();

	//loadSimulationData("dump/75percent.dat", materialMat);
	loadSimulationDataText("dump/10p.dat", materialMat);

	//bVolumeCorr = true;
	//bJetGas = true;
	bNormalizePos = true;
	numEdgeCons = edgeConstraints.size();
	numTriangles = trianglelist.size();

	setupDeviceBuffer();
	copy2Device();
}

// CASE 2

void PBFSolver::setupWaterClothScene() {
	loadParam("config/pbf_waterClothScene.xml");
	setupHostBuffer();

	//cloth
	
	//addWaterBag();
	addFluidSource();
	//addfluidvolumes();

	numEdgeCons = edgeConstraints.size();
	numTriangles = trianglelist.size();

	setupDeviceBuffer();
	copy2Device();
}

void PBFSolver::setupInjection() {
	loadParam("config/pbf_scene.xml");
	setupHostBuffer();

	//cloth
	//addCloth();
	//addWaterBag();
	//loadSimulationData("dump/3506.dat");
	addBalloon();

	//addFluidSource();
	addParticleFluidSource();
	//addfluidvolumes();

	//addBalloon();
	hParam.restvol = 0;
	bVolumeCorr = false;
	bJetGas = false;
	numEdgeCons = edgeConstraints.size();
	numTriangles = trianglelist.size();

	setupDeviceBuffer();
	copy2Device();
}

// CASE 3
void PBFSolver::setupMultiBalloonScene() {
	
	loadParam("config/pbf_scene.xml");
	setupHostBuffer();

	//cloth

	cmat4 materialMat;
	materialMat = IDENTITY_MAT;
	float scale = 20;
	//ScaleMatrix(materialMat, cfloat3(scale, scale, scale));
	TranslateMatrix(materialMat, cfloat3(0, 30, 0));

	//addBalloon(materialMat);
	//loadSimulationData("dump/371.dat", materialMat);
	//addfluidvolumes();
	//TranslateMatrix(materialMat, cfloat3(0, 20, 0));
	//addBalloon(materialMat);

	//addParticleFluidSource();
	addFluidSource();

	loadSimulationDataText("dump/0p.dat", materialMat);

	TranslateMatrix(materialMat, cfloat3(0, 20, 0));
	
	loadSimulationDataText("dump/10p.dat", materialMat);

	TranslateMatrix(materialMat, cfloat3(0, 20, 0));

	loadSimulationDataText("dump/50p.dat", materialMat);

	TranslateMatrix(materialMat, cfloat3(-10, 20, 0));

	loadSimulationDataText("dump/0p.dat", materialMat);

	TranslateMatrix(materialMat, cfloat3(20, 10, 0));

	loadSimulationDataText("dump/100p.dat", materialMat);

	loadParticleSample("model/boundary/glass40.dat");

	hParam.restvol = 0;//10000;
	bVolumeCorr = false;
	bModelPorous = false;
	bReleaseSource = false;
	bNormalizePos = false;

	numEdgeCons = edgeConstraints.size();
	numTriangles = trianglelist.size();

	setupDeviceBuffer();
	copy2Device();
}

void PBFSolver::setupArmadilloScene() {
	loadParam("config/pbf_scene.xml");
	setupHostBuffer();

	addArmadilo();

	hParam.restvol = 0;//10000;
	bNormalizePos = false;
	numEdgeCons = edgeConstraints.size();
	numTriangles = trianglelist.size();

	setupDeviceBuffer();
	copy2Device();
}


void PBFSolver::addFluidSource() {
	FluidSrc fs;

	fs.srcpos = cfloat3(25, 80, 0);
	fs.radius = 8;
	fs.norm = cfloat3(-1, -1.5, 0);
	fs.norm /= fs.norm.mode();
	fs.speed = 20;
	fs.type = FLUIDSRC_ROUND;
	fs.interval = hParam.spacing / fs.speed/hParam.dt * 1.5;

	fluidsrcs.push_back(fs);
}

void PBFSolver::fluidSrcEmit() {
	vector<cfloat3> pArr;
	vector<cfloat3> vArr;
	pArr.clear();
	vArr.clear();

	for (int i = 0; i < fluidsrcs.size(); i++) {
		FluidSrc& fs = fluidsrcs[i];
		cmat4 mm = IDENTITY_MAT;
		RotateAboutZ(mm,acos(fs.norm.y));
		cfloat3 xzmap = cfloat3(fs.norm.x, 0, fs.norm.z);
		if(xzmap.mode()>EPSILON)
			RotateAboutY(mm, -acos(fs.norm.x /xzmap.mode()));
		
		cfloat4 t(0, 1, 0, 1);
		t = mm*t;
		//printf("%f %f %f %f\n", t.x, t.y, t.z, t.w);
		
		if (frameNo % fs.interval != 0)
			continue;

		for (float x = -fs.radius; x < fs.radius; x += hParam.spacing) {
			for (float z = -fs.radius; z < fs.radius; z += hParam.spacing) {
				if (abs(x*x + z*z) > fs.radius*fs.radius)
					continue;
				cfloat4 p(x, 0, z, 1);
				p = mm*p;

				cfloat3 addp = cfloat3(p.x, p.y, p.z) + fs.srcpos;
				//printf("%f %f %f\n", addp.x, addp.y, addp.z);
				
				pArr.push_back(addp);
				vArr.push_back(fs.norm * fs.speed);
			}
		}
	}

	//int frameinterval = roundf(1 / (fs*speed*hParam.dt));
	int start = numParticles - numParticlesDeleted;
	int pid = start;
	//frameinterval = 10;

	for(int i=0; i<pArr.size(); i++){
		if(pid == hParam.maxpnum)
			return;
		if(pid==numParticles){
			hUniqueId[pid] = numParticles++;
			hJetFlag[pid] = 0;
		}
		else {
			numParticlesDeleted --;
		}
		hPos[pid] = pArr[i];
		hVel[pid] = vArr[ i];
		hColor[pid] = cfloat4(0.7, 0.75, 0.95, 0.8);
		hMass[pid] = hParam.restdensity*pow(hParam.spacing, 3);
		hType[pid] = TYPE_FLUID;
		hInvMass[pid] = 1 / hMass[pid];

		pid ++;
	}
	//printf("%d\n",numParticles);
	copy2Device_partial(start, pid);
}


// Particle Fluid Source

void PBFSolver::addParticleFluidSource() {
	FluidSrc fs;

	int pid = addDefaultParticle();
	hPos[pid] = cfloat3(0, 40, 0);
	hColor[pid] = cfloat4(0.5, 0.9, 1, 1);
	hType[pid] = TYPE_EMITTER;
	hMass[pid] = 1;
	hInvMass[pid] = 0;	

	fs.srcpos = cfloat3(0, 40, 0);
	fs.radius = 0.8;
	fs.norm = cfloat3(0, -1, 0);
	fs.speed = 1;
	fs.type = FLUIDSRC_ROUND;
	fluidsrcs.push_back(fs);

	pid = addDefaultParticle();
	hPos[pid] = cfloat3(-1, 40, 0);
	hColor[pid] = cfloat4(0.5, 0.9, 1, 1);
	hType[pid] = TYPE_EMITTER;
	hMass[pid] = 1;
	hInvMass[pid] = 0;

	pid = addDefaultParticle();
	hPos[pid] = cfloat3(0, 40, -1);
	hColor[pid] = cfloat4(0.5, 0.9, 1, 1);
	hType[pid] = TYPE_EMITTER;
	hMass[pid] = 1;
	hInvMass[pid] = 0;

	pid = addDefaultParticle();
	hPos[pid] = cfloat3(-1, 40, -1);
	hColor[pid] = cfloat4(0.5, 0.9, 1, 1);
	hType[pid] = TYPE_EMITTER;
	hMass[pid] = 1;
	hInvMass[pid] = 0;

	//printf("added %d particles...\n", addcount);
	
}

void PBFSolver::particleFluidSrcEmit() {
	vector<cfloat3> pArr;
	vector<cfloat3> vArr;
	pArr.clear();
	vArr.clear();

	for (int i = 0; i < fluidsrcs.size(); i++) {
		FluidSrc& fs = fluidsrcs[i];
		cmat4 mm = IDENTITY_MAT;
		RotateAboutZ(mm, acos(fs.norm.y));
		RotateAboutY(mm, -acos(fs.norm.x));

		if (frameNo % fs.interval != 0)
			continue;

		for (float x = -fs.radius; x < fs.radius; x += hParam.spacing) {
			for (float z = -fs.radius; z < fs.radius; z += hParam.spacing) {
				if (abs(x*x + z*z) > fs.radius*fs.radius)
					continue;
				cfloat4 p(x, 0, z, 1);
				p = mm*p;

				cfloat3 addp = cfloat3(p.x, p.y, p.z) + fs.srcpos;
				//printf("%f %f %f\n", addp.x, addp.y, addp.z);

				pArr.push_back(addp);
				vArr.push_back(fs.norm * fs.speed);
			}
		}
	}

	//int frameinterval = roundf(1 / (fs*speed*hParam.dt));
	int start = numParticles - numParticlesDeleted;
	int pid = start;
	//frameinterval = 10;
	
	for (int i=0; i<pArr.size(); i++) {
		if (pid == hParam.maxpnum)
			return;
		if (pid==numParticles) {
			hUniqueId[pid] = numParticles++;
			hJetFlag[pid] = 0;
		}
		else {
			numParticlesDeleted --;
		}
		hPos[pid] = pArr[i];
		hPos[pid].y += (float)rand()/RAND_MAX*0.01;
		hVel[pid] = vArr[i];
		hColor[pid] = cfloat4(0.7, 0.75, 0.95, 0.8);
		hMass[pid] = hParam.restdensity*pow(hParam.spacing, 3);
		hType[pid] = TYPE_FLUID;
		hInvMass[pid] = 1 / hMass[pid];
		hGroup[pid] = 0;

		pid ++;
	}
	//printf("%d\n", numParticles);
	copy2Device_partial(start, pid);
	printf("Total %d particles at franeNo %d! -- Jyt\n", numParticles, frameNo);
}

void PBFSolver::releaseFluidSource() {
	fluidsrcs.clear();
	cudaMemcpy(hType, dData.type, sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hInvMass, dData.invMass, sizeof(float)*numParticles, cudaMemcpyDeviceToHost);

	for (int i=0; i<numParticles; i++) {
		if (hType[i]==TYPE_EMITTER) {
			hType[i] = TYPE_FLUID;
			hColor[i] = cfloat4(0.7, 0.75, 0.95, 0.8);
			hInvMass[i] = 1;
		}
	}
	cudaMemcpy(dData.type,  hType,  sizeof(int)*numParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color, hColor,  sizeof(cfloat4)*numParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(dData.invMass, hInvMass, sizeof(float)*numParticles, cudaMemcpyHostToDevice);
}








//================================================
//
//              SIMULATION DATA
//
//================================================

void PBFSolver::dumpRenderingData() {
	printf("Dumping rendering data at frame %d\n", frameNo);

	char filepath[1000];
	sprintf(filepath, ".\\dump\\r%03d.dat", frameNo);
	FILE* fp = fopen(filepath, "w+");

	if (fp==NULL)
		printf("error opening file\n");

	copy2host_full();

	fprintf(fp, "%d\n", numParticles);
	for (int i=0; i<numParticles; i++) {
		fprintf(fp, "%f %f %f ", hPos[i].x, hPos[i].y, hPos[i].z);
		fprintf(fp, "%f %f %f %f ", hColor[i].x, hColor[i].y, hColor[i].z, hColor[i].w);
		fprintf(fp, "%d %d\n", hType[i], hUniqueId[i]);
	}

	int trinum = trianglelist.size();
	fprintf(fp, "%d\n", trinum);
	for (int i=0; i<trinum; i++) {
		objtriangle& t = trianglelist[i];
		fprintf(fp,"%d %d %d ",t.plist[0], t.plist[1], t.plist[2]);
		fprintf(fp, "%d\n", t.objectId);
	}
	
	int objnum = objectvec.size();
	fprintf(fp, "%d\n", objnum);
	for (int i=0; i<objnum; i++) {
		SimulationObject& o = objectvec[i];
		// 6
		fprintf(fp, "%d %d %d %d %d %d ", o.type, o.id, o.pnum, o.connum, o.trinum, o.indexedvbosize);
		// 4
		fprintf(fp, "%d %d %d %d ", o.startpid, o.startconid, o.starttriid, o.startvboid);
		// 4
		fprintf(fp, "%f %d %d %d\n", o.nRT, o.bVolumeCorr, o.bInjectGas, o.bJetGas);
	}
	
	fclose(fp);
}

void PBFSolver::dumpSimulationData() {

	printf("Dumping simulation data at frame %d\n", frameNo);
	char filepath[1000];
	sprintf(filepath, ".\\dump\\%03d.dat", frameNo);
	FILE* fp = fopen(filepath, "wb");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}

	copy2host_full();

	cfloat3* writePos = hPos;
	if (bNormalizePos) {
		cfloat3 center(0, 0, 0);
		// Manipulate Data
		for (int i = 0; i < numParticles; i++)
			center += hPos[i];
		center /= numParticles;
		for (int i = 0; i < numParticles; i++)
			hPosWriteBuf[i] = hPos[i] - center;
		printf("The center of particles is: %f %f %f", center.x, center.y, center.z);
		writePos = hPosWriteBuf;
	}
	
	// Particle Data
	fwrite(&numParticles, sizeof(int), 1, fp);

	fwrite(writePos, sizeof(cfloat3),	numParticles, fp);
	fwrite(hColor, sizeof(cfloat4), numParticles, fp);
	fwrite(hVel, sizeof(cfloat3),	numParticles, fp);
	fwrite(hType, sizeof(int),		numParticles, fp);
	fwrite(hGroup, sizeof(int),		numParticles, fp);
	fwrite(hMass, sizeof(float),	numParticles, fp);
	fwrite(hInvMass, sizeof(float), numParticles, fp);
	fwrite(hUniqueId, sizeof(int),	numParticles, fp);
	fwrite(hJetFlag, sizeof(char),	numParticles, fp);

	// Edge Data
	int edgenum = edgeConstraints.size();
	fwrite(&edgenum, sizeof(int), 1, fp);
	fwrite(edgeConstraints.data(), sizeof(edgeConstraint), edgenum, fp);
	
	// Triangle Data
	int trinum = trianglelist.size();
	fwrite(&trinum, sizeof(int), 1, fp);
	fwrite(trianglelist.data(), sizeof(objtriangle), trinum, fp);

	// OpenGL indexed VBO Data
	int indexvbosz = indexedVBO.size();
	fwrite(&indexvbosz, sizeof(int), 1, fp);
	fwrite(indexedVBO.data(), sizeof(unsigned int), indexvbosz, fp);

	// Simulation Object Data
	int objnum = objectvec.size();
	fwrite(&objnum,sizeof(int),1,fp);
	fwrite(objectvec.data(), sizeof(SimulationObject), objnum, fp);

	//fwrite(&hParam, sizeof(SimParam), 1, fp);
	fclose(fp);
}

void PBFSolver::dumpSimulationDataText() {

	printf("Dumping simulation data text at frame %d\n", frameNo);
	char filepath[1000];
	sprintf(filepath, ".\\dump\\%03d.dat", frameNo);
	FILE* fp = fopen(filepath, "w+");
	if (fp == NULL) {
		printf("error opening file\n"); return;
	}

	copy2host_full();

	cfloat3* writePos = hPos;
	if (bNormalizePos) {
		cfloat3 center(0, 0, 0);
		// Manipulate Data
		for (int i = 0; i < numParticles; i++)
			center += hPos[i];
		center /= numParticles;
		for (int i = 0; i < numParticles; i++)
			hPosWriteBuf[i] = hPos[i] - center;
		printf("The center of particles is: %f %f %f", center.x, center.y, center.z);
		writePos = hPosWriteBuf;
	}

	// Particle Data
	fprintf(fp, "%d\n", numParticles);
	for (int i=0; i<numParticles; i++) {
		fprintf(fp, "%f %f %f ",		writePos[i].x, writePos[i].y, writePos[i].z);
		fprintf(fp, "%f %f %f %f ", hColor[i].x, hColor[i].y, hColor[i].z, hColor[i].w);
		fprintf(fp, "%f %f %f ",	hVel[i].x, hVel[i].y, hVel[i].z);
		fprintf(fp, "%d ",			hType[i] );
		fprintf(fp, "%d ",hGroup[i]);
		fprintf(fp, "%f ",hMass[i]);
		fprintf(fp, "%f ",hInvMass[i]);
		fprintf(fp, "%d ", hUniqueId[i]);
		fprintf(fp, "%d\n", hJetFlag[i]);
	}

	

	// Edge Data
	int edgenum = edgeConstraints.size();
	fprintf(fp, "%d\n", edgenum);
	for (int i=0; i<edgenum; i++) {
		edgeConstraint& e = edgeConstraints[i];
		fprintf(fp, "%d %d %d %d ", e.p1, e.p2, e.p3, e.p4);
		fprintf(fp, "%f %f\n", e.L0, e.Phi0);
	}
	
	// Triangle Data
	int trinum = trianglelist.size();
	fprintf(fp, "%d\n", trinum);
	for (int i=0; i<trinum; i++) {
		objtriangle& t = trianglelist[i];
		fprintf(fp, "%d %d %d ", t.plist[0], t.plist[1], t.plist[2]);
		fprintf(fp, "%d\n", t.objectId);
	}

	// OpenGL indexed VBO Data
	int indexvbosz = indexedVBO.size();
	fprintf(fp, "%d\n", indexvbosz);
	for(int i=0; i<indexvbosz; i++)
		fprintf(fp, "%d\n", indexedVBO[i]);
	
	// Simulation Object Data
	int objnum = objectvec.size();
	fprintf(fp, "%d\n", objnum);
	for (int i=0; i<objnum; i++) {
		SimulationObject& o = objectvec[i];
		// 6
		fprintf(fp, "%d %d %d %d %d %d ", o.type, o.id, o.pnum, o.connum, o.trinum, o.indexedvbosize); 
		// 4
		fprintf(fp, "%d %d %d %d ", o.startpid, o.startconid, o.starttriid, o.startvboid);
		// 4
		fprintf(fp, "%f %d %d %d\n", o.nRT, o.bVolumeCorr, o.bInjectGas, o.bJetGas);
	}
	//fwrite(&hParam, sizeof(SimParam), 1, fp);
	fclose(fp);
}

//void PBFSolver::loadSimulationData(char* filepath) {
//	FILE* fp = fopen(filepath, "rb");
//	
//	//particle
//
//	int loadpnum;
//	fread(&loadpnum, sizeof(int), 1, fp);
//	int tmppnum = numParticles;
//	numParticles += loadpnum;
//
//	fread(hPos + tmppnum,	sizeof(cfloat3),loadpnum, fp);
//	fread(hColor + tmppnum, sizeof(cfloat4), loadpnum, fp);
//	fread(hVel+tmppnum,		sizeof(cfloat3), loadpnum, fp);
//	fread(hType + tmppnum,	sizeof(int),	loadpnum, fp);
//	fread(hMass + tmppnum,	sizeof(float), loadpnum, fp);
//	fread(hInvMass + tmppnum,	sizeof(float), loadpnum, fp);
//	fread(hUniqueId + tmppnum,	sizeof(int), loadpnum, fp);
//	fread(hJetFlag + tmppnum, sizeof(char), loadpnum, fp);
//	//edge
//
//	int tmpedgenum = edgeConstraints.size();
//	int loadedgenum;
//	fread(&loadedgenum, sizeof(int), 1, fp);
//	edgeConstraints.resize(tmpedgenum+loadedgenum);
//	
//	fread(edgeConstraints.data()+tmpedgenum, sizeof(edgeConstraint), loadedgenum, fp);
//	for (int i=tmpedgenum; i<tmpedgenum+loadedgenum; i++) {
//		edgeConstraints[i].p1 += tmppnum;
//		edgeConstraints[i].p2 += tmppnum;
//		edgeConstraints[i].p3 += tmppnum;
//		edgeConstraints[i].p4 += tmppnum;
//	}
//
//	//triangle
//	int tmptrianglenum = trianglelist.size();
//	int loadtrianglenum;
//	fread(&loadtrianglenum, sizeof(int),1,fp);
//	trianglelist.resize(tmptrianglenum+loadtrianglenum);
//	fread(trianglelist.data()+tmptrianglenum, sizeof(objtriangle), loadtrianglenum, fp);
//	for (int i=tmptrianglenum; i<tmptrianglenum+loadtrianglenum; i++) {
//		trianglelist[i].plist[0] += tmppnum;
//		trianglelist[i].plist[1] += tmppnum;
//		trianglelist[i].plist[2] += tmppnum;
//	}
//
//	int tmpIndexVBOSz = indexedVBO.size();
//	int loadIndexVBOSz;
//	fread(&loadIndexVBOSz, sizeof(int), 1, fp);
//	indexedVBO.resize(tmpIndexVBOSz + loadIndexVBOSz);
//	fread(indexedVBO.data()+tmpIndexVBOSz, sizeof(unsigned int), loadIndexVBOSz, fp);
//	for(int i=tmpIndexVBOSz; i<tmpIndexVBOSz+loadIndexVBOSz; i++)
//		indexedVBO[i] += tmppnum;
//
//	//simulation object
//	int tmpobjectnum = objectvec.size();
//	int loadobjectnum;
//	fread(&loadobjectnum, sizeof(int),1,fp);
//	objectvec.resize(tmpobjectnum+loadobjectnum);
//	fread(objectvec.data()+tmpobjectnum, sizeof(SimulationObject), loadobjectnum, fp);
//	for (int i=tmpobjectnum; i<tmpobjectnum+loadobjectnum; i++) {
//		objectvec[i].startpid += tmppnum;
//		objectvec[i].startconid += tmpedgenum;
//		objectvec[i].starttriid += tmptrianglenum;
//		objectvec[i].id = i;
//		objectvec[i].startvboid += tmpIndexVBOSz;
//	}
//
//	fread(&hParam, sizeof(SimParam), 1, fp);
//
//}

void PBFSolver::loadSimulationData(char* filepath) {
	cmat4 mm = IDENTITY_MAT;
	loadSimulationData(filepath, mm);
}

void PBFSolver::loadSimulationData(char* filepath, cmat4& materialMat) {
	FILE* fp = fopen(filepath, "rb");

	//particle

	int loadpnum;
	fread(&loadpnum, sizeof(int), 1, fp);
	int tmppnum = numParticles;
	numParticles += loadpnum;

	fread(hPos + tmppnum, sizeof(cfloat3), loadpnum, fp);
	fread(hColor + tmppnum, sizeof(cfloat4), loadpnum, fp);
	fread(hVel+tmppnum, sizeof(cfloat3), loadpnum, fp);
	fread(hType + tmppnum, sizeof(int), loadpnum, fp);
	fread(hGroup + tmppnum, sizeof(int), loadpnum, fp);
	fread(hMass + tmppnum, sizeof(float), loadpnum, fp);
	fread(hInvMass + tmppnum, sizeof(float), loadpnum, fp);
	fread(hUniqueId + tmppnum, sizeof(int), loadpnum, fp);
	fread(hJetFlag + tmppnum, sizeof(char), loadpnum, fp);
	
	for (int i=tmppnum; i<tmppnum+loadpnum; i++) {
		cfloat4 p(hPos[i].x, hPos[i].y, hPos[i].z, 1);
		p = materialMat * p;
		hPos[i].Set(p.x, p.y, p.z);
		hUniqueId[i] += tmppnum;
	}

	
	//edge
	int tmpedgenum = edgeConstraints.size();
	int loadedgenum;
	fread(&loadedgenum, sizeof(int), 1, fp);
	edgeConstraints.resize(tmpedgenum+loadedgenum);

	fread(edgeConstraints.data()+tmpedgenum, sizeof(edgeConstraint), loadedgenum, fp);
	for (int i=tmpedgenum; i<tmpedgenum+loadedgenum; i++) {
		edgeConstraints[i].p1 += tmppnum;
		edgeConstraints[i].p2 += tmppnum;
		edgeConstraints[i].p3 += tmppnum;
		edgeConstraints[i].p4 += tmppnum;
	}

	//triangle
	int tmptrianglenum = trianglelist.size();
	int loadtrianglenum;
	fread(&loadtrianglenum, sizeof(int), 1, fp);
	trianglelist.resize(tmptrianglenum+loadtrianglenum);
	fread(trianglelist.data()+tmptrianglenum, sizeof(objtriangle), loadtrianglenum, fp);
	for (int i=tmptrianglenum; i<tmptrianglenum+loadtrianglenum; i++) {
		trianglelist[i].plist[0] += tmppnum;
		trianglelist[i].plist[1] += tmppnum;
		trianglelist[i].plist[2] += tmppnum;
		trianglelist[i].objectId += objectvec.size();
	}

	int tmpIndexVBOSz = indexedVBO.size();
	int loadIndexVBOSz;
	fread(&loadIndexVBOSz, sizeof(int), 1, fp);
	indexedVBO.resize(tmpIndexVBOSz + loadIndexVBOSz);
	fread(indexedVBO.data()+tmpIndexVBOSz, sizeof(unsigned int), loadIndexVBOSz, fp);
	for (int i=tmpIndexVBOSz; i<tmpIndexVBOSz+loadIndexVBOSz; i++)
		indexedVBO[i] += tmppnum;

	//simulation object
	int tmpobjectnum = objectvec.size();
	int loadobjectnum;
	fread(&loadobjectnum, sizeof(int), 1, fp);
	objectvec.resize(tmpobjectnum+loadobjectnum);
	fread(objectvec.data()+tmpobjectnum, sizeof(SimulationObject), loadobjectnum, fp);
	for (int i=tmpobjectnum; i<tmpobjectnum+loadobjectnum; i++) {
		objectvec[i].startpid += tmppnum;
		objectvec[i].startconid += tmpedgenum;
		objectvec[i].starttriid += tmptrianglenum;
		objectvec[i].id = i;
		objectvec[i].startvboid += tmpIndexVBOSz;
	}

	//fread(&hParam, sizeof(SimParam), 1, fp);
	fclose(fp);
}

void PBFSolver::loadSimulationDataText(char* filepath, cmat4& materialMat) {
	FILE* fp = fopen(filepath, "r");

	//particle

	int loadpnum;
	fscanf(fp, "%d\n", &loadpnum);
	int tmppnum = numParticles;
	numParticles += loadpnum;
	for (int i=0; i<loadpnum; i++) {
		int pid = tmppnum+i;
		fscanf(fp, "%f %f %f ",		&hPos[pid].x, &hPos[pid].y, &hPos[pid].z);
		fscanf(fp, "%f %f %f %f ",	&hColor[pid].x, &hColor[pid].y, &hColor[pid].z, &hColor[pid].w);
		fscanf(fp, "%f %f %f ",		&hVel[pid].x, &hVel[pid].y, &hVel[pid].z);
		fscanf(fp, "%d ",  &hType[pid]);
		fscanf(fp, "%d ",  &hGroup[pid]);
		fscanf(fp, "%f ",  &hMass[pid]);
		fscanf(fp, "%f ",  &hInvMass[pid]);
		fscanf(fp, "%d ",  &hUniqueId[pid]);
		fscanf(fp, "%d\n", &hJetFlag[pid]);

		cfloat4 p(hPos[pid].x, hPos[pid].y, hPos[pid].z, 1);
		p = materialMat * p;
		hPos[pid].Set(p.x, p.y, p.z);
		hUniqueId[pid] += tmppnum;
	}

	//edge
	int tmpedgenum = edgeConstraints.size();
	int loadedgenum;
	fscanf(fp, "%d\n", &loadedgenum);
	edgeConstraints.resize(tmpedgenum+loadedgenum);
	for (int i=0; i<loadedgenum; i++) {
		edgeConstraint& e = edgeConstraints[tmpedgenum + i];
		fscanf(fp, "%d %d %d %d",	&e.p1, &e.p2, &e.p3, &e.p4);
		fscanf(fp, "%f %f\n",		&e.L0, &e.Phi0);
		e.p1 += tmppnum;
		e.p2 += tmppnum;
		e.p3 += tmppnum;
		e.p4 += tmppnum;
	}

	//triangle
	int tmptrianglenum = trianglelist.size();
	int loadtrianglenum;
	fscanf(fp, "%d\n",&loadtrianglenum);
	trianglelist.resize(tmptrianglenum+loadtrianglenum);

	for (int i=0; i<loadtrianglenum; i++) {
		objtriangle& t = trianglelist[ tmptrianglenum+i ];
		fscanf(fp, "%d %d %d %d\n",&t.plist[0], &t.plist[1], &t.plist[2], &t.objectId);
		t.plist[0] += tmppnum;
		t.plist[1] += tmppnum;
		t.plist[2] += tmppnum;
		t.objectId += objectvec.size();
	}

	int tmpIndexVBOSz = indexedVBO.size();
	int loadIndexVBOSz;
	fscanf(fp, "%d\n",&loadIndexVBOSz);
	indexedVBO.resize(tmpIndexVBOSz + loadIndexVBOSz);
	
	for (int i=0; i<loadIndexVBOSz; i++) {
		fscanf(fp,"%d\n", &indexedVBO[tmpIndexVBOSz+i]);
		indexedVBO[tmpIndexVBOSz+i] += tmppnum;
	}

	//simulation object
	int tmpobjectnum = objectvec.size();
	int loadobjectnum;
	fscanf(fp, "%d\n",&loadobjectnum);
	objectvec.resize(tmpobjectnum+loadobjectnum);
	for (int i=0; i<loadobjectnum; i++) {
		SimulationObject& o = objectvec[tmpobjectnum+i];

		// 6
		fscanf(fp, "%u %d %d %d %d %d ", &o.type, &o.id, &o.pnum, &o.connum, &o.trinum, &o.indexedvbosize);
		// 4
		fscanf(fp, "%d %d %d %d ", &o.startpid, &o.startconid, &o.starttriid, &o.startvboid);
		// 4
		fscanf(fp, "%f %u %u %u\n", &o.nRT, &o.bVolumeCorr, &o.bInjectGas, &o.bJetGas);

		o.startpid += tmppnum;
		o.startconid += tmpedgenum;
		o.starttriid += tmptrianglenum;
		o.id += tmpobjectnum;
		o.startvboid += tmpIndexVBOSz;
	}

	//fread(&hParam, sizeof(SimParam), 1, fp);
	fclose(fp);
}

void PBFSolver::setupHostBuffer() {
	int maxNumP = hParam.maxpnum;

	hPos =		(cfloat3*)malloc(sizeof(cfloat3)*maxNumP);
	hColor =	(cfloat4*)malloc(sizeof(cfloat4)*maxNumP);
	hVel =		(cfloat3*)malloc(sizeof(cfloat3)*maxNumP);
	hType =		(int*)malloc(sizeof(int)*maxNumP);
	hGroup =    (int*)malloc(sizeof(int)*maxNumP);
	hMass =		(float*)malloc(sizeof(float)*maxNumP);
	hInvMass =	(float*)malloc(sizeof(float)*maxNumP);
	hIndexTable =	(int*)malloc(sizeof(int)*maxNumP);
	hUniqueId =		(int*)malloc(sizeof(int)*maxNumP);
	hJetFlag = (char*)malloc(sizeof(char)*maxNumP);
	hAbsorbBuf = (float*)malloc(sizeof(float)*maxNumP);
	hDripBuf = (float*)malloc(sizeof(float)*maxNumP);

	hPosRender =	(cfloat3*)malloc(sizeof(cfloat3)*maxNumP);
	hColorRender =	(cfloat4*)malloc(sizeof(cfloat4)*maxNumP);
	hPosWriteBuf =  (cfloat3*)malloc(sizeof(cfloat3)*maxNumP);

	//int maxNumT = hParam.maxtrianglenum;
	//hBaryCenter = (cfloat3*)malloc(sizeof(cfloat3)*maxNumT);
	//hBaryCenterTriangleId = (int*)malloc(sizeof(int)*maxNumT);

}

void PBFSolver::setupDeviceBuffer() {

	//particle
	int maxpnum = hParam.maxpnum;
	int maxconstraintnum = hParam.maxconstraintnum;
	int maxtrianglenum = hParam.maxtrianglenum;
	int maxobjnum = hParam.maxobjnum;

	cudaMalloc(&dData.pos,		maxpnum * sizeof(float3));
	cudaMalloc(&dData.oldPos,	maxpnum * sizeof(float3));
	cudaMalloc(&dData.deltaPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.avgDeltaPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.vel,		maxpnum * sizeof(float3));
	cudaMalloc(&dData.normal, maxpnum * sizeof(float3));

	cudaMalloc(&dData.color, maxpnum * sizeof(cfloat4));

	cudaMalloc(&dData.lambda ,	maxpnum * sizeof(float));
	
	cudaMalloc(&dData.type,		maxpnum * sizeof(int));
	cudaMalloc(&dData.group,	maxpnum * sizeof(int));
	cudaMalloc(&dData.uniqueId, maxpnum * sizeof(int));
	cudaMalloc(&dData.numCons, maxpnum * sizeof(int));
	cudaMalloc(&dData.jetFlag, maxpnum * sizeof(char));

	cudaMalloc(&dData.invMass,	maxpnum * sizeof(float));
	cudaMalloc(&dData.mass,     maxpnum * sizeof(float));
	cudaMalloc(&dData.absorbBuf, maxpnum * sizeof(float));
	cudaMalloc(&dData.dripBuf,   maxpnum * sizeof(float));
	cudaMalloc(&dData.normalizeAbsorb, maxpnum * sizeof(float));
	cudaMalloc(&dData.normalizeDrip,   maxpnum * sizeof(float));
	cudaMalloc(&dData.deltaAbsorb, maxpnum * sizeof(float));
	cudaMalloc(&dData.deltaDrip, maxpnum * sizeof(float));

	cudaMalloc(&dData.sortedPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedOldPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedVel, maxpnum * sizeof(float3));
	cudaMalloc(&dData.sortedColor, maxpnum * sizeof(cfloat4));
	cudaMalloc(&dData.sortedInvMass, maxpnum * sizeof(float));
	cudaMalloc(&dData.sortedMass, maxpnum * sizeof(float));
	cudaMalloc(&dData.sortedType, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedGroup, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedUniqueId, maxpnum * sizeof(int));
	cudaMalloc(&dData.sortedJetFlag, maxpnum * sizeof(char));
	cudaMalloc(&dData.sortedAbsorbBuf, maxpnum * sizeof(float));
	cudaMalloc(&dData.sortedDripBuf, maxpnum * sizeof(float));
	cudaMalloc(&dData.indexTable, maxpnum * sizeof(int));

	//edge data
	cudaMalloc(&dData.edgeCons, maxconstraintnum * sizeof(edgeConstraint));
	cudaMalloc(&dData.edgeConsVar, maxconstraintnum*sizeof(edgeConsVar));

	//triangle data
	cudaMalloc(&dData.triangles,	maxtrianglenum * sizeof(objtriangle));
	cudaMalloc(&dData.facetVol,		maxtrianglenum * sizeof(float));
	cudaMalloc(&dData.facetArea,	maxtrianglenum * sizeof(float));
	cudaMalloc(&dData.baryCenter,					maxtrianglenum * sizeof(cfloat3));
	cudaMalloc(&dData.baryCenterTriangleId,			maxtrianglenum * sizeof(int));
	cudaMalloc(&dData.sortedBaryCenter,				maxtrianglenum * sizeof(cfloat3));
	cudaMalloc(&dData.sortedBaryCenterTriangleId,	maxtrianglenum * sizeof(int));

	//object data
	cudaMalloc(&dData.objs, maxobjnum * sizeof(SimulationObject));

	int glen = hParam.gridres.prod();
	
	cudaMalloc(&dData.particleHash,	maxpnum * sizeof(int));
	cudaMalloc(&dData.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&dData.gridCellStart,	glen * sizeof(int));
	cudaMalloc(&dData.gridCellEnd,		glen * sizeof(int));
	cudaMalloc(&dData.gridCellCollisionFlag, glen * sizeof(int));

	//cudaMalloc(&dData.baryCenterHash, maxtrianglenum * sizeof(int));
	//cudaMalloc(&dData.baryCenterIndex, maxtrianglenum * sizeof(int));
	//cudaMalloc(&dData.gridCellStartBaryCenter, glen * sizeof(int));
	//cudaMalloc(&dData.gridCellEndBaryCenter, glen * sizeof(int));
}




#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "pbfsolver.h"
#include "catpaw/cpXMLHelper.h"


extern SimParam hParam;

#include <algorithm>
using namespace std;

inline float rand_in_range(float min, float max) {
	return min + (float)rand()/RAND_MAX * (max-min);
}

void PBFSolver::copy2Device() {
	
	int constraintnum = edgeConstraints.size();
	int trianglenum = trianglelist.size();
	printf("constraint number: %d\n", constraintnum);

	cudaMemcpy(dData.pos,	hPos,		numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.vel,	hVel,		numParticles * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color, hColor,		numParticles * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.type,	hType,		numParticles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.group, hGroup,		numParticles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.mass,  hMass,      numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.invMass,	hInvMass,	numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.uniqueId,	hUniqueId,	numParticles * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.jetFlag,   hJetFlag,   numParticles * sizeof(char), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.dripBuf, hDripBuf, numParticles * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.absorbBuf, hAbsorbBuf, numParticles * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(dData.edgeCons,	edgeConstraints.data(), constraintnum * sizeof(edgeConstraint), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.triangles, trianglelist.data(),	trianglenum * sizeof(objtriangle), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.objs,		objectvec.data(),		objectvec.size()*sizeof(SimulationObject), cudaMemcpyHostToDevice);

	copyDeviceBuffer();
}


void PBFSolver::removeInvalidParticles() {
	for (int i = 0; i<numParticles; i++) {
		if (hType[i] != TYPE_FLUID)
			continue;
		else if (hMass[i] < EPSILON) //invalid
		{
			hType[i] = TYPE_NULL;
			hPos[i].Set(-9999, -9999, -9999);
			numParticlesDeleted++;
		}
	}
}

void PBFSolver::emitWaterFromCloth() {
	
	int start = numParticles;
	int pid = start;

	for (int i = 0; i<numParticles; i++) {
		if (hType[i] != TYPE_CLOTH)
			continue;
		if (hDripBuf[i] > hParam.emitThres) //invalid
		{
			if (pid == hParam.maxpnum)
				return;
			if (pid == numParticles) {
				hUniqueId[pid] = numParticles++;
				hJetFlag[pid] = 0;
			}
			else {
				numParticlesDeleted--;
			}
			
			hPos[pid] = hPos[i]+cfloat3(0,-1,0);
			hVel[pid] = cfloat3(0,0,0);
			hColor[pid] = cfloat4(0.7, 0.75, 0.95, 0.8);
			hMass[pid] = hParam.emitThres;
			hDripBuf[i] -= hParam.emitThres;

			hType[pid] = TYPE_FLUID;
			hInvMass[pid] = 1 / hMass[pid];
			pid++;
		}
	}

	copy2Device_partial(start, pid);
}

void PBFSolver::copy2Host() {
	
	cudaMemcpy(hMass,		dData.mass,			sizeof(float)*numParticles,		cudaMemcpyDeviceToHost);
	cudaMemcpy(hPos,		dData.pos,			sizeof(cfloat3)*numParticles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel,		dData.vel,			sizeof(cfloat3)*numParticles,	cudaMemcpyDeviceToHost);
	cudaMemcpy(hUniqueId,	dData.uniqueId,		sizeof(int)*numParticles,		cudaMemcpyDeviceToHost);
	cudaMemcpy(hIndexTable, dData.indexTable,	hParam.maxpnum * sizeof(int),	cudaMemcpyDeviceToHost);
	cudaMemcpy(hColor,		dData.color,		numParticles * sizeof(cfloat4), cudaMemcpyDeviceToHost);


	if (bModelPorous) {
		cudaMemcpy(hType,		dData.type,		sizeof(int)*numParticles,	cudaMemcpyDeviceToHost);
		cudaMemcpy(hDripBuf,	dData.dripBuf,	sizeof(float)*numParticles, cudaMemcpyDeviceToHost);

		removeInvalidParticles();
		emitWaterFromCloth();
	
		cudaMemcpy(dData.mass,		hMass,		sizeof(float)*numParticles,		cudaMemcpyHostToDevice);
		cudaMemcpy(dData.type,		hType,		sizeof(int)*numParticles,		cudaMemcpyHostToDevice);
		cudaMemcpy(dData.pos,		hPos,		sizeof(cfloat3)*numParticles,	cudaMemcpyHostToDevice);
		cudaMemcpy(dData.dripBuf,	hDripBuf,	sizeof(float)*numParticles,		cudaMemcpyHostToDevice);
	}

	//Rendering
	for (int i = 0; i < numParticles; i++) {
		hPosRender[i] = hPos[hIndexTable[i]];
		hColorRender[i] = hColor[hIndexTable[i]];
	}

}

#define GRID_UNDEF 99999999

void PBFSolver::sortBaryCenter() {
	
	sortBaryCenterCUDA(dData, numTriangles, numGridCells);

	cudaMemcpy(dData.baryCenter,			dData.sortedBaryCenter,				numTriangles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.baryCenterTriangleId,	dData.sortedBaryCenterTriangleId,	numTriangles * sizeof(int),		cudaMemcpyDeviceToDevice);
}

void PBFSolver::sort() {
	calcHash(dData,numParticles);

	sortParticle(dData,numParticles);

	cudaMemset(dData.gridCellCollisionFlag, 0, sizeof(char)*numGridCells);

	reorderDataAndFindCellStart(dData,numParticles,numGridCells);

	cudaMemcpy(dData.pos,		dData.sortedPos,		numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.oldPos,	dData.sortedOldPos,		numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.vel,		dData.sortedVel,		numParticles * sizeof(cfloat3), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.color,		dData.sortedColor,		numParticles * sizeof(cfloat4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.type,		dData.sortedType,		numParticles * sizeof(int),		cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.group,		dData.sortedGroup,		numParticles * sizeof(int),		cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.invMass,	dData.sortedInvMass,	numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.mass,		dData.sortedMass,		numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.uniqueId,	dData.sortedUniqueId,	numParticles * sizeof(int),		cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.jetFlag,	dData.sortedJetFlag,	numParticles * sizeof(char),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.absorbBuf, dData.sortedAbsorbBuf,	numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
	cudaMemcpy(dData.dripBuf,   dData.sortedDripBuf,	numParticles * sizeof(float),	cudaMemcpyDeviceToDevice);
}

void PBFSolver::solvePBF() {
	//Solve fluid constraints.
	for (int i = 0; i < 3; i++) {
		calcLambda(dData, numParticles);
		calcDeltaPos(dData, numParticles);
		updatePos(dData, numParticles);
	}

	//collision detection
	detectCollision(dData, numParticles);
	updatePos(dData, numParticles);
	

	//Update Velocity.
	updateVel(dData, numParticles, hParam.dt);
	applyXSPH(dData, numParticles);

}


void PBFSolver::Step() {
	if (numParticles > 0) {

		predictPosition( dData, hParam.dt, numParticles);

		sort();

		solvePBF();

		copy2Host();
	}

	//'e' emit particle
	if (bEmitParticle) {
		fluidSrcEmit();
	}

	frameNo++;
	time += hParam.dt;
}

void PBFSolver::loadFluidVols(XMLElement* sceneEle) {
	Tinyxml_Reader reader;
	fluidvol fv;
	XMLElement* fvele = sceneEle->FirstChildElement("Vol");
	while (fvele!=NULL && strcmp(fvele->Name(), "Vol")==0) {
		reader.Use(fvele);
		fv.xmin = reader.GetFloat3("VolMin");
		fv.xmax = reader.GetFloat3("VolMax");
		reader.GetFloatN(fv.volfrac, hParam.maxtypenum, "VolFrac");
		fluidvols.push_back(fv);
		fvele = fvele->NextSiblingElement();
	}
}


void PBFSolver::HandleKeyEvent(char key) {
	switch (key) {
	case 'b':
		dumpSimulationDataText();
		break;
	case 'r':
		dumpRenderingData();
		break;
	case 'e':
		bEmitParticle = !bEmitParticle;
		if (bEmitParticle)
			printf("Start to emit particles.\n");
		else
			printf("Stop emitting particles.\n");
		break;
	case 'f':
		bReleaseSource = true;
		break;
	case 'i':
		//solver->bInjectGas = !solver->bInjectGas;
		for (int i=0; i<objectvec.size(); i++)
			objectvec[i].bInjectGas = !objectvec[i].bInjectGas;
		if (objectvec[0].bInjectGas) {
			printf("Start to inject gas.\n");
		}
		else
			printf("Stop injecting particles.\n");
		break;
	case 'j':
		for (int i=0; i<objectvec.size(); i++)
			objectvec[i].bJetGas = !objectvec[i].bJetGas;
		if (objectvec[0].bJetGas) {
			printf("Start to jet gas.\n");
		}
		else
			printf("Stop jetting particles.\n");
		break;
	}
}








void PBFSolver::copy2host_full() {
	//sync data
	cudaMemcpy(hPos, dData.pos,		sizeof(cfloat3)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hColor, dData.color, sizeof(cfloat4)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hVel, dData.vel,		sizeof(cfloat3)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hType, dData.type,	sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hInvMass, dData.invMass,		sizeof(float)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hUniqueId, dData.uniqueId,	sizeof(int)*numParticles, cudaMemcpyDeviceToHost);
	cudaMemcpy(hJetFlag, dData.jetFlag, sizeof(char)*numParticles, cudaMemcpyDeviceToHost);
}


void PBFSolver::emitParticle() {
	//lock source facet
	cfloat3 norm;
	cfloat3 srcpos;
	
	cfloat3 x1,x2,x3;
	x1 = hPos[hIndexTable[799]];
	x2 = hPos[hIndexTable[792]];
	x3 = hPos[hIndexTable[798]];

	cfloat3 x1x2, x1x3;
	x1x2 = x2-x1;
	x1x3 = x3-x1;
	norm = cross(x1x2, x1x3); //outward
	norm  = norm * (-1); //inward

	srcpos = (x1+x2+x3)/3.0f; //barycentric center
	//srcpos = cfloat3((float)rand()/RAND_MAX,10, (float)rand()/RAND_MAX);
	//norm = cfloat3(0,-1,0);
	float area = norm.Norm()*3;
	norm = norm / norm.Norm();
	float speed = 30;
	
	int frameinterval = roundf( 1 / (area*speed*hParam.dt));
	int start = numParticles;
	//frameinterval = 10;

	if (frameNo % 1==0) {
		int pid = addDefaultParticle();
		hPos[pid] = srcpos + norm*0.5;
		hVel[pid] = cfloat3(0,-0.1,0);//norm;// * speed;
		hColor[pid] = cfloat4(0.7, 0.75, 0.95, 0.8);
		hMass[pid] = hParam.restdensity*pow(hParam.spacing, 3);
		hType[pid] = TYPE_FLUID;
		hInvMass[pid] = 1 / hMass[pid];
	}

	copy2Device_partial(start, numParticles);
}

void PBFSolver::copy2Device_partial(int begin, int end) {
	int len = end - begin;
	
	cudaMemcpy(dData.pos+begin,		hPos+begin,		len * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.vel+begin,		hVel+begin,		len * sizeof(cfloat3), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color+begin,	hColor + begin,		len * sizeof(cfloat4), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.type+begin,	hType + begin,		len * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.mass+begin,	hMass + begin,		len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.invMass+begin,	hInvMass + begin,	len * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dData.uniqueId+begin,	hUniqueId + begin, len * sizeof(int), cudaMemcpyHostToDevice);
	
}



void PBFSolver::addCloth() {
	ObjBuilder builder;
	int height, width;
	height=30;
	width=30;

	ObjContainer* oc =  builder.build_rectangular(height, width, 1);
	oc->buildEdgeList();

	int temppnum = numParticles;

	cmat4 materialmat;
	materialmat = IDENTITY_MAT;
	//ScaleMatrix(materialmat, cfloat3(10,10,10));
	RotateAboutX(materialmat, 3.14159/2);
	TranslateMatrix(materialmat, cfloat3(0, 30, 0));

	loadClothObj(*oc, materialmat);

	//fix points

	fixedvec.push_back(temppnum+(height-1)*width); //top left
	fixedvec.push_back(temppnum+height*width - 1); //top right

	fixedvec.push_back(temppnum + 0); //bottom left
	fixedvec.push_back(temppnum + width-1); //bottom right

	for (int i=0; i<fixedvec.size(); i++) {
		hInvMass[fixedvec[i]] = 0;
	}

	//params
	runmode = RUN_GPU;
	bVolumeCorr = false;

	delete oc;
}

void PBFSolver::addMesh() {
	ObjBuilder builder;
	ObjContainer* oc;
	ObjReader or;

	int temppnum = numParticles;
	cmat4 materialmat;
	materialmat = IDENTITY_MAT;
	int c = 10;

	switch (c) {
	case 0:
		oc = or.readobj("model/fineball.obj");
		ScaleMatrix(materialmat, cfloat3(40, 40, 40));
		TranslateMatrix(materialmat, cfloat3(0, 20, 0));
		hParam.restvol = 0;
		bVolumeCorr = true;

		fixedvec.push_back(temppnum + 2672);
		fixedvec.push_back(temppnum + 2732);
		fixedvec.push_back(temppnum + 2577);

		break;

	case 1:
		oc = or.readobj("untitled.obj");
		ScaleMatrix(materialmat, cfloat3(0.3, 0.3, 0.3));
		TranslateMatrix(materialmat, cfloat3(0, 0, -15));
		break;
	case 2:
		oc = or.readobj("armadillo.obj");
		//oc = or.readobj("bunny.obj");
		ScaleMatrix(materialmat, cfloat3(0.15, 0.15, 0.15));
		TranslateMatrix(materialmat, cfloat3(20, 0, 0));
		break;
	case 3:
		oc = or.readobj("model/armadillo.obj");
		//oc = or.readobj("bunny.obj");
		ScaleMatrix(materialmat, cfloat3(15, 15, 15));
		hParam.restvol = 8000;
		TranslateMatrix(materialmat, cfloat3(20, 0, 0));
		break;
	}
	oc->buildEdgeList();

	TranslateMatrix(materialmat, cfloat3(0, 15, 0));

	loadClothObj(*oc, materialmat);

	//fix points
	//fixedvec.push_back(temppnum+(height-1)*width); //top left
	//fixedvec.push_back(temppnum+height*width - 1); //top right

	//fixedvec.push_back(temppnum + 508);
	//fixedvec.push_back(temppnum + 512);
	//fixedvec.push_back(temppnum + 513);



	//fixedvec.push_back(temppnum + 0); //bottom left
	//fixedvec.push_back(temppnum + 33); //bottom left
	//fixedvec.push_back(temppnum + 66); //bottom left
	//fixedvec.push_back(temppnum + width-1); //bottom right

	for (int i = 0; i < fixedvec.size(); i++) {
		//hInvMass[fixedvec[i]] = 0;
		hJetFlag[fixedvec[i]] = 1;
		hColor[fixedvec[i]].Set(1, 1, 1, 1);
	}


	//hParam.enable_volcorr = 0;

	delete oc;
}

void PBFSolver::addWaterBag() {
	ObjBuilder builder;
	ObjContainer* oc;
	ObjReader or;

	int temppnum = numParticles;
	cmat4 materialmat;


	oc = or.readobj("model/opencube32.obj");
	oc->buildEdgeList();

	for (int i=0; i<oc->pointlist.size(); i++) {
		if (abs(oc->pointlist[i].pos.y - 0.5) < 0.00001) {
			fixedvec.push_back(temppnum+i);
		}
	}

	materialmat = IDENTITY_MAT;
	float scale = 30;
	ScaleMatrix(materialmat, cfloat3(scale, scale, scale));
	TranslateMatrix(materialmat, cfloat3(0, 50, 0));



	loadClothObj(*oc, materialmat);

	for (int i = 0; i < fixedvec.size(); i++) {
		hInvMass[fixedvec[i]] = 0;
		//hJetFlag[fixedvec[i]] = 1;
		hColor[fixedvec[i]].Set(1, 1, 1, 1);
	}

	delete oc;
}



void PBFSolver::addBalloon() {

	ObjContainer* oc;
	ObjReader or;

	int temppnum = numParticles;
	cmat4 materialmat;
	materialmat = IDENTITY_MAT;

	oc = or.readobj("model/ball.obj");
	oc->buildEdgeList();

	float scale = 20;
	ScaleMatrix(materialmat, cfloat3(scale, scale, scale));
	TranslateMatrix(materialmat, cfloat3(0, 35, 0));
	loadClothObj(*oc, materialmat);

	int lockCenter = 304;
	for (int i=0; i<oc->trianglelist.size(); i++) {
		objtriangle& t = oc->trianglelist[i];
		if (t.plist[0]==lockCenter || t.plist[1]==lockCenter || t.plist[2]==lockCenter) {
			fixedvec.push_back(temppnum + t.plist[0]);
			fixedvec.push_back(temppnum + t.plist[1]);
			fixedvec.push_back(temppnum + t.plist[2]);
		}
	}
	for (int i = 0; i < fixedvec.size(); i++) {
		hInvMass[fixedvec[i]] = 0;
		//hJetFlag[fixedvec[i]] = 1;
		hColor[fixedvec[i]].Set(1, 1, 1, 1);
	}

	delete oc;
}

void PBFSolver::addBalloon(cmat4& materialMat) {

	ObjContainer* oc;
	ObjReader or;

	int temppnum = numParticles;


	oc = or.readobj("model/ball.obj");
	oc->buildEdgeList();

	loadClothObj(*oc, materialMat);

	bool bLock = false;
	if (bLock) {
		int lockCenter = 304;
		for (int i=0; i<oc->trianglelist.size(); i++) {
			objtriangle& t = oc->trianglelist[i];
			if (t.plist[0]==lockCenter || t.plist[1]==lockCenter || t.plist[2]==lockCenter) {
				fixedvec.push_back(temppnum + t.plist[0]);
				fixedvec.push_back(temppnum + t.plist[1]);
				fixedvec.push_back(temppnum + t.plist[2]);
			}
		}
		for (int i = 0; i < fixedvec.size(); i++) {
			hInvMass[fixedvec[i]] = 0;
			//hJetFlag[fixedvec[i]] = 1;
			hColor[fixedvec[i]].Set(1, 1, 1, 1);
		}
	}



	delete oc;
}


void PBFSolver::addArmadilo() {
	ObjBuilder builder;
	ObjContainer* oc;
	ObjReader or;

	int temppnum = numParticles;
	cmat4 materialmat;
	materialmat = IDENTITY_MAT;
	int c = 10;

	oc = or.readobj("model/armadillo.obj");
	oc->buildEdgeList();

	float scale = 20;
	scale = 15;

	ScaleMatrix(materialmat, cfloat3(scale, scale, scale));
	TranslateMatrix(materialmat, cfloat3(0, 20, 0));
	hParam.restvol = 500;

	int lockCenter = 304;
	for (int i=0; i<oc->trianglelist.size(); i++) {
		objtriangle& t = oc->trianglelist[i];
		if (t.plist[0]==lockCenter || t.plist[1]==lockCenter || t.plist[2]==lockCenter) {
			fixedvec.push_back(temppnum + t.plist[0]);
			fixedvec.push_back(temppnum + t.plist[1]);
			fixedvec.push_back(temppnum + t.plist[2]);
		}
	}

	TranslateMatrix(materialmat, cfloat3(0, 15, 0));

	loadClothObj(*oc, materialmat);

	for (int i = 0; i < fixedvec.size(); i++) {
		hInvMass[fixedvec[i]] = 0;
		//hJetFlag[fixedvec[i]] = 1;
		hColor[fixedvec[i]].Set(1, 1, 1, 1);
	}

	delete oc;
}




int PBFSolver::addDefaultParticle() {
	if (numParticles ==hParam.maxpnum)
		return -1;
	else {
		hMass[numParticles] = 0;
		hVel[numParticles].Set(0, 0, 0);
		hUniqueId[numParticles] = numParticles;
		hJetFlag[numParticles] = 0;
		hAbsorbBuf[numParticles] = 0;
		hDripBuf[numParticles] = 0;

		numParticles++;
		return numParticles-1;
	}
}

void PBFSolver::addfluidvolumes() {
	int addcount=0;

	for (int i=0; i<fluidvols.size(); i++) {
		cfloat3 xmin = fluidvols[i].xmin;
		cfloat3 xmax = fluidvols[i].xmax;
		float* vf = fluidvols[i].volfrac;
		float spacing = hParam.spacing;
		float pden = hParam.restdensity;
		float mp = spacing*spacing*spacing* pden;
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
					hInvMass[pid] = 1 / mp;
					hGroup[pid] = 0;
				}

		printf("added %d particles...\n", addcount);
	}
done_add:
	printf("added %d particles...\n", addcount);
}



//=============================================
//
//
//             BOUNDARY SETUP
//
//
//=============================================



//BOOKMARK: BOUNDARY PARTICLES

void inline read(FILE* fp, cfloat3& dst) {
	fscanf(fp, "%f,%f,%f", &dst.x, &dst.y, &dst.z);
}
void inline read(FILE* fp, float& dst) {
	fscanf(fp, "%f", &dst);
}

void PBFSolver::loadboundary(string boundfile) {

	printf("Loading boundaries.\n");

	FILE* fp = fopen(boundfile.c_str(), "r");
	if (fp==NULL) {
		printf("null boundary file");
		return;
	}

	char buffer[1000];
	cfloat3 min, max;
	while (true) {
		if (fscanf(fp, "%s", buffer)==EOF)
			break;
		if (!strcmp(buffer, "WALL")) {
			read(fp, min);
			read(fp, max);
			addwall(min, max);
		}
		if (!strcmp(buffer, "OPEN_BOX")) {
			read(fp, min);
			read(fp, max);
			float thickness;
			read(fp, thickness);
			addopenbox(min, max, thickness);
		}
	}
	return;
}

void PBFSolver::loadParticleSample(string filepath) {
	printf("Loading particle sample.\n");

	FILE* fp = fopen(filepath.c_str(), "r");
	if (fp==NULL) {
		printf("null boundary file");
		return;
	}

	int numP;
	fscanf(fp, "%d\n", &numP);
	cfloat3 tmpF;
	for (int i=0; i<numP; i++) {
		fscanf(fp, "%f %f %f\n", &tmpF.x, &tmpF.y, &tmpF.z);
		int pid = addDefaultParticle();
		hPos[pid] = tmpF;
		hColor[pid].Set(1, 1, 1, 0.5);
		hMass[pid] = 1;
		hInvMass[pid] = 0;
		hGroup[pid] = -1;
		hType[pid] = TYPE_BOUNDARY;
	}

	return;
}

void PBFSolver::addwall(cfloat3 min, cfloat3 max) {
	cfloat3 pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;

	float spacing = hParam.spacing;
	cntx = ceil((max.x-min.x) / spacing);
	cntz = ceil((max.z-min.z) / spacing);

	//rotation initialization

	int cnt = cntx * cntz;
	int xp, yp, zp, c2;
	float odd;

	dx = max.x-min.x;
	dy = max.y-min.y;
	dz = max.z-min.z;

	c2 = cnt/2;

	float randx[3]={0,0,0};
	float ranlen = 0.2;

	for (float y = min.y; y <= max.y; y += spacing) {
		for (int xz=0; xz < cnt; xz++) {
			x = min.x + (xz % int(cntx))*spacing;
			z = min.z + (xz / int(cntx))*spacing;

			int pid = addDefaultParticle();
			if (pid==-1)
				goto done_add;

			hPos[pid] = cfloat3(x, y, z);
			if (z>0)
				hColor[pid].Set(1, 1, 1, 0);
			else
				hColor[pid].Set(1, 1, 1, 0.2);

			hType[pid] = TYPE_BOUNDARY;
		}
	}
done_add:
	return;
}

void PBFSolver::addopenbox(cfloat3 min, cfloat3 max, float thickness) {
	//ground
	addwall(cfloat3(min.x, min.y-thickness, min.z), cfloat3(max.x, min.y, max.z));
	//front
	addwall(cfloat3(min.x-thickness, min.y, min.z-thickness), cfloat3(max.x+thickness, max.y, min.z));
	//back
	addwall(cfloat3(min.x-thickness, min.y, max.z), cfloat3(max.x+thickness, max.y, max.z+thickness));
	//left
	addwall(cfloat3(min.x-thickness, min.y, min.z), cfloat3(min.x, max.y, max.z));
	//right
	addwall(cfloat3(max.x, min.y, min.z), cfloat3(max.x+thickness, max.y, max.z));
}



int PBFSolver::loadClothObj(ObjContainer& oc, cmat4 materialMat) {
	SimulationObject obj;
	obj.type = TYPE_CLOTH;
	obj.id = objectvec.size();
	obj.pnum = oc.pointlist.size();
	obj.connum = oc.edgeConstraints.size();
	obj.trinum = oc.trianglelist.size();
	obj.bInjectGas = false;
	obj.bVolumeCorr = false;
	obj.bJetGas = false;

	//start from current particle number
	obj.startpid = numParticles;
	obj.startconid = edgeConstraints.size();
	obj.starttriid = trianglelist.size();

	//Load particle data
	for (int i = 0; i<obj.pnum; i++) {
		int pid = addDefaultParticle();

		//default arrays
		cfloat4 x(oc.pointlist[i].pos, 1);
		x = materialMat * x;

		hPos[pid].Set(x.x, x.y, x.z);
		hColor[pid].Set(0.8, 0.8, 0.4, 1);
		hType[pid] = TYPE_CLOTH;
		hGroup[pid] = obj.id + 1;

	}

	//Load Indexed VBO
	int trinum = oc.trianglelist.size();
	obj.startvboid = indexedVBO.size();
	obj.indexedvbosize = trinum * 3;
	indexedVBO.resize(indexedVBO.size() + trinum * 3);

	for (int i = 0; i<trinum; i++) {
		indexedVBO[obj.startvboid + i * 3] = oc.trianglelist[i].plist[0] + obj.startpid;
		indexedVBO[obj.startvboid + i * 3 + 1] = oc.trianglelist[i].plist[1] + obj.startpid;
		indexedVBO[obj.startvboid + i * 3 + 2] = oc.trianglelist[i].plist[2] + obj.startpid;
	}

	//Load Triangle List
	trianglelist.resize(trianglelist.size() + trinum);
	for (int i = 0; i<trinum; i++) {
		trianglelist[obj.starttriid + i] = oc.trianglelist[i];
		trianglelist[obj.starttriid + i].plist[0] += obj.startpid;
		trianglelist[obj.starttriid + i].plist[1] += obj.startpid;
		trianglelist[obj.starttriid + i].plist[2] += obj.startpid;
		trianglelist[obj.starttriid + i].objectId = obj.id;
		//hBaryCenterTriangleId[obj.starttriid + i] = obj.starttriid + i;
	}

	//Load Edge Constraint
	for (int i = 0; i<oc.edgeConstraints.size(); i++) {
		edgeConstraint& ec = oc.edgeConstraints[i];
		ec.p1 += obj.startpid;
		ec.p2 += obj.startpid;
		if (ec.p3 != -1) {
			ec.p3 += obj.startpid;
			ec.p4 += obj.startpid;
		}
		edgeConstraints.push_back(ec);
	}

	objectvec.push_back(obj);
	nRTvec.push_back(0);
	printf("Added No. %d Object with %d particles.\n", obj.id, obj.pnum);



	//initialize constraints
	//initialize mass
	for (int i = 0; i<oc.trianglelist.size(); i++) {
		int p1, p2, p3;
		p1 = oc.trianglelist[i].plist[0];
		p2 = oc.trianglelist[i].plist[1];
		p3 = oc.trianglelist[i].plist[2];
		cfloat3 a = hPos[obj.startpid + p1] - hPos[obj.startpid + p2];
		cfloat3 b = hPos[obj.startpid + p1] - hPos[obj.startpid + p3];
		float area = cross(a, b).Norm()*0.5;
		hMass[obj.startpid + p1] += area / 3.0f;
		hMass[obj.startpid + p2] += area / 3.0f;
		hMass[obj.startpid + p3] += area / 3.0f;
	}

	//get inverse mass
	for (int i = 0; i<obj.pnum; i++) {
		hMass[i + obj.startpid] *= hParam.cloth_density;
		hInvMass[i + obj.startpid] = 1.0f / hMass[i + obj.startpid];
	}


	//initialize constraint length
	float maxlen = 0;
	for (int i = 0; i<oc.edgeConstraints.size(); i++) {
		edgeConstraint& ec = edgeConstraints[obj.startconid + i];
		int p1 = ec.p1, p2 = ec.p2, p3 = ec.p3, p4 = ec.p4;
		cfloat3 p1p2, p1p3, p1p4;
		p1p2 = hPos[p2] - hPos[p1];

		if (p1p2.Norm()>maxlen)
			maxlen = p1p2.Norm();

		float L0 = p1p2.Norm();
		ec.L0 = L0;

		if (ec.t2<0)
			continue;

		p1p3 = hPos[p3] - hPos[p1];
		p1p4 = hPos[p4] - hPos[p1];
		cfloat3 temp = cross(p1p2, p1p3);
		temp = temp / temp.Norm();
		cfloat3 temp1 = cross(p1p2, p1p4);
		temp1 = temp1 / temp1.Norm();

		float d = dot(temp, temp1);
		d = fmin(1, fmax(d, -1));
		float Phi0 = acos(d);
		ec.Phi0 = Phi0;
	}
	printf("constraint maxlen is %f\n", maxlen);


	return objectvec.size()-1;
}





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
	fs.norm /= fs.norm.Norm();
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
		RotateAboutZ(mm, acos(fs.norm.y));
		cfloat3 xzmap = cfloat3(fs.norm.x, 0, fs.norm.z);
		if (xzmap.Norm()>EPSILON)
			RotateAboutY(mm, -acos(fs.norm.x /xzmap.Norm()));

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
		hVel[pid] = vArr[i];
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
	cudaMemcpy(dData.type, hType, sizeof(int)*numParticles, cudaMemcpyHostToDevice);
	cudaMemcpy(dData.color, hColor, sizeof(cfloat4)*numParticles, cudaMemcpyHostToDevice);
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
		fprintf(fp, "%d %d %d ", t.plist[0], t.plist[1], t.plist[2]);
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

	fwrite(writePos, sizeof(cfloat3), numParticles, fp);
	fwrite(hColor, sizeof(cfloat4), numParticles, fp);
	fwrite(hVel, sizeof(cfloat3), numParticles, fp);
	fwrite(hType, sizeof(int), numParticles, fp);
	fwrite(hGroup, sizeof(int), numParticles, fp);
	fwrite(hMass, sizeof(float), numParticles, fp);
	fwrite(hInvMass, sizeof(float), numParticles, fp);
	fwrite(hUniqueId, sizeof(int), numParticles, fp);
	fwrite(hJetFlag, sizeof(char), numParticles, fp);

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
	fwrite(&objnum, sizeof(int), 1, fp);
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
		fprintf(fp, "%f %f %f ", writePos[i].x, writePos[i].y, writePos[i].z);
		fprintf(fp, "%f %f %f %f ", hColor[i].x, hColor[i].y, hColor[i].z, hColor[i].w);
		fprintf(fp, "%f %f %f ", hVel[i].x, hVel[i].y, hVel[i].z);
		fprintf(fp, "%d ", hType[i]);
		fprintf(fp, "%d ", hGroup[i]);
		fprintf(fp, "%f ", hMass[i]);
		fprintf(fp, "%f ", hInvMass[i]);
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
	for (int i=0; i<indexvbosz; i++)
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
		fscanf(fp, "%f %f %f ", &hPos[pid].x, &hPos[pid].y, &hPos[pid].z);
		fscanf(fp, "%f %f %f %f ", &hColor[pid].x, &hColor[pid].y, &hColor[pid].z, &hColor[pid].w);
		fscanf(fp, "%f %f %f ", &hVel[pid].x, &hVel[pid].y, &hVel[pid].z);
		fscanf(fp, "%d ", &hType[pid]);
		fscanf(fp, "%d ", &hGroup[pid]);
		fscanf(fp, "%f ", &hMass[pid]);
		fscanf(fp, "%f ", &hInvMass[pid]);
		fscanf(fp, "%d ", &hUniqueId[pid]);
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
		fscanf(fp, "%d %d %d %d", &e.p1, &e.p2, &e.p3, &e.p4);
		fscanf(fp, "%f %f\n", &e.L0, &e.Phi0);
		e.p1 += tmppnum;
		e.p2 += tmppnum;
		e.p3 += tmppnum;
		e.p4 += tmppnum;
	}

	//triangle
	int tmptrianglenum = trianglelist.size();
	int loadtrianglenum;
	fscanf(fp, "%d\n", &loadtrianglenum);
	trianglelist.resize(tmptrianglenum+loadtrianglenum);

	for (int i=0; i<loadtrianglenum; i++) {
		objtriangle& t = trianglelist[tmptrianglenum+i];
		fscanf(fp, "%d %d %d %d\n", &t.plist[0], &t.plist[1], &t.plist[2], &t.objectId);
		t.plist[0] += tmppnum;
		t.plist[1] += tmppnum;
		t.plist[2] += tmppnum;
		t.objectId += objectvec.size();
	}

	int tmpIndexVBOSz = indexedVBO.size();
	int loadIndexVBOSz;
	fscanf(fp, "%d\n", &loadIndexVBOSz);
	indexedVBO.resize(tmpIndexVBOSz + loadIndexVBOSz);

	for (int i=0; i<loadIndexVBOSz; i++) {
		fscanf(fp, "%d\n", &indexedVBO[tmpIndexVBOSz+i]);
		indexedVBO[tmpIndexVBOSz+i] += tmppnum;
	}

	//simulation object
	int tmpobjectnum = objectvec.size();
	int loadobjectnum;
	fscanf(fp, "%d\n", &loadobjectnum);
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

	cudaMalloc(&dData.pos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.oldPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.deltaPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.avgDeltaPos, maxpnum * sizeof(float3));
	cudaMalloc(&dData.vel, maxpnum * sizeof(float3));
	cudaMalloc(&dData.normal, maxpnum * sizeof(float3));

	cudaMalloc(&dData.color, maxpnum * sizeof(cfloat4));

	cudaMalloc(&dData.lambda, maxpnum * sizeof(float));

	cudaMalloc(&dData.type, maxpnum * sizeof(int));
	cudaMalloc(&dData.group, maxpnum * sizeof(int));
	cudaMalloc(&dData.uniqueId, maxpnum * sizeof(int));
	cudaMalloc(&dData.numCons, maxpnum * sizeof(int));
	cudaMalloc(&dData.jetFlag, maxpnum * sizeof(char));

	cudaMalloc(&dData.invMass, maxpnum * sizeof(float));
	cudaMalloc(&dData.mass, maxpnum * sizeof(float));
	cudaMalloc(&dData.absorbBuf, maxpnum * sizeof(float));
	cudaMalloc(&dData.dripBuf, maxpnum * sizeof(float));
	cudaMalloc(&dData.normalizeAbsorb, maxpnum * sizeof(float));
	cudaMalloc(&dData.normalizeDrip, maxpnum * sizeof(float));
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
	cudaMalloc(&dData.triangles, maxtrianglenum * sizeof(objtriangle));
	cudaMalloc(&dData.facetVol, maxtrianglenum * sizeof(float));
	cudaMalloc(&dData.facetArea, maxtrianglenum * sizeof(float));
	cudaMalloc(&dData.baryCenter, maxtrianglenum * sizeof(cfloat3));
	cudaMalloc(&dData.baryCenterTriangleId, maxtrianglenum * sizeof(int));
	cudaMalloc(&dData.sortedBaryCenter, maxtrianglenum * sizeof(cfloat3));
	cudaMalloc(&dData.sortedBaryCenterTriangleId, maxtrianglenum * sizeof(int));

	//object data
	cudaMalloc(&dData.objs, maxobjnum * sizeof(SimulationObject));

	int glen = hParam.gridres.prod();

	cudaMalloc(&dData.particleHash, maxpnum * sizeof(int));
	cudaMalloc(&dData.particleIndex, maxpnum * sizeof(int));
	cudaMalloc(&dData.gridCellStart, glen * sizeof(int));
	cudaMalloc(&dData.gridCellEnd, glen * sizeof(int));
	cudaMalloc(&dData.gridCellCollisionFlag, glen * sizeof(int));

	//cudaMalloc(&dData.baryCenterHash, maxtrianglenum * sizeof(int));
	//cudaMalloc(&dData.baryCenterIndex, maxtrianglenum * sizeof(int));
	//cudaMalloc(&dData.gridCellStartBaryCenter, glen * sizeof(int));
	//cudaMalloc(&dData.gridCellEndBaryCenter, glen * sizeof(int));
}
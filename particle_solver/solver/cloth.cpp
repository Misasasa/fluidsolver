#include "cuda.h"
#include "cuda_runtime.h"
#include "host_defines.h"

#include "pbfsolver.h"

extern SimParam hParam;

inline float rand_in_range(float min, float max) {
	return min + (float)rand()/RAND_MAX * (max-min);
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

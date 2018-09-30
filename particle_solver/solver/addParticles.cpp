
#include "pbfsolver.h"




extern SimParam hParam;

int PBFSolver::addDefaultParticle() {
	if (numParticles ==hParam.maxpnum)
		return -1;
	else {
		hMass[numParticles] = 0;
		hVel[numParticles].Set(0,0,0);
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

void inline read(FILE* fp, cfloat3& dst){
    fscanf(fp, "%f,%f,%f", &dst.x, &dst.y, &dst.z);
}
void inline read(FILE* fp, float& dst) {
	fscanf(fp, "%f", &dst);
}

void PBFSolver::loadboundary(string boundfile){

	printf("Loading boundaries.\n");

    FILE* fp = fopen(boundfile.c_str(), "r");
    if (fp==NULL){
        printf("null boundary file");
		return;
	}

    char buffer[1000];
    cfloat3 min,max;
    while (true){
        if(fscanf(fp, "%s", buffer)==EOF)
            break;
        if (!strcmp(buffer, "WALL")){
            read(fp, min);
            read(fp, max);
            addwall(min,max);
        }
		if (!strcmp(buffer, "OPEN_BOX")) {
			read(fp,min);
			read(fp,max);
			float thickness;
			read(fp, thickness);
			addopenbox(min,max,thickness);
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
		hColor[pid].Set(1,1,1,0.5);
		hMass[pid] = 1;
		hInvMass[pid] = 0;
		hGroup[pid] = -1;
		hType[pid] = TYPE_BOUNDARY;
	}

	return;
}

void PBFSolver::addwall(cfloat3 min, cfloat3 max){
    cfloat3 pos;
	int n = 0, p;
	float dx, dy, dz, x, y, z;
	int cntx, cnty, cntz;

    float spacing = hParam.spacing;
	cntx = ceil( (max.x-min.x) / spacing );
	cntz = ceil( (max.z-min.z) / spacing );
	
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

	for (float y = min.y; y <= max.y; y += spacing ) {	
		for (int xz=0; xz < cnt; xz++ ) {
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
	addwall(cfloat3(min.x,min.y-thickness,min.z), cfloat3(max.x, min.y, max.z));
	//front
	addwall(cfloat3(min.x-thickness,min.y,min.z-thickness), cfloat3(max.x+thickness, max.y, min.z));
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
		float area = cross(a, b).mode()*0.5;
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

		if (p1p2.mode()>maxlen)
			maxlen = p1p2.mode();

		float L0 = p1p2.mode();
		ec.L0 = L0;

		if (ec.t2<0)
			continue;

		p1p3 = hPos[p3] - hPos[p1];
		p1p4 = hPos[p4] - hPos[p1];
		cfloat3 temp = cross(p1p2, p1p3);
		temp = temp / temp.mode();
		cfloat3 temp1 = cross(p1p2, p1p4);
		temp1 = temp1 / temp1.mode();

		float d = dot(temp, temp1);
		d = fmin(1, fmax(d, -1));
		float Phi0 = acos(d);
		ec.Phi0 = Phi0;
	}
	printf("constraint maxlen is %f\n", maxlen);


	return objectvec.size()-1;
}

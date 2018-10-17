#pragma once
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "catpaw/cpToolBox.h"
#include "RenderObject.h"
#include "Solver.h"

using namespace std;

struct fluidInfo {
	int maxpnum;
	int pnum;
	vertex* data;
};





#define PARTICLE_RENDERER 0
#define CUBE_RENDERER 1
#define TRIANGLE_RENDERER 2
#define TRIANGLE_PARTICLE_RENDERER 3

#define SPH 0
#define STOKES 1

class SolverGUI{

private:
	fluidInfo finfo;

	//RenderObject
	ParticleRO* particleRO;
	CubeRO*     cubeRO;
	TriangleRO* triangleRO;
	GeometryRO* geomRO;
	

	//simple geometry
	varr planeGrid;
	GeometryEntity boundingBox;
	vector<GeometryRO> geoROs;

	clock_t LastTime = 0;
	cCamera camera;

	//buffer bindings
	Solver* solver;
	vecf3* hPos;
	vecf4* hColor;
	vertex*  vbuffer;	   //vertex=cfloat3+cfloat4
	cmat4* rotationBuffer; //local rotation of cube
	
	
	bool bDrawGeometry = true;
	bool loadparticleRO = true;
	bool loadcubeRO = false;
	bool loadtriangleRO = true;
	bool bTakeSnapshot = false;
	bool bPause = false;

	int rendermode;
	int frameNo;
	int mode;

public:
	void GetBoundingBox();
	void Run();
	void Exit();

	void setRenderMode(int _mode) {
		rendermode = _mode;
	}

	//--- bind buffer ---
	void bindSolver(Solver* solver_) {
		solver = solver_;
		hPos = & solver->hPos;
		hColor = & solver->hColor;
		GetBoundingBox();
	}

	//######## OPENGL #############

	void Initialize(int argc, char** argv);
	void InitializeGL(int argc, char** argv);
	void LoadIndexedVBO(vector<unsigned int>& indexedvbo);

	~SolverGUI();

	void keyUp(unsigned char key);
	void keyDown(unsigned char key);
	void drag(int dx,int dy);
	void MouseClick(int x,int y,int state, int button);

	void MoveCamera();
	void ReSize(int width,int height);
	void PrintCameraParam();
	
	void render();
	void takeSnapshot();
	void setParticlesz(float sz);
	
};


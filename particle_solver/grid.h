#pragma once

typedef float scalar;

struct Grid
{
	int xlen, ylen, zlen; //cell center dimension
	int ulen, vlen, wlen; //face dimension
	int Size, uSize, vSize, wSize;
	scalar h; //cell size

	void setSize(int x, int y, int z) {
		xlen = x; ylen = y; zlen = z;
		ulen = xlen+1;
		vlen = ylen+1;
		wlen = zlen+1;
		Size = xlen*ylen*zlen;
		uSize = ulen*ylen*zlen;
		vSize = xlen*vlen*zlen;
		wSize = xlen*ylen*wlen;
	}
	int cellId(int x, int y, int z) {
		return (z*ylen+y)*xlen+x;
	}
	int uId(int x, int y, int z) {
		return (z*ylen+y)*ulen+x;
	}
	int vId(int x, int y, int z) {
		return (z*vlen+y)*xlen+x;
	}
	int wId(int x, int y, int z) {
		return (z*ylen+y)*xlen+x;
	}
};

class GridSolver {
public:
	Grid grid;
	scalar* u; //velocity u
	scalar* v; //velocity v
	scalar* w; //velocity w
	scalar* p; //pressure
	scalar* divU;
	
	scalar* b;  // tmp: right hand side
	scalar* Aq; // tmp: A * conjugate basis
	scalar* r;  // tmp: residual
	scalar* q;  // tmp: conjugate basis 0
	
	//param
	scalar dt;
	scalar rho;
	scalar divUsum;
	int frame;

	void loadConfig();
	void allocate();
	void mvproduct(scalar* v, scalar* dst);
	scalar dotproduct(scalar* v1, scalar* v2);
	
	void testcase();
	void divVelocity();
	void makeRHS();
	void solve();
	void updateU();
	void advect();
	void step();
};
#include "grid.h"
#include "catpaw/cpXMLHelper.h"

void GridSolver::loadConfig() {
	tinyxml2::XMLDocument doc;
	int result = doc.LoadFile("config/Eulerian.xml");
	Tinyxml_Reader reader;

	XMLElement* param = doc.FirstChildElement("Param");
	reader.Use(param);
	cint3 dim = reader.GetInt3("dim");
	printf("%d %d %d\n", dim.x, dim.y, dim.z);
	grid.setSize(dim.x, dim.y, dim.z);
}

void GridSolver::allocate() {
	u = (scalar*)malloc(sizeof(scalar)* grid.uSize);
	v = (scalar*)malloc(sizeof(scalar)* grid.vSize);
	w = (scalar*)malloc(sizeof(scalar)* grid.wSize);
	
	p =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	b =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	Aq = (scalar*)malloc(sizeof(scalar)* grid.Size);
	r =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	q =  (scalar*)malloc(sizeof(scalar)* grid.Size);
}

// matrix-free method 
// the matrix A here is the laplacian function, so
void GridSolver::mvproduct(scalar* v, scalar* dst) {
	for (int k=0; k<grid.zlen; k++) {
		for (int j=0; j<grid.ylen; j++) {
			for (int i=0; i<grid.xlen; i++) {
				float res = 0;
				int cellid = grid.cellId(i,j,k);
				if (i==0)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i-1,j,k)];

				if(i==grid.xlen-1)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i+1,j,k)];

				if (j==0)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j-1, k)];

				if (j==grid.ylen-1)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j+1, k)];

				if (k==0)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j, k-1)];

				if (k==grid.zlen-1)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j, k+1)];
				
				dst[cellid] = res;
			}
		}
	}
}

scalar GridSolver::dotproduct(scalar* v1, scalar* v2) {
	scalar res = 0;
	for(int i=0; i<grid.Size; i++)
		res += v1[i]*v2[i];
	return res;
}
#include <time.h>
void GridSolver::makeRHS() {
	//srand(time(NULL));
	//for (int i=0; i<grid.Size; i++) {
	//	b[i] = (scalar)rand()/RAND_MAX;
	//}


}

void GridSolver::solve() {
	//conjugate gradient
	
	int iter = 0;
	scalar alpha, beta;
	scalar rr, qAq;
	scalar rabs;

	//initialize x_0 = 0
	for (int i=0; i<grid.Size; i++) {
		r[i] = b[i];
		q[i] = r[i];
		p[i] = 0;
	}

	while(true) {
		mvproduct(q, Aq);
		rr = dotproduct(r,r);
		qAq = dotproduct(q, Aq);
		alpha = rr / qAq;

		for (int i=0; i<grid.Size; i++) {
			p[i] += alpha * q[i];
			r[i] -= alpha * Aq[i];
		}
		
		beta = dotproduct(r,r) / rr;
		rabs = 0;

		for (int i=0; i<grid.Size; i++) {
			q[i] = r[i] + beta * q[i];
			rabs += abs(r[i]);
		}
		printf("turn %d residual: %f\n", iter, rabs);
		if(rabs < 0.0000001)
			break;
		iter ++;
	};

}
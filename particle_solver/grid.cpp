#include "grid.h"
#include "catpaw/cpXMLHelper.h"
#include <time.h>

void GridSolver::loadConfig() {
	tinyxml2::XMLDocument doc;
	int result = doc.LoadFile("config/Eulerian.xml");
	Tinyxml_Reader reader;

	XMLElement* param = doc.FirstChildElement("Param");
	reader.Use(param);
	cint3 dim = reader.GetInt3("dim");
	grid.h = reader.GetFloat("h");
	grid.setSize(dim.x, dim.y, dim.z);
	printf("grid dimension: %d %d %d %f \n", dim.x, dim.y, dim.z, grid.h);

	dt = reader.GetFloat("dt");
	rho = reader.GetFloat("rho");
	frame = 0;
}

void GridSolver::allocate() {
	u = (scalar*)malloc(sizeof(scalar)* grid.uSize);
	v = (scalar*)malloc(sizeof(scalar)* grid.vSize);
	w = (scalar*)malloc(sizeof(scalar)* grid.wSize);
	memset(u, 0, sizeof(scalar)*grid.uSize);
	memset(v, 0, sizeof(scalar)*grid.vSize);
	memset(w, 0, sizeof(scalar)*grid.wSize);

	p =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	b =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	Aq = (scalar*)malloc(sizeof(scalar)* grid.Size);
	r =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	q =  (scalar*)malloc(sizeof(scalar)* grid.Size);
	divU = (scalar*)malloc(sizeof(scalar)*grid.Size);

	memset(p, 0, sizeof(scalar)*grid.Size);
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

				if (j==0) // near solid!
					//res += v[cellid];
					res += 0;
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

void GridSolver::testcase() {
	for(int i=0; i<grid.xlen; i++)
		for(int j=0; j<grid.vlen; j++)
			for (int k=0; k<grid.zlen; k++) {
				v[grid.vId(i,j,k)] = (scalar)rand()/RAND_MAX;
			}
}

void GridSolver::divVelocity() {
	divUsum = 0;
	for (int k=0; k<grid.zlen; k++) {
		for (int j=0; j<grid.ylen; j++) {
			for (int i=0; i<grid.xlen; i++) {
				scalar div = 0;
				div += u[grid.uId(i+1,j,k)]-u[grid.uId(i,j,k)];
				div += v[grid.vId(i,j+1,k)]-v[grid.vId(i,j,k)];
				div += w[grid.wId(i,j,k+1)]-w[grid.wId(i,j,k)];
				div /= grid.h;
				divU[grid.cellId(i,j,k)] = -div;
				divUsum += div*div;
			}
		}
	}
	printf("div sum %f\n", divUsum);
}

void GridSolver::makeRHS() {
	//srand(time(NULL));
	//for (int i=0; i<grid.Size; i++) {
	//	b[i] = (scalar)rand()/RAND_MAX;
	//}
	divVelocity();
	for (int k=0; k<grid.zlen; k++) {
		for (int j=0; j<grid.ylen; j++) {
			for (int i=0; i<grid.xlen; i++) {
				
				int cellid = grid.cellId(i, j, k);
				float rhs = divU[cellid];
				if (j==0) //near solid!
					rhs += (0-v[grid.vId(i,j,k)])/grid.h;
				b[cellid] = rhs * rho / dt;
				//printf("%f %f\n",b[cellid], divU[cellid]);
			}
		}
	}

}

void GridSolver::solve() {
	//conjugate gradient
	
	int iter = 0;
	scalar alpha, beta;
	scalar rr, qAq;
	scalar rabs;

	mvproduct(p, Aq);

	//initialize x_0 = 0
	for (int i=0; i<grid.Size; i++) {
		r[i] = b[i] - Aq[i];
		q[i] = r[i];
	}

	while(true) {
		mvproduct(q, Aq);
		rr = dotproduct(r,r);
		qAq = dotproduct(q, Aq);
		if (rr==0 || qAq==0) {
			printf("trivial all-zero condition. quit.\n");
			break;
		}
			
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

	//mvproduct(p, Aq);
	//for(int i=0; i<grid.Size; i++)
	//	printf("%f %f\n", Aq[i], b[i]);
	
}

void GridSolver::updateU() {
	float c = dt/rho/grid.h;
	for (int k=0; k<grid.zlen; k++) {
		for (int j=0; j<grid.ylen; j++) {
			for (int i=0; i<grid.xlen; i++) {	
				int cellid = grid.cellId(i, j, k);
				
				if (i==0)
					u[grid.uId(i,j,k)] -= (p[cellid]-0)*c;
				else
					u[grid.uId(i,j,k)] -= (p[cellid] - p[grid.cellId(i-1,j,k)])*c;

				if (i==grid.xlen-1)
					u[grid.uId(i+1, j, k)] -= (0 - p[cellid])*c;
				

				if (j==0) // near solid!
					v[grid.vId(i,j,k)] = 0;
				else
					v[grid.vId(i,j,k)] -= (p[cellid] - p[grid.cellId(i,j-1,k)])*c;

				if (j==grid.ylen-1)
					v[grid.vId(i,j+1,k)] -= (0-p[cellid])*c;
				

				if (k==0)
					w[grid.wId(i,j,k)] -= (p[cellid]-0)*c;
				else
					w[grid.wId(i,j,k)] -= (p[cellid] - p[grid.cellId(i,j,k-1)])*c;

				if (k==grid.zlen-1)
					w[grid.wId(i,j,k+1)] -= (0-p[cellid])*c;
				

			}
		}
	}
}

void GridSolver::advect() {
	//do nothing

	//gravity
	for (int i=0; i<grid.xlen; i++)
		for (int j=0; j<grid.vlen; j++)
			for (int k=0; k<grid.zlen; k++)
				v[grid.vId(i, j, k)] += -10 * dt;
			
}

void GridSolver::step() {
	advect();
	//gs.testcase();
	makeRHS();
	solve();
	updateU();
	divVelocity();
	frame ++;
}
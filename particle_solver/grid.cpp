#include "grid.h"
#include "catpaw/cpXMLHelper.h"
#include <time.h>

void GridSolver::loadConfig() {
	tinyxml2::XMLDocument doc;
	int result = doc.LoadFile("config/Eulerian.xml");
	Tinyxml_Reader reader;

	XMLElement* param = doc.FirstChildElement("Param");
	reader.Use(param);
	grid.dim = reader.GetInt3("dim");
	grid.padding = reader.GetInt("padding");
	grid.h = reader.GetFloat("h");
	grid.xmin.x = -grid.dim.x/2*grid.h;
	grid.xmin.y = -grid.dim.y/2*grid.h;
	grid.xmin.z = -grid.dim.z/2*grid.h;

	grid.setSize();
	printf("grid dimension: %d %d %d %f \n", grid.dim.x, grid.dim.y, grid.dim.z, grid.h);

	dt = reader.GetFloat("dt");
	rho = reader.GetFloat("rho");
	frame = 0;
	pad = grid.padding;
}

void GridSolver::allocate() {
	u = (scalar*)malloc(sizeof(scalar)* grid.uSize);
	v = (scalar*)malloc(sizeof(scalar)* grid.vSize);
	w = (scalar*)malloc(sizeof(scalar)* grid.wSize);
	uadv = (scalar*)malloc(sizeof(scalar)* grid.uSize);
	vadv = (scalar*)malloc(sizeof(scalar)* grid.vSize);
	wadv = (scalar*)malloc(sizeof(scalar)* grid.wSize);

	memset(u, 0, sizeof(scalar)*grid.uSize);
	memset(v, 0, sizeof(scalar)*grid.vSize);
	memset(w, 0, sizeof(scalar)*grid.wSize);
	memset(uadv, 0, sizeof(scalar)*grid.uSize);
	memset(vadv, 0, sizeof(scalar)*grid.vSize);
	memset(wadv, 0, sizeof(scalar)*grid.wSize);

	p =  (scalar*)malloc(sizeof(scalar)* grid.dimSize);
	divU = (scalar*)malloc(sizeof(scalar)*grid.dimSize);
	b =  (scalar*)malloc(sizeof(scalar)* grid.dimSize);
	Aq = (scalar*)malloc(sizeof(scalar)* grid.dimSize);
	r =  (scalar*)malloc(sizeof(scalar)* grid.dimSize);
	q =  (scalar*)malloc(sizeof(scalar)* grid.dimSize);
	

	memset(p, 0, sizeof(scalar)*grid.dimSize);
}



// matrix-free method 
// the matrix A here is the laplacian function, so
void GridSolver::mvproduct(scalar* v, scalar* dst) {

	for (int k=0; k<grid.dim.z; k++) {
		for (int j=0; j<grid.dim.y; j++) {
			for (int i=0; i<grid.dim.x; i++) {
				float res = 0;
				int cellid = grid.cellId(i,j,k);
				if (i==0)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i-1,j,k)];

				if(i==grid.dim.x-1)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i+1,j,k)];

				if (j==0) // near solid!
					//res += v[cellid];
					res += 0;
				else
					res += v[cellid] - v[grid.cellId(i, j-1, k)];

				if (j==grid.dim.y-1)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j+1, k)];

				if (k==0)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j, k-1)];

				if (k==grid.dim.z-1)
					res += v[cellid];
				else
					res += v[cellid] - v[grid.cellId(i, j, k+1)];
				
				dst[cellid] = res;
				//printf("%f\n",res);
			}
		}
	}
}

scalar GridSolver::dotproduct(scalar* v1, scalar* v2) {
	scalar res = 0;
	for(int i=0; i<grid.dimSize; i++)
		res += v1[i]*v2[i];
	return res;
}

void GridSolver::divVelocity() {
	divUsum = 0;
	for (int k=0; k<grid.dim.z; k++) {
		for (int j=0; j<grid.dim.y; j++) {
			for (int i=0; i<grid.dim.x; i++) {
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
	
	divVelocity();
	for (int k=0; k<grid.dim.z; k++) {
		for (int j=0; j<grid.dim.y; j++) {
			for (int i=0; i<grid.dim.x; i++) {
				
				int cellid = grid.cellId(i, j, k);
				float rhs = divU[cellid];
				if (j==0) //near solid!
					rhs += (0-v[grid.vId(i,j,k)])/grid.h;
				b[cellid] = rhs * rho * grid.h*grid.h / dt;
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
	for (int i=0; i<grid.dimSize; i++) {
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

		for (int i=0; i<grid.dimSize; i++) {
			p[i] += alpha * q[i];
			r[i] -= alpha * Aq[i];
		}
		
		beta = dotproduct(r,r) / rr;
		rabs = 0;
		

		for (int i=0; i<grid.dimSize; i++) {
			q[i] = r[i] + beta * q[i];
			rabs += abs(r[i]);
		}
		rabs/=grid.dimSize;
		printf("turn %d residual: %f\n", iter, rabs);
		if(rabs < 0.0000001)
			break;
		iter ++;
	};

	mvproduct(p, Aq);
	//for(int i=0; i<grid.dimSize; i++)
	//	printf("%f %f\n", Aq[i], b[i]);
	
}

void GridSolver::updateU() {
	float c = dt/rho/grid.h;
	cfloat3 ucell;
	float ulenth, maxu = 0;

	for (int k=0; k<grid.dim.z; k++) {
		for (int j=0; j<grid.dim.y; j++) {
			for (int i=0; i<grid.dim.x; i++) {	
				int cellid = grid.cellId(i, j, k);
				
				if (i==0)
					u[grid.uId(i,j,k)] -= (p[cellid]-0)*c;
				else
					u[grid.uId(i,j,k)] -= (p[cellid] - p[grid.cellId(i-1,j,k)])*c;

				if (i==grid.dim.x-1)
					u[grid.uId(i+1, j, k)] -= (0 - p[cellid])*c;	

				if (j==0) // near solid!
					v[grid.vId(i,j,k)] = 0;
				else
					v[grid.vId(i,j,k)] -= (p[cellid] - p[grid.cellId(i,j-1,k)])*c;

				if (j==grid.dim.y-1)
					v[grid.vId(i,j+1,k)] -= (0-p[cellid])*c;
				

				if (k==0)
					w[grid.wId(i,j,k)] -= (p[cellid]-0)*c;
				else
					w[grid.wId(i,j,k)] -= (p[cellid] - p[grid.cellId(i,j,k-1)])*c;

				if (k==grid.dim.z-1)
					w[grid.wId(i,j,k+1)] -= (0-p[cellid])*c;
				
				//get max velocity
				ucell.x = u[grid.uId(i, j, k)]+u[grid.uId(i+1, j, k)]*0.5;
				ucell.y = v[grid.vId(i,j,k)]+v[grid.vId(i,j+1,k)]*0.5;
				ucell.z = w[grid.wId(i,j,k)]+w[grid.wId(i,j,k+1)]*0.5;
				ulenth = sqrt(dot(ucell,ucell));
				maxu = ulenth>maxu? ulenth : maxu;
			}
		}
	}
	
	//update time step size
	maxu += sqrt(5*grid.h*9.8);
	//dt = 5*grid.h / maxu;
	//printf("time step size updated as %f.\n", dt);
}

cint3 GridSolver::locateCell(cfloat3 p) {
	float x, y, z;
	x = (p.x - grid.xmin.x)/grid.h;
	y = (p.y - grid.xmin.y)/grid.h;
	z = (p.z - grid.xmin.z)/grid.h;
	int xi, yi, zi;
	xi = (int)x;
	yi = (int)y;
	zi = (int)z;

	//clamp
	if (xi==-1 && x>-EPSILON) {
		xi = 0;
	}
	if (yi==-1 && y>-EPSILON) {
		yi = 0;
	}
	if (zi==-1 && z>-EPSILON) {
		zi = 0;
	}
	if (xi==grid.dim.x && x<xi+EPSILON) {
		xi = grid.dim.x-1;
	}
	if (yi==grid.dim.y && y<yi+EPSILON) {
		yi = grid.dim.y-1;
	}
	if (zi==grid.dim.z && z<zi+EPSILON) {
		zi = grid.dim.z-1;
	}
	return cint3(xi,yi,zi);
}

scalar GridSolver::sampleU(cfloat3 p) {
	scalar ures;
	cint3 c;
	float x,y,z;
	x = (p.x - grid.xmin.x)/grid.h;
	y = (p.y - grid.xmin.y)/grid.h;
	z = (p.z - grid.xmin.z)/grid.h;
	//floor
	c.x = x;
	c.y = y;
	c.z = z;
	//outside of boundary
	if(c.x<0 || c.x>grid.dim.x || c.y<0 || c.y>grid.dim.y
		||c.z<0 || c.z>grid.dim.z)
		return 0;
	//dual grid of u
	if(y-c.y<0.5)
		c.y--;
	if(z-c.z<0.5)
		c.z--;
	
	float w1,w2,w3;
	w1 = (p.x - (grid.xmin.x+ c.x*grid.h)      )/grid.h;
	w2 = (p.y - (grid.xmin.y+(c.y+0.5)*grid.h) )/grid.h;
	w3 = (p.z - (grid.xmin.z+(c.z+0.5)*grid.h) )/grid.h;
	
	ures = 
		(1-w1)*((1-w2)*((1-w3)*u[grid.uId(c.x,c.y,c.z)]    +w3*u[grid.uId(c.x,c.y,c.z+1)]) +
			       w2 *((1-w3)*u[grid.uId(c.x, c.y+1, c.z)]+w3*u[grid.uId(c.x, c.y+1, c.z+1)]))
		+ 
		   w1* ((1-w2)*((1-w3)*u[grid.uId(c.x+1, c.y, c.z)]  +w3*u[grid.uId(c.x+1, c.y, c.z+1)])+
		           w2 *((1-w3)*u[grid.uId(c.x+1, c.y+1, c.z)]+w3*u[grid.uId(c.x+1, c.y+1, c.z+1)]));
	
	return ures;
}

scalar GridSolver::sampleV(cfloat3 p) {
	scalar vres;
	cint3 c;
	float x, y, z;
	x = (p.x - grid.xmin.x)/grid.h;
	y = (p.y - grid.xmin.y)/grid.h;
	z = (p.z - grid.xmin.z)/grid.h;
	//floor
	c.x = x;
	c.y = y;
	c.z = z;
	//outside of boundary
	if (c.x<0 || c.x>grid.dim.x || c.y<0 || c.y>grid.dim.y
		||c.z<0 || c.z>grid.dim.z)
		return 0;
	//dual grid of u
	if (x-c.x<0.5)
		c.x--;
	if (z-c.z<0.5)
		c.z--;

	float w1, w2, w3;
	w1 = (p.x - (grid.xmin.x+(c.x+0.5)*grid.h))/grid.h;
	w2 = (p.y - (grid.xmin.y+ c.y*grid.h))/grid.h;
	w3 = (p.z - (grid.xmin.z+(c.z+0.5)*grid.h))/grid.h;

	vres =
		(1-w1)*((1-w2)*((1-w3)*v[grid.vId(c.x, c.y, c.z)]    +w3*v[grid.vId(c.x, c.y, c.z+1)]) +
			w2 *((1-w3)*v[grid.vId(c.x, c.y+1, c.z)]+w3*v[grid.vId(c.x, c.y+1, c.z+1)]))
		+
		w1* ((1-w2)*((1-w3)*v[grid.vId(c.x+1, c.y, c.z)]  +w3*v[grid.vId(c.x+1, c.y, c.z+1)])+
			w2 *((1-w3)*v[grid.vId(c.x+1, c.y+1, c.z)]+w3*v[grid.vId(c.x+1, c.y+1, c.z+1)]));

	return vres;
}

scalar GridSolver::sampleW(cfloat3 p) {
	scalar wres;
	cint3 c;
	float x, y, z;
	x = (p.x - grid.xmin.x)/grid.h;
	y = (p.y - grid.xmin.y)/grid.h;
	z = (p.z - grid.xmin.z)/grid.h;
	//floor
	c.x = x;
	c.y = y;
	c.z = z;
	//outside of boundary
	if (c.x<0 || c.x>grid.dim.x || c.y<0 || c.y>grid.dim.y
		||c.z<0 || c.z>grid.dim.z)
		return 0;
	//dual grid of u
	if (x-c.x<0.5)
		c.x--;
	if (y-c.y<0.5)
		c.y--;

	float w1, w2, w3;
	w1 = (p.x - (grid.xmin.x+ (c.x+0.5)*grid.h))/grid.h;
	w2 = (p.y - (grid.xmin.y+(c.y+0.5)*grid.h))/grid.h;
	w3 = (p.z - (grid.xmin.z+c.z*grid.h))/grid.h;

	wres =
		(1-w1)*((1-w2)*((1-w3)*w[grid.wId(c.x, c.y, c.z)]    +w3*w[grid.wId(c.x, c.y, c.z+1)]) +
			w2 *((1-w3)*w[grid.wId(c.x, c.y+1, c.z)]+w3*w[grid.wId(c.x, c.y+1, c.z+1)]))
		+
		w1* ((1-w2)*((1-w3)*w[grid.wId(c.x+1, c.y, c.z)]  +w3*w[grid.wId(c.x+1, c.y, c.z+1)])+
			w2 *((1-w3)*w[grid.wId(c.x+1, c.y+1, c.z)]+w3*w[grid.wId(c.x+1, c.y+1, c.z+1)]));

	return wres;
}



void GridSolver::advect() {
	//semi-lagrangian advection
	cfloat3 u_interp;
	cfloat3 utmp, utmp1;
	cfloat3 xtmp;
	cfloat3 xmid;
	//x-component
	for(int k=0; k<grid.dim.z; k++)
		for(int j=0;j<grid.dim.y; j++)
			for (int i=0; i<grid.dim.x+1; i++) {
				utmp.x = u[grid.uId(i,j,k)];
				utmp.y = (v[grid.vId(i,j,k)]+v[grid.vId(i-1,j,k)]
					+v[grid.vId(i,j+1,k)]+v[grid.vId(i-1,j+1,k)])*0.25;
				utmp.z = (w[grid.wId(i,j,k)]+w[grid.wId(i-1,j,k)]
					+w[grid.wId(i,j,k+1)]+w[grid.wId(i-1,j,k+1)])*0.25;
				
				//back 1/2 time step
				xtmp.x = grid.xmin.x + grid.h*i;
				xtmp.y = grid.xmin.y + grid.h*(j+0.5);
				xtmp.z = grid.xmin.z + grid.h*(k+0.5);
				xmid = xtmp + utmp * dt * (-0.5);

				//get u
				utmp1.x = sampleU(xmid);
				utmp1.y = sampleV(xmid);
				utmp1.z = sampleW(xmid);
				xmid = xtmp + utmp1 * dt * (-1);

				uadv[grid.uId(i,j,k)] = sampleU(xmid);

				//testing! passed
				/*if (dot(utmp, utmp)!=0) {
					printf("1 %f %f %f\n", utmp.x, utmp.y, utmp.z);
					printf("2 %f %f %f\n", sampleU(xtmp), sampleV(xtmp), sampleW(xtmp));

				}*/
			}

	//y-component
	for (int k=0; k<grid.dim.z; k++)
		for (int j=0; j<grid.dim.y+1; j++)
			for (int i=0; i<grid.dim.x; i++) {
				utmp.x = (u[grid.uId(i, j, k)]+u[grid.uId(i+1,j,k)]
					+u[grid.uId(i,j-1,k)]+u[grid.uId(i+1,j-1,k)])*0.25;
				utmp.y = v[grid.vId(i, j, k)];
				utmp.z = (w[grid.wId(i, j, k)]+w[grid.wId(i, j, k+1)]
					+w[grid.wId(i, j-1, k)]+w[grid.wId(i, j-1, k+1)])*0.25;

				//back 1/2 time step
				xtmp.x = grid.xmin.x + grid.h*(i+0.5);
				xtmp.y = grid.xmin.y + grid.h*j;
				xtmp.z = grid.xmin.z + grid.h*(k+0.5);
				xmid = xtmp + utmp * dt * (-0.5);

				//get u
				utmp1.x = sampleU(xmid);
				utmp1.y = sampleV(xmid);
				utmp1.z = sampleW(xmid);
				xmid = xtmp + utmp1 * dt * (-1);

				vadv[grid.vId(i, j, k)] = sampleV(xmid);
			}

	//z-component
	for (int k=0; k<grid.dim.z+1; k++)
		for (int j=0; j<grid.dim.y; j++)
			for (int i=0; i<grid.dim.x; i++) {
				utmp.x = (u[grid.uId(i, j, k)]+u[grid.uId(i,j,k-1)]
					+ u[grid.uId(i+1,j,k)]+u[grid.uId(i+1,j,k-1)] )*0.25;
				utmp.y = (v[grid.vId(i, j, k)]+v[grid.vId(i, j, k-1)]
					+v[grid.vId(i, j+1, k)]+v[grid.vId(i, j+1, k-1)])*0.25;
				utmp.z = w[grid.wId(i, j, k)];

				//back 1/2 time step
				xtmp.x = grid.xmin.x + grid.h*(i+0.5);
				xtmp.y = grid.xmin.y + grid.h*(j+0.5);
				xtmp.z = grid.xmin.z + grid.h* k;
				xmid = xtmp + utmp * dt * (-0.5);

				//get u
				utmp1.x = sampleU(xmid);
				utmp1.y = sampleV(xmid);
				utmp1.z = sampleW(xmid);
				xmid = xtmp + utmp1 * dt * (-1);

				wadv[grid.wId(i, j, k)] = sampleW(xmid);
			}
}

void GridSolver::bodyForce() {
	

	//gravity
	for (int i=0; i<grid.dim.x; i++)
		for (int j=0; j<grid.dim.y+1; j++)
			for (int k=0; k<grid.dim.z; k++)
				v[grid.vId(i, j, k)] += -9.8 * dt;
			
}

void GridSolver::testcase() {
	for (int i=0; i<grid.dim.x; i++)
		for (int j=0; j<grid.dim.y; j++)
			for (int k=0; k<grid.dim.z; k++) {
				u[grid.uId(i,j,k)] = i*2;
				v[grid.vId(i,j,k)] = j*3;
				w[grid.wId(i,j,k)] = k*4;
				if(i==grid.dim.x-1)
					u[grid.uId(i+1,j,k)] = (i+1)*2;
				if (j==grid.dim.y-1)
					v[grid.vId(i, j+1, k)] = (j+1)*3;
				if (k==grid.dim.z-1)
					w[grid.wId(i, j, k+1)] = (k+1)*4;

			}
				
}

void GridSolver::step() {
	//advect();
	bodyForce();
	makeRHS();
	solve();
	updateU();
	divVelocity();
	frame ++;
}
#pragma once

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

#ifdef BUILD_CUDA
#include <host_defines.h>
#define HDFUNC __host__ __device__

#else
#define HDFUNC
#define __inline__ inline
#endif

#define ZERO(a) (fabsf(a) < 1e-10)

HDFUNC inline float cmin(float a, float b) {
	return a < b ? a : b;
}


HDFUNC inline float cmax(float a, float b) {
	return a > b ? a : b;
}



struct cfloat2 {
	float x,y;
	HDFUNC cfloat2(){}
	HDFUNC cfloat2(float _x,float _y):x(_x),y(_y){}
	HDFUNC void Set(float _x,float _y){ x=_x; y=_y; }
	HDFUNC cfloat2 operator*(float s){
		return cfloat2(x*s, y*s);
	}
	HDFUNC cfloat2 operator/(float s){
		return cfloat2(x/s,y/s);
	}
	HDFUNC cfloat2 operator-(cfloat2 b){
		return cfloat2(x-b.x,y-b.y);
	}
	HDFUNC cfloat2 operator+(cfloat2 b){
		return cfloat2(x+b.x, y+b.y);
	}
};


inline float dot(cfloat2& a, cfloat2& b) {
	return a.x*b.x + a.y*b.y;
}





struct cfloat3 {
	float x, y, z;
	HDFUNC cfloat3() {}
	HDFUNC cfloat3(float _x, float _y, float _z) :x(_x), y(_y), z(_z) {}
	HDFUNC void Set(float _x, float _y, float _z) {
		x = _x;
		y = _y;
		z = _z;
	}
	HDFUNC cfloat3 operator- (cfloat3 b) {
		return cfloat3(x - b.x, y - b.y, z - b.z);
	}
	HDFUNC cfloat3 operator+(cfloat3 b) {
		return cfloat3(x + b.x, y + b.y, z + b.z);
	}
	HDFUNC cfloat3 operator*(cfloat3 b) {
		return cfloat3(x * b.x, y * b.y, z * b.z);
	}
	HDFUNC cfloat3 operator/(cfloat3 b) {
		return cfloat3(x / b.x, y / b.y, z / b.z);
	}
	HDFUNC cfloat3 operator/(float b){
		return cfloat3(x/b,y/b,z/b);
	}

	HDFUNC cfloat3 operator*(float s) {
		return cfloat3(x*s, y*s, z*s);
	}
	HDFUNC cfloat3 operator+(float a) {
		return cfloat3(x + a, y + a, z + a);
	}
	HDFUNC cfloat3 operator-(float a) {
		return cfloat3(x - a, y - a, z - a);
	}
	HDFUNC void operator+= (cfloat3 b) {
		x+=b.x; y+=b.y; z+=b.z;
	}
	HDFUNC void operator-= (cfloat3 b){
		x-=b.x; y-=b.y; z-=b.z;
	}
	HDFUNC void operator*= (float b) {
		x*=b; y*=b; z*=b;
	}
	HDFUNC void operator/= (float b) {
		x/=b; y/=b; z/=b;
	}
	HDFUNC void operator /= (cfloat3 b){
		x/=b.x; y/=b.y; z/=b.z;
	}

	HDFUNC inline float minx() {
		return cmin(x, cmin(y, z));
	}
	HDFUNC inline float Norm() {
		return sqrt(x*x+y*y+z*z);
	}
	HDFUNC inline float square() {
		return x*x + y*y + z*z;
	}
	HDFUNC inline float dot(cfloat3 right) {
		return x*right.x + y*right.y + z*right.z;
	}
	HDFUNC inline float& operator[](int index) {
		switch (index)
		{
		case 0:
			return x;
		case 1:
			return y;
		default:
			return z;
		}
	}
};

HDFUNC inline cfloat3 minfilter(cfloat3 a, cfloat3 b) {
	return cfloat3(cmin(a.x, b.x), cmin(a.y, b.y), cmin(a.z, b.z));
}

inline cfloat3 maxfilter(cfloat3 a, cfloat3 b) {
	return cfloat3(cmax(a.x, b.x), cmax(a.y, b.y), cmax(a.z, b.z));
}

HDFUNC inline cfloat3 cross(cfloat3 a, cfloat3 b) {
	cfloat3 tmp;
	tmp.x = a.y*b.z - a.z*b.y;
	tmp.y = a.z*b.x - a.x*b.z;
	tmp.z = a.x*b.y - a.y*b.x;
	return tmp;
}

HDFUNC inline float dot(cfloat3& a,cfloat3& b){
	return a.x*b.x+a.y*b.y+a.z*b.z;
}


float angle(cfloat3& a,cfloat3& b);





struct cint3 {
	int x, y, z;
	HDFUNC cint3() {}
	HDFUNC cint3(int _x, int _y, int _z) :x(_x), y(_y), z(_z) {}
	HDFUNC void Set(int _x, int _y, int _z) {
		x = _x;
		y = _y;
		z = _z;
	}
	HDFUNC operator cfloat3(){
		return cfloat3(x,y,z);
	}

	HDFUNC int prod() {
		return x*y*z;
	}
	
	HDFUNC cint3 operator + (cint3& b) {
		return cint3(x+b.x, y+b.y, z+b.z);
	}
	HDFUNC cfloat3 operator * (float b) {
		return cfloat3(x*b, y*b, z*b);
	}
};

HDFUNC inline cint3 CINT3(cfloat3 a) {
	return cint3(a.x,a.y,a.z);
}






struct cmat3 
{
	float data[9];
	HDFUNC cmat3() {
		for(int i=0;i<9;i++) data[i]=0;
	}
	HDFUNC cmat3(float mat[]) { //make sure mat longer than 9
		for(int i=0;i<9;i++) data[i]=mat[i];
	}

	HDFUNC void Identity() {
		data[0] = data[4] = data[8] = 1.0f;
		data[1] = data[2] = data[3] = data[5] = data[6] = data[7] = 0.0f;
	}
	HDFUNC void Set(float mat[]) {
		for (int i=0; i<9; i++) data[i] = mat[i];
	}
	HDFUNC void Set(float c) {
		for(int i=0;i<9;i++) data[i]=c;
	}
	HDFUNC void Print() {
		for (int i=0; i<3; i++) {
			printf("%f %f %f\n", data[i*3], data[i*3+1], data[i*3+2]);
		}
	}
	HDFUNC float* operator[](int i){
		return &data[i*3];
	}
	HDFUNC cmat3 operator+(cmat3& b) {
		cmat3 res;
		for (int k=0; k<9; k++) {
			res.data[k]=data[k]+b.data[k];
		}
		return res;
	}
	HDFUNC cmat3 operator*(float b) {
		cmat3 res;
		for (int k=0; k<9; k++) {
			res.data[k] = data[k] * b;
		}
		return res;
	}

	HDFUNC float Det() {
		return data[0]*(data[4]*data[8]-data[5]*data[7]) + data[1]*(data[5]*data[6]-data[3]*data[8]) + data[2]*(data[3]*data[7]-data[4]*data[6]);
	}
	HDFUNC cmat3 Inv() {
		cmat3 inv;
		float det = Det();
		if (fabs(det)<1e-10) {
			return inv;
		}
		inv[0][0] = (data[4]*data[8]-data[5]*data[7])/det;
		inv[0][1] = (data[2]*data[7]-data[1]*data[8])/det;
		inv[0][2] = (data[1]*data[5]-data[2]*data[4])/det;
		inv[1][0] = (data[5]*data[6]-data[3]*data[8])/det;
		inv[1][1] = (data[0]*data[8]-data[2]*data[6])/det;
		inv[1][2] = (data[2]*data[3]-data[0]*data[5])/det;
		inv[2][0] = (data[3]*data[7]-data[4]*data[6])/det;
		inv[2][1] = (data[1]*data[6]-data[0]*data[7])/det;
		inv[2][2] = (data[0]*data[4]-data[1]*data[3])/det;
		return inv;
	}

	HDFUNC cmat3 Reci()
	{
		cmat3 ret;
		for (int i = 0; i < 9; i++)
		{
			if (ZERO(data[i]))
				ret.data[i] = 0.0f;
			else
				ret.data[i] = 1 / data[i];
		}
		return ret;
	}

	HDFUNC float Norm() {
		float norm=0;
		for(int k=0;k<9;k++)
			norm += data[k]*data[k];
		return sqrt(norm);
	}

	HDFUNC void Add(cmat3& b) {
		for(int k=0; k<9; k++)
			data[k] = data[k] + b.data[k];
	}
	HDFUNC void Multiply(float b) {
		for(int k=0;k<9;k++)
			data[k] = data[k]*b;
	}

	HDFUNC cfloat3 Col(int n) {
		if (n<0 || n>2) {
			printf("Illegal Matrix Column Visit.\n");
			return cfloat3(0,0,0);
		}
		return cfloat3(data[n], data[3+n], data[6+n]);
	}

};

struct cmat2
{
	float data[4];
	HDFUNC cmat2() {
		for (int i = 0; i<4; i++) data[i] = 0;
	}
	HDFUNC cmat2(float a00, float a01, float a10, float a11) {
		data[0] = a00; data[1] = a01; data[2] = a10; data[3] = a11;
	}
	HDFUNC cmat2(float mat[]) { //make sure mat longer than 4
		for (int i = 0; i<4; i++) data[i] = mat[i];
	}

	HDFUNC void Set(float mat[]) {
		for (int i = 0; i<4; i++) data[i] = mat[i];
	}
	HDFUNC void Set(float c) {
		for (int i = 0; i<4; i++) data[i] = c;
	}
	HDFUNC void Print() {
		for (int i = 0; i<2; i++) {
			printf("%f %f\n", data[i * 2], data[i * 2 + 1]);
		}
	}
	HDFUNC float* operator[](int i) {
		return &data[i * 2];
	}
	HDFUNC cmat3 operator+(cmat3& b) {
		cmat3 res;
		for (int k = 0; k<4; k++) {
			res.data[k] = data[k] + b.data[k];
		}
		return res;
	}
	HDFUNC cmat3 operator*(float b) {
		cmat3 res;
		for (int k = 0; k<4; k++) {
			res.data[k] = data[k] * b;
		}
		return res;
	}

	HDFUNC float Det() {
		return data[0] * data[3] - data[1] * data[2];
	}
	HDFUNC cmat2 Inv() {
		cmat2 inv;
		float det = Det();
		if (fabs(det)<1e-10) {
			return inv;
		}
		inv[0][0] = (*this)[1][1] / det;
		inv[0][1] = -(*this)[0][1] / det;
		inv[1][0] = -(*this)[1][0] / det;
		inv[1][1] = (*this)[0][0] / det;
		return inv;
	}

	HDFUNC float Norm() {
		float norm = 0;
		for (int k = 0; k<4; k++)
			norm += data[k] * data[k];
		return sqrt(norm);
	}

	HDFUNC void Add(cmat3& b) {
		for (int k = 0; k<4; k++)
			data[k] = data[k] + b.data[k];
	}
	HDFUNC void Multiply(float b) {
		for (int k = 0; k<4; k++)
			data[k] = data[k] * b;
	}
	HDFUNC cmat2 transpose()
	{
		cmat2 ret;
		ret[0][0] = (*this)[0][0];
		ret[0][1] = (*this)[1][0];
		ret[1][0] = (*this)[0][1];
		ret[1][1] = (*this)[1][1];
		return ret;
	}
	
};

HDFUNC cmat3 directG3(int a, int b, float c, float s);
HDFUNC cmat3 G3(int a, int b, float x, float y);
HDFUNC cmat3 uncG3(int a, int b, float x, float y);

HDFUNC void Zerochasing(cmat3& A, cmat3& U, cmat3& V);
HDFUNC void Bidiagonalize(cmat3& A, cmat3& U, cmat3& V);
HDFUNC void Updateabc(float* a, float* b, float* c, cmat3& B);
HDFUNC cmat2 TopLeft(cmat3& A);
HDFUNC cmat2 BotRight(cmat3& A);
HDFUNC cmat2 G2(float c, float s);

HDFUNC void PolarDecomposition2D(cmat2& A, cmat2& R, cmat2& S);
HDFUNC void SVD2D(cmat2& A, cmat2& U, float& sigma1, float& sigma2, cmat2& V);
HDFUNC cmat3 TopLeftAssemble(cmat2& A);
HDFUNC cmat3 BotRightAssemble(cmat2& A);
HDFUNC void SolveReducedTopLeft(cmat3& B, cmat3& U, cmat3& S, cmat3& V);
HDFUNC void SolveReducedBotRight(cmat3& B, cmat3& U, cmat3& S, cmat3& V);
HDFUNC void SortWithTopLeftSub(cmat3& U, cmat3& S, cmat3& V);
HDFUNC void SortWithBotRightSub(cmat3& U, cmat3& S, cmat3& V);
HDFUNC void PostProcess(cmat3& B, cmat3& U, cmat3& V, cmat3& S, float* a, float* b, float tau);
HDFUNC void SVD(const cmat3& A, cmat3& U, cmat3& S, cmat3 V);

HDFUNC __inline__ cmat3 TensorProduct(cfloat3& a, cfloat3& b) {
	cmat3 res;
	res[0][0] = a.x*b.x; res[0][1] = a.x*b.y; res[0][2] = a.x*b.z;
	res[1][0] = a.y*b.x; res[1][1] = a.y*b.y; res[1][2] = a.y*b.z;
	res[2][0] = a.z*b.x; res[2][1] = a.z*b.y; res[2][2] = a.z*b.z;
	return res;
}

HDFUNC __inline__ void  mat3add(cmat3& a, cmat3& b, cmat3& c) {
	for (int i=0; i<9; i++)
		c.data[i] = a.data[i]+b.data[i];
}

HDFUNC __inline__ void  mat3sub(cmat3& a, cmat3& b, cmat3& c) {
	for (int i=0; i<9; i++)
		c.data[i] = a.data[i]-b.data[i];
}

HDFUNC __inline__ void  mat3prod(cmat3& a, cmat3& b, cmat3& c) {
	c[0][0] = a[0][0]*b[0][0]+a[0][1]*b[1][0]+a[0][2]*b[2][0];
	c[0][1] = a[0][0]*b[0][1]+a[0][1]*b[1][1]+a[0][2]*b[2][1];
	c[0][2] = a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]*b[2][2];

	c[1][0] = a[1][0]*b[0][0]+a[1][1]*b[1][0]+a[1][2]*b[2][0];
	c[1][1] = a[1][0]*b[0][1]+a[1][1]*b[1][1]+a[1][2]*b[2][1];
	c[1][2] = a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]*b[2][2];

	c[2][0] = a[2][0]*b[0][0]+a[2][1]*b[1][0]+a[2][2]*b[2][0];
	c[2][1] = a[2][0]*b[0][1]+a[2][1]*b[1][1]+a[2][2]*b[2][1];
	c[2][2] = a[2][0]*b[0][2]+a[2][1]*b[1][2]+a[2][2]*b[2][2];
}

HDFUNC __inline__ void mat2prod(cmat2& a, cmat2& b, cmat2& c)
{
	c[0][0] = a[0][0] * b[0][0] + a[0][1] * b[1][0];
	c[0][1] = a[0][0] * b[0][1] + a[0][1] * b[1][1];

	c[1][0] = a[1][0] * b[0][0] + a[1][1] * b[1][0];
	c[1][1] = a[1][0] * b[0][1] + a[1][1] * b[1][1];
}

HDFUNC __inline__ void mat3transpose(cmat3& a, cmat3& b) {
	cmat3 tmp;
	tmp[0][0]=a[0][0]; tmp[0][1]=a[1][0]; tmp[0][2]=a[2][0];
	tmp[1][0]=a[0][1]; tmp[1][1]=a[1][1]; tmp[1][2]=a[2][1];
	tmp[2][0]=a[0][2]; tmp[2][1]=a[1][2]; tmp[2][2]=a[2][2];
	b=tmp;
}

/*
Compute the product of cmat3 m and cfloat3 v,
storing the result into cfloat3 c.
c and v can be the same vector.
*/
HDFUNC __inline__ void  mvprod(cmat3& m, cfloat3& v, cfloat3& c) {
	cfloat3 tmp;
	tmp.x = m.data[0]*v.x + m.data[1]*v.y + m.data[2]*v.z;
	tmp.y = m.data[3]*v.x + m.data[4]*v.y + m.data[5]*v.z;
	tmp.z = m.data[6]*v.x + m.data[7]*v.y + m.data[8]*v.z;
	c=tmp;
}



void RotateX(cfloat3& a, float b);
void RotateY(cfloat3& a, float b);
void RotateZ(cfloat3& a, float b);
void RotateXYZ(cfloat3& a, cfloat3& xyz);

#define EPSILON 1e-10

HDFUNC __inline__ void AxisAngle2Matrix(cfloat3 axis, float angle, cmat3& mat)
{
	//Do the normalization if needed.
	float l = axis.Norm();
	if (fabs(l-1)>EPSILON)
		axis /= l;
	float s = sin(angle);
	float c = cos(angle);
	float t = 1 - c;
	mat[0][0] = t*axis.x*axis.x + c;
	mat[1][1] = t*axis.y*axis.y + c;
	mat[2][2] = t*axis.z*axis.z + c;

	float tmp1 = axis.x*axis.y*t;
	float tmp2 = axis.z*s;
	mat[1][0] = tmp1+tmp2;
	mat[0][1] = tmp1-tmp2;
	tmp1 = axis.x*axis.z*t;
	tmp2 = axis.y*s;
	mat[2][0] = tmp1 - tmp2;
	mat[0][2] = tmp1 + tmp2;
	tmp1 = axis.y*axis.z*t;
	tmp2 = axis.x*s;
	mat[2][1] = tmp1+tmp2;
	mat[1][2] = tmp1-tmp2;
}



/* -----------------------------------


	            OpenGL Utility


-------------------------------------*/

const float cPI = 3.1415926536;

struct cvertex{
	float position[4];
	float color[4];
};

struct cfloat4{
	float x,y,z,w;
	HDFUNC void Set(float _x,float _y,float _z,float _w){
		x=_x; y=_y; z=_z; w=_w;
	}
	HDFUNC cfloat4():
		x(0),y(0),z(0),w(0){}
	HDFUNC cfloat4(float _x, float _y, float _z, float _w):
		x(_x),y(_y),z(_z),w(_w){}
	HDFUNC cfloat4(cfloat3 x3, float _w):
		x(x3.x), y(x3.y), z(x3.z), w(_w){}
};

//watch out, column major
struct cmat4{
	float data[16];
	float* operator[](int i){
		return &data[i*4];
	}
	cmat4 operator*(cmat4& b){
		cmat4 tmp;
		for(int i=0;i<4;i++){
			for(int j=0;j<4;j++){
				tmp[i][j] = data[0+i*4]*b[0][j] + data[1+i*4]*b[1][j]
					+ data[2+i*4]*b[2][j] + data[3+i*4]*b[3][j];
			}
		}
		return tmp;
	}

	cfloat4 operator*(cfloat4& b) {
		cfloat4 tmp;
		tmp.x = data[0*4+0]*b.x + data[1*4+0]*b.y + data[2*4+0]*b.z + data[3*4+0]*b.w;
		tmp.y = data[0*4+1]*b.x + data[1*4+1]*b.y + data[2*4+1]*b.z + data[3*4+1]*b.w;
		tmp.z = data[0*4+2]*b.x + data[1*4+2]*b.y + data[2*4+2]*b.z + data[3*4+2]*b.w;
		tmp.w = data[0*4+3]*b.x + data[1*4+3]*b.y + data[2*4+3]*b.z + data[3*4+3]*b.w;
		return tmp;
	}
	
};


extern const cmat4 IDENTITY_MAT;

float cotangent(float angle);
float deg2rad(float deg);
float rad2deg(float rad);

void RotateAboutX(cmat4& m, float ang);
void RotateAboutY(cmat4& m, float ang);
void RotateAboutZ(cmat4& m, float ang);
void ScaleMatrix(cmat4& m, cfloat3 x);
void TranslateMatrix(cmat4& m, cfloat3 x);

cmat4 CreateProjectionMatrix(float fovy, float aspect_ratio,
	float near_plane, float far_plane);

typedef 
	unsigned char
 uchar;

class keystack{
public:
	unsigned char buffer[32];
	unsigned char maxlen,lenth;
	keystack(){
		maxlen = 32;
		lenth = 0;
		memset(buffer, 0, sizeof(buffer));
	}
	int findkey(uchar key){
		int tmp = -1;
		for (int i=0; i<lenth; i ++) {
			if (buffer[i]==key) {
				return i;
			}
		}
		return tmp;
	}
	void push(uchar key){
		if(lenth==31){
			printf("key stack full\n");
			return;
		}
		int id = findkey(key);
		if (id>=0)
			printf("key exists.\n");
		else{
			buffer[lenth] = key;
			lenth += 1;
		}	
	}

	void pop(uchar key){
		int id = findkey(key);
		if(id<0)
			printf("key to pop unfound.\n");
		else{
			for(int i=id; i<lenth-1;i++){
				buffer[i] = buffer[i+1];
			}
			lenth -= 1;
		}
	}
};


class cCamera{
public:

	cmat4 projmat;
	cmat4 viewmat;

	//projection matrix
	float fovy, aspect_ratio, nearclip, farclip;

	cfloat3 dir,up,right,forceup;
	cfloat3 pos, target;
	
	//keyboard translate
	cfloat3 vel;
	float velf, velMax;
	keystack wasd;

	//drag rotation
	cfloat2 rotatexy;
	cfloat2 transxy;
	//drag zoom
	float zoomin;
	
	cCamera(){
		velf = 0.0f;
		rotatexy.Set(0,0);
		transxy.Set(0,0);
	}

	//ViewMatrix Utility
	cmat4 CameraRotateMat();
	void CameraTranslateMat(cmat4& rotation, cfloat3& pos);
	
	//ProjectionMatrix Utility
	void SetProjParam(float f,float a,float nc,float fc){
		fovy=f; aspect_ratio=a; nearclip=nc; farclip=fc;
	}
	void ProjectionMat(){
		projmat = CreateProjectionMatrix(fovy, aspect_ratio, nearclip, farclip);
	}


	void lookat(cfloat3& _pos, cfloat3& _target);

	//Moving Utility
	void MoveCamera(cfloat3 dx){
		pos = pos+dx;
		target = target+dx;
		lookat(pos, target);
	}

#define FRONT_VEL 0
#define RIGHT_VEL 1
#define UP_VEL 2

	void getVelDir(){
		if(wasd.lenth==0)
			return;

		switch (wasd.buffer[wasd.lenth-1]) {
		case 'w': case 'W':
			vel = dir * (-1);
			break;
		case 's': case 'S':
			vel = dir ;
			break;
		case 'd': case 'D':
			vel = right ;
			break;
		case 'a': case 'A':
			vel = right * (-1);
			break;
		case 'q': case 'Q':
			vel = up;
			break;
		case 'z': case 'Z':
			vel = up * (-1);
			break;
		}
	}

	void SetVelocty(unsigned char key){
		key = tolower(key);
		wasd.push(key);
		
	}

	void UnSetVelocity(unsigned char key){
		key = tolower(key);
		wasd.pop(key);
	}

	void AdvanceCamera(float dt);


};


struct vertex {
	cfloat3 pos;
	cfloat4 color;
};

class GivensRotation {
public:
	int rowi;
	int rowk;
	float c;
	float s;

	HDFUNC inline GivensRotation(int rowi_in, int rowk_in) : rowi(rowi_in), rowk(rowk_in), c(1), s(0) {}
	HDFUNC inline GivensRotation(float a, float b, int rowi_in, int rowk_in) : rowi(rowi_in), rowk(rowk_in) { compute(a, b); }
	HDFUNC inline void transposeInPlace() { s = -s; }
	HDFUNC inline void compute(const float a, const float b)
	{
		float d = a * a + b * b;
		c = 1;
		s = 0;
		if (!ZERO(d))
		{
			float t = 1 / sqrt(d);
			c = a * t;
			s = -b * t;
		}
	}
	HDFUNC inline void computeUnconventional(const float a, const float b)
	{
		float d = a * a + b * b;
		c = 0;
		s = 1;
		if (d != 0) {
			float t = 1 / sqrt(d);
			s = a * t;
			c = b * t;
		}
	}
	HDFUNC inline void fill2(cmat2& R) const
	{
		R[rowi][rowi] = c;
		R[rowk][rowi] = -s;
		R[rowi][rowk] = s;
		R[rowk][rowk] = c;
	}
	HDFUNC inline void fill3(cmat3& R) const
	{
		R.Identity();
		R[rowi][rowi] = c;
		R[rowk][rowi] = -s;
		R[rowi][rowk] = s;
		R[rowk][rowk] = c;
	}

	HDFUNC inline void rowRotation2(cmat2& A) const
	{
		for (int j = 0; j < 2; j++) {
			float tau1 = A[rowi][j];
			float tau2 = A[rowk][j];
			A[rowi][j] = c * tau1 - s * tau2;
			A[rowk][j] = s * tau1 + c * tau2;
		}
	}

	HDFUNC inline void rowRotation3(cmat3& A) const
	{
		for (int j = 0; j < 3; j++) {
			float tau1 = A[rowi][j];
			float tau2 = A[rowk][j];
			A[rowi][j] = c * tau1 - s * tau2;
			A[rowk][j] = s * tau1 + c * tau2;
		}
	}

	HDFUNC inline void columnRotation2(cmat2& A) const
	{
		for (int j = 0; j < 2; j++) {
			float tau1 = A[j][rowi];
			float tau2 = A[j][rowk];
			A[j][rowi] = c * tau1 - s * tau2;
			A[j][rowk] = s * tau1 + c * tau2;
		}
	}
	HDFUNC inline void columnRotation3(cmat3& A) const
	{
		for (int j = 0; j < 3; j++) {
			float tau1 = A[j][rowi];
			float tau2 = A[j][rowk];
			A[j][rowi] = c * tau1 - s * tau2;
			A[j][rowk] = s * tau1 + c * tau2;
		}
	}

	HDFUNC inline void operator*=(const GivensRotation& A)
	{
		float new_c = c * A.c - s * A.s;
		float new_s = s * A.c + c * A.s;
		c = new_c;
		s = new_s;
	}

	HDFUNC inline GivensRotation operator*(const GivensRotation& A) const
	{
		GivensRotation r(*this);
		r *= A;
		return r;
	}
};

struct adjacent
{
	int up;
	int down;
	int left;
	int right;
	int upright;
	int downright;
	int downleft;
	int upleft;
	int upup;
	int downdown;
	int leftleft;
	int rightright;

	HDFUNC int& operator[](int index)
	{
		switch (index)
		{
		case 0: return up;
		case 1: return down;
		case 2: return left;
		case 3: return right;
		case 4: return upright;
		case 5: return downright;
		case 6: return downleft;
		case 7: return upleft;
		case 8: return upup;
		case 9: return downdown;
		case 10: return leftleft;
		case 11: return rightright;
		}
	}
};
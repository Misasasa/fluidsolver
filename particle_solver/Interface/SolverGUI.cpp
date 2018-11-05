
#include "SolverGUI.h"
#include "solver/pbfsolver.h"
#include "SOIL.h"

#ifdef BUILD_CUDA
#include "cuda_runtime.h"
#endif


float window_width  = 1024;
float window_height = 768;


float		light_fov;

										// Mouse control
#define DRAG_OFF		0				// mouse states
#define DRAG_LEFT		1
#define DRAG_RIGHT		2
#define DRAG_UP			3
#define DRAG_DOWN		4
#define DRAG_MIDDLE     5
int		last_x = -1, last_y = -1;		// mouse vars
int		mode = 0;
int		dragging = 0;


extern SolverGUI solverGUI;

//void SolverGUI::drawScene(float* viewmat, bool bShade)
//{
//	if (iShade <= 1 && bShade) {
//
//		glEnable(GL_LIGHTING);
//		glEnable(GL_LIGHT0);
//		glDisable(GL_COLOR_MATERIAL);
//
//		Vector4DF amb, diff, spec;
//		float shininess = 5.0;
//
//		glColor3f(1, 1, 1);
//		glLoadIdentity();
//		glLoadMatrixf(viewmat);
//
//		float pos[4];
//		pos[0] = light[0].x;
//		pos[1] = light[0].y;
//		pos[2] = light[0].z;
//		pos[3] = 1;
//		amb.Set(0, 0, 0, 1); diff.Set(1, 1, 1, 1); spec.Set(1, 1, 1, 1);
//		glLightfv(GL_LIGHT0, GL_POSITION, (float*)&pos[0]);
//		glLightfv(GL_LIGHT0, GL_AMBIENT, (float*)&amb.x);
//		glLightfv(GL_LIGHT0, GL_DIFFUSE, (float*)&diff.x);
//		glLightfv(GL_LIGHT0, GL_SPECULAR, (float*)&spec.x);
//
//		amb.Set(0, 0, 0, 1); diff.Set(.3, .3, .3, 1); spec.Set(.1, .1, .1, 1);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, (float*)&amb.x);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, (float*)&diff.x);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (float*)&spec.x);
//		glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, (float*)&shininess);
//
//
//		//glColorMaterial ( GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE );
//
//		glLoadMatrixf(viewmat);
//		//绘制平面
//#ifdef DRAW_PLANE
//		glBegin(GL_QUADS);
//		glNormal3f(0, 1, 0.001);
//		for (float x=-1000; x <= 1000; x += 100.0) {
//			for (float y=-1000; y <= 1000; y += 100.0) {
//				glVertex3f(x, 0.0, y);
//				glVertex3f(x+100, 0.0, y);
//				glVertex3f(x+100, 0.0, y+100);
//				glVertex3f(x, 0.0, y+100);
//			}
//		}
//		glEnd();
//#endif
//
//#ifdef DRAW_LINE
//
//#ifdef DRAW_PLANE
//		glColor3f(0.1, 0.1, 0.2);
//#else
//		glColor3f(0.6, 0.6, 0.6);
//#endif
//
//		glDisable(GL_LIGHTING);
//		glBegin(GL_LINES);
//		//绘制网格线
//		for (float n=-100; n <= 100; n += 10.0) {
//			glVertex3f(-100, 0.1, n);
//			glVertex3f(100, 0.1, n);
//			glVertex3f(n, 0.1, -100);
//			glVertex3f(n, 0.1, 100);
//		}
//		glVertex3f(light[0].x, light[0].y, 0);
//		glVertex3f(light[0].x, light[0].y, light[0].z);
//		glEnd();
//
//		psys->Draw(cam, 0.8);				// Draw particles		
//#endif
//	}
//
//}



//void SolverGUI::draw2D()
//{
//
//	glDisable(GL_LIGHTING);
//	glDisable(GL_DEPTH_TEST);
//
//	glMatrixMode(GL_PROJECTION);
//	glLoadIdentity();
//	glScalef(2.0/window_width, -2.0/window_height, 1);		// Setup view (0,0) to (800,600)
//	glTranslatef(-window_width/2.0, -window_height/2, 0.0);
//
//
//	float view_matrix[16];
//	glMatrixMode(GL_MODELVIEW);
//	glLoadIdentity();
//
//	char disp[200];
//
//	/*psys->getModeClr ();
//	strcpy ( disp, psys->getModeStr().c_str() ); drawText ( 20, 40, disp );*/
//
//	glColor4f(1.0, 1.0, 1.0, 1.0);
//	strcpy(disp, "Press H for help.");		drawText(10, 20, disp);
//
//	glColor4f(1.0, 1.0, 0.0, 1.0);
//	strcpy(disp, "");
//	if (psys->GetToggle(PCAPTURE)) strcpy(disp, "CAPTURING VIDEO");
//	drawText(200, 20, disp);
//	if (psys->GetYan(START_OUTPUT) == 1) strcpy(disp, "OUTPUTING DATA");
//	drawText(500, 20, disp);
//	if (bHelp) {
//
//		sprintf(disp, "Mode:                %s", psys->getModeStr().c_str());					drawText(20, 40, disp);
//
//		sprintf(disp, "Scene:               %s (id: %d)", psys->getSceneName().c_str(), (int)psys->GetParam(PEXAMPLE));				drawText(20, 60, disp);
//
//		sprintf(disp, "# Particles:         %d", psys->NumPoints());					drawText(20, 80, disp);
//
//		sprintf(disp, "Grid Density:        %f", psys->GetParam(PGRID_DENSITY));		drawText(20, 100, disp);
//		
//		//sprintf(disp, "Grid Count:          %f", (float)psys->GetParam(PSTAT_GRIDCNT) / psys->GetParam(PSTAT_OCCUPY));	drawText(20, 110, disp);
//		//sprintf(disp, "Grid Occupancy:      %f%%", (float)psys->GetParam(PSTAT_OCCUPY) / psys->getGridTotal());		drawText(20, 130, disp);
//		
//		sprintf(disp, "Grid Resolution:     %d x %d x %d (%d)", (int)psys->GetGridRes().x, (int)psys->GetGridRes().y, (int)psys->GetGridRes().z, psys->getGridTotal());		drawText(20, 140, disp);
//		
//		int nsrch = pow(psys->getSearchCnt(), 1/3.0);
//		sprintf(disp, "Grid Search:         %d x %d x %d", nsrch, nsrch, nsrch);			drawText(20, 150, disp);
//		//sprintf(disp, "Search Count:        %d, ave: %f, max: %f", (int)psys->GetParam(PSTAT_SRCH), psys->GetParam(PSTAT_SRCH)/psys->NumPoints(), psys->GetParam(PSTAT_SRCHMAX)/psys->NumPoints());		drawText(20, 160, disp);
//		//sprintf(disp, "Neighbor Count:      %d, ave: %f, max: %f", (int)psys->GetParam(PSTAT_NBR), psys->GetParam(PSTAT_NBR)/psys->NumPoints(), psys->GetParam(PSTAT_NBRMAX)/psys->NumPoints());		drawText(20, 170, disp);
//		//sprintf(disp, "Search Overhead:     %.2fx", psys->GetParam(PSTAT_SRCH)/psys->GetParam(PSTAT_NBR));		drawText(20, 180, disp);
//
//		//sprintf(disp, "Insert Time:         %.3f ms", psys->GetParam(PTIME_INSERT));			drawText(20, 200, disp);
//		sprintf(disp, "Sort Time:           %.3f ms", psys->GetParam(PTIME_SORT));			drawText(20, 210, disp);
//		//sprintf(disp, "Count Time:          %.3f ms", psys->GetParam(PTIME_COUNT));			drawText(20, 220, disp);
//		sprintf(disp, "Pressure Time:       %.3f ms", psys->GetParam(PTIME_PRESS));			drawText(20, 230, disp);
//		sprintf(disp, "Force Time:          %.3f ms", psys->GetParam(PTIME_FORCE));			drawText(20, 240, disp);
//		sprintf(disp, "Advance Time:        %.3f ms", psys->GetParam(PTIME_ADVANCE));			drawText(20, 250, disp);
//
//		//multi fluid
//		float st = psys->GetParam(PTIME_SORT) + psys->GetParam(PTIME_PRESS)+psys->GetParam(PTIME_FORCE) + psys->GetParam(PTIME_ADVANCE) + psys->GetParam(PTIMEDRIFTVEL) + psys->GetParam(PTIMEALPHA) + psys->GetParam(PTIMECORR);
//
//		//for performance
//		if (!bPause) {
//			mint::Time nowTime;
//			nowTime.SetSystemTime(ACC_MSEC);
//			double now = nowTime.GetMSec();
//			if (psys->lastTime == 0) {
//				psys->lastTime = now;
//			}
//			dst = now - psys->lastTime;
//			if (dst == 0)
//				dst = 1.0f;
//			psys->lastTime = now;
//		}
//		sprintf(disp, "Time Performance:      %.3f ms, %.1f fps", dst, tm_fps);									drawText(20, 260, disp);
//		sprintf(disp, "Simulation Performance: %.3f ms", st); drawText(20, 270, disp);
//		//sprintf ( disp,	"Performance:         %d particles/sec", (int) ((psys->NumPoints()*1000.0)/dst) );			drawText ( 20, 270,  disp );
//
//		//sprintf(disp, "Particle Memory:     %.4f MB", (float)psys->GetParam(PSTAT_PMEM)/1000000.0f);		drawText(20, 290, disp);
//		//sprintf(disp, "Grid Memory:         %.4f MB", (float)psys->GetParam(PSTAT_GMEM)/1000000.0f);		drawText(20, 300, disp);
//
//		size_t fm, tm;
//		cudaMemGetInfo(&fm, &tm);
//		sprintf(disp, "GPU Free Memory:     %u MB", fm/1000000); drawText(20, 310, disp);
//		sprintf(disp, "GPU Total Memory:    %u MB", tm/1000000); drawText(20, 320, disp);
//		sprintf(disp, "GPU Memory Load:     %.2f%%",  (1-(float)fm/tm)*100); drawText(20,330, disp);
//		//multi fluid
//		
//		//sprintf(disp, "Drift Velocity Time:        %.3f ms", psys->GetParam(PTIMEDRIFTVEL));			drawText(20, 330, disp);
//		//sprintf(disp, "Alpha Advance Time:        %.3f ms", psys->GetParam(PTIMEALPHA));			drawText(20, 340, disp);
//		//sprintf(disp, "Alpha Correct Time:        %.3f ms", psys->GetParam(PTIMECORR));			drawText(20, 350, disp);
//		//sprintf(disp, "Split Time:                %.3f ms", psys->GetParam(PTIMESPLIT));			drawText(20, 360, disp);
//		sprintf(disp, "Frame NO:                  %d ", psys->frameNum());			drawText(20, 370, disp);
//	}
//}



void display()
{
	solverGUI.render();
}

void resize(int width, int height)
{
	// set window height and width
	window_width  = (float)width;
	window_height = (float)height;
	glViewport(0, 0, width, height);

	solverGUI.ReSize(width,height);
}

void SolverGUI::ReSize(int width, int height) {

	camera.SetProjParam(40, (float)width/height, 0.01f, 10.0f);
	camera.ProjectionMat();
}


void keyboard_func(unsigned char key, int x, int y)
{
	solverGUI.keyDown(key);
}


void SolverGUI::keyDown(unsigned char key){

	switch (key) {
	case ' ':
		bPause = !bPause;
		break;
	//case 'h': case 'H':	
	//	bHelp = !bHelp; break;

	case 'a': case 'A':		
	case 'd': case 'D':		
	case 'w': case 'W':		
	case 's': case 'S':		
	case 'q': case 'Q':		
	case 'z': case 'Z':		
		camera.SetVelocty(key);
		break;

	case 'c':
		PrintCameraParam();
		break;
	case 27:
		exit(0); break;
	case 'p':
		printf("take a snapshot\n");
		takeSnapshot();
		break;
	case '`':				
		bTakeSnapshot = !bTakeSnapshot;
		if(bTakeSnapshot)
			printf("taking snapshot\n");
		else
			printf("stop taking snapshot\n");
		break;
	default:
		break;
	}

	solver->HandleKeyEvent(key);
}

void SolverGUI::PrintCameraParam() {

	//convert from GL to 3D
	cfloat3 origin(camera.pos.x, -camera.pos.z, camera.pos.y);
	cfloat3 target(camera.target.x,-camera.target.z,camera.target.y);
	cfloat3 up(camera.up.x,-camera.up.z,camera.up.y);

	FILE* foutput = fopen("camera param.xml","w+");
	fprintf(foutput,"<float name=\"fov\" value=\"%f\"/>\n",camera.fovy);
	fprintf(foutput, "<float name=\"nearClip\" value=\"%f\"/>\n", camera.nearclip);
	fprintf(foutput, "<float name=\"farClip\" value=\"%f\"/>\n", camera.farclip);
	
	fprintf(foutput,"<transform name=\"toWorld\">\n");
	fprintf(foutput,"<lookat ");
	fprintf(foutput," origin=\" %f,%f,%f\" ",origin.x,origin.y,origin.z);
	fprintf(foutput,"target=\"%f,%f,%f\" ", target.x, target.y,target.z);
	fprintf(foutput,"up=\"%f,%f,%f\"/>\n",up.x,up.y,up.z);
	fprintf(foutput,"</transform>\n");
	fclose(foutput);
}

void keyUpFunc(unsigned char key, int x,int y){
	solverGUI.keyUp(key);	
}
void SolverGUI::keyUp(unsigned char key){
	switch (key) {
	case 'a': case 'A':
	case 'd': case 'D':
	case 'w': case 'W':
	case 's': case 'S':
	case 'q': case 'Q':
	case 'z': case 'Z':
		camera.UnSetVelocity(key);
		break;
	}
}


//Vector3DF cangs;
//Vector3DF ctp;
float cdist;

#define GLUT_WHEEL_UP	3      
#define GLUT_WHEEL_DOWN 4  

void mouse_click_func(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		
		if (button == GLUT_LEFT_BUTTON)		
			dragging = DRAG_LEFT;
		else if (button == GLUT_RIGHT_BUTTON) 
			dragging = DRAG_RIGHT;
		else if (button == GLUT_MIDDLE_BUTTON)
			dragging = DRAG_MIDDLE;


		else if(button == GLUT_WHEEL_UP){
			
		}
		else if(button == GLUT_WHEEL_DOWN){
			
		}

	}
	else if (state==GLUT_UP) { //wheel up and down
		if (button == GLUT_WHEEL_UP)
			dragging = DRAG_UP;
		else if (button == GLUT_WHEEL_DOWN)
			dragging = DRAG_DOWN;
		else{
			dragging = DRAG_OFF;
		}
	}

	last_x = x;
	last_y = y;

	solverGUI.MouseClick(x,y,state,button);
}


void mouse_drag_func(int x,int y){
	int dx = x - last_x;
	int dy = y - last_y;
	last_x = x;
	last_y = y;
	solverGUI.drag(dx,dy);
}

void SolverGUI::MouseClick(int x,int y,int state,int button){
	if(state==GLUT_DOWN){
		if(button == GLUT_WHEEL_UP || button == GLUT_WHEEL_DOWN){
			if(button == GLUT_WHEEL_UP)
				camera.zoomin = camera.zoomin * 0.8 + 0.2 * 50;
			else
				camera.zoomin = camera.zoomin * 0.8 - 0.2 * 50;
			camera.fovy += camera.zoomin * 0.5;
			camera.fovy = fmin(100, fmax(10, camera.fovy));
			camera.ProjectionMat();
		}
	}
}

void SolverGUI::drag(int dx,int dy){

	//small displacement increment scheme
	if(dragging == DRAG_LEFT){
		float alpha = 0.4;
		camera.rotatexy = camera.rotatexy *(1-alpha) + cfloat2(dx,dy)*alpha;
	}else if(dragging == DRAG_RIGHT){
		camera.zoomin = camera.zoomin * 0.8 + 0.2 * dy;
		camera.fovy += camera.zoomin * 0.5;
		camera.fovy = fmin(100, fmax(10, camera.fovy));
		camera.ProjectionMat();
	}else if(dragging == DRAG_MIDDLE){
		float alpha = 0.4;
		camera.transxy = camera.transxy *(1-alpha) + cfloat2(dx, dy)*alpha;
	}
}


void idle_func()
{
	glutPostRedisplay();
}


#include "GL/glew.h"
#include "GL/freeglut.h"



//void SolverGUI::CreateCube(){
//	const cvertex VERTICES[8] =
//	{
//		{{-.5f, -.5f,  .5f, 1},{0, 0, 1, 1}},
//		{{-.5f,  .5f,  .5f, 1},{1, 0, 0, 1}},
//		{{.5f,  .5f,  .5f, 1},{0, 1, 0, 1}},
//		{{.5f, -.5f,  .5f, 1},{1, 1, 0, 1}},
//		{{-.5f, -.5f, -.5f, 1},{1, 1, 1, 1}},
//		{{-.5f,  .5f, -.5f, 1},{1, 0, 0, 1}},
//		{{.5f,  .5f, -.5f, 1},{1, 0, 1, 1}},
//		{{.5f, -.5f, -.5f, 1},{0, 0, 1, 1}}
//	};
//	const GLuint INDICES[36] =
//	{
//		0,2,1,  0,3,2,
//		4,3,0,  4,7,3,
//		4,1,5,  4,0,1,
//		3,6,2,  3,7,6,
//		1,6,5,  1,2,6,
//		7,5,6,  7,4,5
//	};
//
//	ShaderIds[0] = glCreateProgram();
//	ExitOnGLError("ERROR: Could not create the shader program");
//
//	ShaderIds[1] = LoadShader("shader/SimpleShader.fragment.glsl",	GL_FRAGMENT_SHADER);
//
//	//ShaderIds[2] = LoadShader("shader/SimpleShader.geometry.glsl",	GL_GEOMETRY_SHADER);
//	ShaderIds[2] = LoadShader("shader/Cubic.geometry.glsl", GL_GEOMETRY_SHADER);
//
//	ShaderIds[3] = LoadShader("shader/SimpleShader.vertex.glsl",	GL_VERTEX_SHADER);
//	
//	glAttachShader(ShaderIds[0], ShaderIds[1]);
//	glAttachShader(ShaderIds[0], ShaderIds[2]);
//	glAttachShader(ShaderIds[0], ShaderIds[3]);
//
//	glLinkProgram(ShaderIds[0]);
//	ExitOnGLError("ERROR: Could not link the shader program");
//
//	ModelMatrixUniformLocation = glGetUniformLocation(ShaderIds[0], "ModelMatrix");
//	ViewMatrixUniformLocation = glGetUniformLocation(ShaderIds[0], "ViewMatrix");
//	ProjectionMatrixUniformLocation = glGetUniformLocation(ShaderIds[0], "ProjectionMatrix");
//	ParticleSizeUniformLocation = glGetUniformLocation(ShaderIds[0],"particle_size");
//	ExitOnGLError("ERROR: Could not get the shader uniform locations");
//
//	glGenBuffers(2, &BufferIds[1]);
//	ExitOnGLError("ERROR: Could not generate the buffer objects");
//
//	glGenVertexArrays(1, &BufferIds[0]);
//	ExitOnGLError("ERROR: Could not generate the VAO");
//	glBindVertexArray(BufferIds[0]);
//	ExitOnGLError("ERROR: Could not bind the VAO");
//
//	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);
//	ExitOnGLError("ERROR: Could not enable vertex attributes");
//
//	glBindBuffer(GL_ARRAY_BUFFER, BufferIds[1]);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES), VERTICES, GL_STATIC_DRAW);
//	ExitOnGLError("ERROR: Could not bind the VBO to the VAO");
//
//	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(VERTICES[0]), (GLvoid*)0);
//	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(VERTICES[0]), (GLvoid*)sizeof(VERTICES[0].position));
//	ExitOnGLError("ERROR: Could not set VAO attributes");
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, BufferIds[2]);
//	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(INDICES), INDICES, GL_STATIC_DRAW);
//	ExitOnGLError("ERROR: Could not bind the IBO to the VAO");
//
//	glBindVertexArray(0);
//
//
//	//Rendering Cubes
//	//glEnable(GL_DEPTH_TEST);
//	//glDepthFunc(GL_LESS);
//	//glEnable(GL_CULL_FACE);
//	//glCullFace(GL_BACK);
//	//glFrontFace(GL_CCW);
//	
//}

GLuint boxIndex[]={
	0,1, 1,2, 2,3, 3,0,
	0,4, 1,5, 2,6, 3,7,
	4,5, 5,6, 6,7, 7,4
};
uarr index(boxIndex, boxIndex+24);




SolverGUI::~SolverGUI(){
	
	if(particleRO)
		delete particleRO;
	if(cubeRO)
		delete cubeRO;
	if(geomRO)
		delete geomRO;

}


//void SolverGUI::DrawCube(){
//	float CubeAngle;
//	clock_t Now = clock();
//	if (LastTime == 0)
//		LastTime = Now;
//
//	CubeRotation += 45.0f * ((float)(Now - LastTime) / CLOCKS_PER_SEC);
//	CubeAngle = deg2rad(CubeRotation);
//	LastTime = Now;
//
//	ModelMatrix = IDENTITY_MAT;
//	RotateAboutY(ModelMatrix, CubeAngle);
//	RotateAboutX(ModelMatrix, CubeAngle);
//
//	glUseProgram(shaders[shaderId].programid);
//	ExitOnGLError("ERROR: Could not use the shader program");
//
//	glUniformMatrix4fv(ModelMatrixUniformLocation, 1, GL_FALSE, ModelMatrix.data);
//	glUniformMatrix4fv(ViewMatrixUniformLocation, 1, GL_FALSE, ViewMatrix.data);
//	glUniform1f(ParticleSizeUniformLocation, 0.05f);
//	ExitOnGLError("ERROR: Could not set the shader uniforms");
//
//	glBindVertexArray(BufferIds[0]);
//	ExitOnGLError("ERROR: Could not bind the VAO for drawing purposes");
//
//	//glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLvoid*)0);
//	//glPointSize(10);
//	//glBindTexture(GL_TEXTURE_2D, img.getID());
//
//	glDrawArrays(GL_POINTS, 0, 8);
//	ExitOnGLError("ERROR: Could not draw the cube");
//
//	glBindVertexArray(0);
//	glUseProgram(0);
//}


void SolverGUI::InitializeGL(int argc, char** argv) {
	// set up the window
	glutInit(&argc, &argv[0]);

	glutInitContextVersion(4, 0);
	glutInitContextFlags(GLUT_FORWARD_COMPATIBLE);
	glutInitContextProfile(GLUT_CORE_PROFILE);

	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH |GLUT_MULTISAMPLE);
	glutInitWindowSize((int)window_width, (int)window_height);
	glutCreateWindow("Multiphase Nekobus");


	// GL extensions
	GLenum GlewInitResult = glewInit();
	if (GLEW_OK != GlewInitResult) {
		fprintf(stderr, "ERROR: %s\n", glewGetErrorString(GlewInitResult));
		exit(EXIT_FAILURE);
	}
	fprintf(stdout, "INFO: OpenGL Version: %s\n", glGetString(GL_VERSION));


	//############   callbacks  ############
	glutDisplayFunc(display);
	glutReshapeFunc(resize);
	//glutCloseFunc(Destroy);
	glutIdleFunc(idle_func);

	// Input
	glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF);
	glutKeyboardFunc(keyboard_func);
	glutKeyboardUpFunc(keyUpFunc);

	glutMouseFunc(mouse_click_func);
	glutMotionFunc(mouse_drag_func);
	



	//#############   Camera Parameters   #############
	
	camera.forceup = cfloat3(0, 1, 0);
	camera.lookat(cfloat3(0, 0.45, 1.5), cfloat3(0, 0.35, -0.8));
	

	//=============  Render Object ==============
	if(loadparticleRO)
		particleRO = new ParticleRO();
	if(loadcubeRO)
		cubeRO = new CubeRO();
	if(loadtriangleRO)
		triangleRO = new TriangleRO();
	if(bDrawGeometry)
		geomRO = new GeometryRO();


	//#############   Rendering flags   #################
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.4, 0.4, 0.4, 1);
}

void cudaInit(int argc, char **argv)
{
	//CUT_DEVICE_INIT(argc, argv);

	cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);

	printf("-- CUDA --\n");
	printf("Name:       %s\n", p.name);
	printf("Revision:   %d.%d\n", p.major, p.minor);
	printf(" Mem: %d MB\n", p.totalGlobalMem/1000000);
	printf("Shared/Blk: %d\n", p.sharedMemPerBlock);
	printf("Regs/Blk:   %d\n", p.regsPerBlock);
	printf("Warp Size:  %d\n", p.warpSize);
	printf("Mem Pitch:  %d\n", p.memPitch);
	printf("Thrds/Blk:  %d\n", p.maxThreadsPerBlock);
	printf("Const Mem:  %d KB\n", p.totalConstMem/1000);
	printf("Clock Rate: %d\n", p.clockRate);
};

void SolverGUI::Initialize(int argc, char** argv) {
	// Initialize CUDA
#ifdef BUILD_CUDA
	cudaInit(argc, argv);
#endif

	rendermode = PARTICLE_RENDERER;
	InitializeGL(argc,argv);

	frameNo = 0;
	bPause = false;

	//vbuffer = new vertex[1];
	//vbuffer[0].pos = cfloat3(0,0,0);
	//vbuffer[0].color = cfloat4(1,1,1,1);
	//pnum = 1;
}

void SolverGUI::GetBoundingBox() {
	//hard code
	cfloat3 min = solver->domainMin;
	cfloat3 max = solver->domainMax;
	boundingBox.vertices.resize(8);

	boundingBox.vertices[0].pos.Set(min.x,max.y,min.z);
	boundingBox.vertices[1].pos.Set(max.x,max.y,min.z);
	boundingBox.vertices[2].pos.Set(max.x,max.y,max.z);
	boundingBox.vertices[3].pos.Set(min.x,max.y,max.z);

	boundingBox.vertices[4].pos.Set(min.x, min.y, min.z);
	boundingBox.vertices[5].pos.Set(max.x, min.y, min.z);
	boundingBox.vertices[6].pos.Set(max.x, min.y, max.z);
	boundingBox.vertices[7].pos.Set(min.x, min.y, max.z);

	for(int i=0;i<8;i++)
		boundingBox.vertices[i].color.Set(1,1,1,1);
	
	boundingBox.indices.resize(24);
	unsigned int id[24] = {0,1,1,2,2,3,3,0,
	4,5,5,6,6,7,7,4,    0,4,1,5,2,6,3,7};
	for(int i=0; i<24; i++)
		boundingBox.indices[i] = id[i];
}



void SolverGUI::MoveCamera() {
	clock_t Now = clock();
	if (LastTime == 0)
		LastTime = Now;

	//Update Camera Position
	camera.velMax = 2.5;
	camera.AdvanceCamera((float)(Now-LastTime)/1000.f);
	LastTime = Now;
}


void SolverGUI::setParticlesz(float sz) {
	particleRO->setParticleSize(sz);
}

int tenc=0;
float tsum=0;
float tshow=0;
char tmp[100];

float frametimer=0;

void SolverGUI::render() {

	if (!bPause) {

		catpaw::cTime clock;
		clock.tick();

		solver->step();
		
		tsum += clock.tack()*1000;
		tenc++;
		if (tenc==10) {
			tshow = tsum / 10; 
			tenc=tsum=0;
		}
		
		//sprintf(tmp,"%f fps %d",1000/tshow,frameNo);
		sprintf(tmp, "%f mspf %d", tshow, frameNo);
		glutSetWindowTitle(tmp);
		
		if (bTakeSnapshot) {
			frametimer += solver->dt;
			if (frametimer>1/60.0) {
				takeSnapshot();
				frametimer = 0;
			}
		}
			
		frameNo++;
	}
	
	MoveCamera();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	switch (rendermode) {
	case PARTICLE_RENDERER:
		particleRO->Draw(hPos->data(), hColor->data(), camera, solver->numP);
		break;
	/*
	case TRIANGLE_RENDERER:
		triangleRO->Draw(hPos->data(), hColor->data(), camera, hPos->size());
		break;
	case TRIANGLE_PARTICLE_RENDERER:
		triangleRO->Draw(hPos->data(), hColor->data(), camera, hPos->size());
		particleRO->Draw(hPos->data(), hColor->data(), camera, hPos->size());
		break;
	*/
	}
	if (bDrawGeometry) {
		geomRO->Draw(&boundingBox, camera);
	}

	//draw2D();

	glutSwapBuffers();
	glutPostRedisplay();

}

#include "png.h"
void SolverGUI::takeSnapshot()
{

	int width = window_width;
	int height = window_height;

	char fileName[64];
	sprintf(fileName, "../snapshot/snapshot_%04d.png", frameNo);
	// record frame buffer directly to image pixels
	uchar * pixels = new uchar[width*height*3];
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, (GLvoid*)pixels);

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
	if (!png)
		return;

	png_infop info = png_create_info_struct(png);
	if (!info) {
		png_destroy_write_struct(&png, &info);
		return;
	}

	FILE* fp = fopen(fileName, "wb");
	png_init_io(png, fp);
	png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
	png_colorp palette = (png_colorp)png_malloc(png, PNG_MAX_PALETTE_LENGTH * sizeof(png_color));
	if (!palette) {
		fclose(fp);
		png_destroy_write_struct(&png, &info);
		return;
	}
	png_set_PLTE(png, info, palette, PNG_MAX_PALETTE_LENGTH);
	png_write_info(png, info);
	png_set_packing(png);

	png_bytepp rows = (png_bytepp)png_malloc(png, height*sizeof(png_bytep));
	for (int i=0; i<height; i++)
		rows[i] = (png_bytep)(pixels + (height-i-1)*width*3);
	png_write_image(png, rows);
	png_write_end(png, info);
	png_free(png, palette);
	png_free(png, rows);
	png_destroy_write_struct(&png, &info);

	fclose(fp);

	delete pixels;

	return;
}


void SolverGUI::Run(){
	glutMainLoop();
}





void SolverGUI::Exit(){
}

void SolverGUI::LoadIndexedVBO(vector<unsigned int>& indexedvbo) {
	if(triangleRO==NULL)
		return;
	triangleRO->UpdateIndexedVBO(indexedvbo);
}


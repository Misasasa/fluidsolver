

#include "Interface/SolverGUI.h"
#include "solver/pbfsolver.h"
#include "grid.h"

using namespace catpaw;

SolverGUI solverGUI;
PBFSolver* solver;


void SetupPBFSolver() {
	solver = new PBFSolver();
	solver->setupBasicScene();

	solverGUI.LoadIndexedVBO(solver->indexedVBO);
	solverGUI.setRenderMode(TRIANGLE_PARTICLE_RENDERER);
	solverGUI.setParticlesz(1);
}

//unit test
int main_(int argc, char **argv) {

	solverGUI.Initialize(argc, argv);
	
	SetupPBFSolver();

	solverGUI.Run();

	return 0;
}

int main(int argc, char **argv) {

	solverGUI.Initialize(argc, argv);
	//solverGUI.setRenderMode(TRIANGLE_PARTICLE_RENDERER);

	GridSolver gs;
 	gs.setup();
	
	solverGUI.bindSolver(&gs);
	solverGUI.setParticlesz(0.005);

	solverGUI.Run();
	
	
}

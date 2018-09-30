

#include "Interface/SolverGUI.h"
#include "solver/pbfsolver.h"

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
int main(int argc, char **argv) {

	solverGUI.Initialize(argc, argv);
	
	SetupPBFSolver();

	solverGUI.Run();

	return 0;
}

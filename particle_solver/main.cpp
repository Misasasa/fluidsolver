

#include "Interface/SolverGUI.h"
#include "solver/MultiphaseSPHSolver.h"
#include "catpaw\geometry.h"

using namespace catpaw;

SolverGUI solverGUI;

void testSPH() {
	MultiphaseSPHSolver ss;
	ss.Setup();

	solverGUI.bindSolver(&ss);
	solverGUI.setParticlesz(0.005);

	solverGUI.Run();
}

extern int singularValueDecomposition(cmat3& A, cmat3& U, cfloat3& sigma, cmat3& V, float tol = 128 * 1e-10);

extern void singularValueDecomposition2(cmat2& A, GivensRotation& U, cfloat2& Sigma, GivensRotation& V, const float tol = 64 * 1e-10);

int main(int argc, char **argv) {

	solverGUI.Initialize(argc, argv);

	testSPH();

	return 0;
}

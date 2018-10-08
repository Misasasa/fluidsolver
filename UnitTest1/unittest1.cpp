#include "stdafx.h"
#include "CppUnitTest.h"
#include "../particle_solver/grid.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UnitTest1
{		
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(TestMethod1)
		{
			GridSolver gs;
			gs.loadConfig();
			gs.allocate();
		}

	};
}
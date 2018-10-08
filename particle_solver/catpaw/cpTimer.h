#ifndef CPTIMER
#define CPTIMER

#include <chrono>
#include <time.h>
#include <stdio.h>

using namespace std::chrono;
namespace catpaw{

	struct cTime{
		steady_clock::time_point t1,t2;
		
		cTime(){}

		inline void tick(){
			t1 = steady_clock::now();
		}

		inline double tack(){
			t2 = steady_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(t2-t1);
			return time_span.count();
		}
	};
	
}
#endif
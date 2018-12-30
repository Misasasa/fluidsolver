
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

typedef unsigned int uint;

#define EPSILON 1e-10
#define GRID_UNDEF 99999999

inline uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
inline void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = iDivUp(n, numThreads);
}
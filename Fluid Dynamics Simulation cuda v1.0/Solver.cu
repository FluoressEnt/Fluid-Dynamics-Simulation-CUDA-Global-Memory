#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "Solver.h"

void Solver::CalculateWrapper() {
	int numBlocks, threadsPerBlock;
	//threads per block maximum is 1024
	threadsPerBlock = 256;

	numBlocks = (ALENGTH + threadsPerBlock - 1) / threadsPerBlock;

	cCalculate <<<numBlocks, threadsPerBlock >>> (cSDens, cSVelX, cSVelY, cNewDens, cOldDens, cNewVelX, cNewVelY, cOldVelX, cOldVelY);
}

//needs rework to involve every threadID?
__global__ void cCalculate(float* cSDens, float* cSVelX, float* cSVelY, float* cNewDens, float* cOldDens, float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	//density step
	cAddSource(cNewDens, cSDens);
	__syncthreads();
	cSwap(&cOldDens, &cNewDens);
	__syncthreads();

	for (int GaussIterator = 0; GaussIterator < 20; GaussIterator++) {
		cDiffuse(1, cNewDens, cOldDens);
		__syncthreads();
	}

	cSwap(&cOldDens, &cNewDens);
	__syncthreads();

	cAdvection(cOldDens, cNewDens, cNewVelX, cNewVelY);
	__syncthreads();

	cSetBoundary(2, cNewDens);
	__syncthreads();

	//velocity step
	cAddSource(cNewVelX, cSVelX);
	__syncthreads();
	cAddSource(cNewVelX, cSVelX);
	__syncthreads();

	cSwap(&cOldVelX, &cNewVelX);
	__syncthreads();
	for (int GaussIterator = 0; GaussIterator < 20; GaussIterator++) {
		cDiffuse(1, cNewVelX, cOldVelX);
		__syncthreads();
	}
	cSwap(&cOldVelY, &cNewVelY);
	__syncthreads();
	for (int GaussIterator = 0; GaussIterator < 20; GaussIterator++) {
		cDiffuse(2, cNewVelY, cOldVelY);
		__syncthreads();
	}

	cProjectionInY(cNewVelX, cNewVelY, cOldVelX, cOldVelY);
	__syncthreads();
	for (int GaussIterator = 0; GaussIterator < 20; GaussIterator++) {
		cProjectionInX(cOldVelX, cOldVelY);
		__syncthreads();
	}
	cFinalProjection(cNewVelX, cNewVelY, cOldVelX);
	__syncthreads();

	cSwap(&cOldVelX, &cNewVelX);
	__syncthreads();
	cSwap(&cOldVelY, &cNewVelY);
	__syncthreads();

	cAdvection(cNewVelX, cOldVelX, cOldVelX, cOldVelY);
	__syncthreads();
	cAdvection(cNewVelY, cOldVelY, cOldVelX, cOldVelY);
	__syncthreads();

	cProjectionInY(cNewVelX, cNewVelY, cOldVelX, cOldVelY);
	__syncthreads();
	for (int GaussIterator = 0; GaussIterator < 20; GaussIterator++) {
		cProjectionInX(cOldVelX, cOldVelY);
		__syncthreads();
	}
	cFinalProjection(cNewVelX, cNewVelY, cOldVelX);
	__syncthreads();
}

__device__ void cAddSource(float *cNewDens, float *sourceArray) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if (ID < ALENGTH) {
		float temp;
		temp = (DT * sourceArray[ID]);
		cNewDens[ID] += temp;
	}
}

__device__ void cDiffuse(int b, float* cNewDens, float* cOldDens) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		//conservation of diffusivity
		float a = DT * VISC * RES * RES;
		cNewDens[ID] = (cOldDens[ID] + a * (cNewDens[cGetArrayPos(xPos - 1, yPos)] + cNewDens[cGetArrayPos(xPos + 1, yPos)]
			+ cNewDens[cGetArrayPos(xPos, yPos - 1)] + cNewDens[cGetArrayPos(xPos, yPos + 1)])) / (1 + 4 * a);

		cSetBoundary(b, cNewDens);
	}
}

__device__ void cAdvection(float* cNewDens, float* cOldDens, float* cVelX, float* cVelY) {
	int left, bottom, right, top, ID;
	float x, y, distToRight, distToTop, distToLeft, distToBottom;

	ID = blockIdx.x * blockDim.x + threadIdx.x;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {

		float dt0 = DT * RES;

		x = xPos - dt0 * cVelX[ID];
		y = yPos - dt0 * cVelY[ID];

		//neighbourhood of previous position
		if (x < 0.5) x = 0.5f;
		if (x > RES + 0.5) x = RES + 0.5f;
		left = (int)x;
		right = left + 1;

		if (y < 0.5) y = 0.5;
		if (y > RES + 0.5) y = RES + 0.5f;
		bottom = (int)y;
		top = bottom + 1;

		//interpolation part
		distToLeft = x - left;
		distToRight = 1 - distToLeft;
		distToBottom = y - bottom;
		distToTop = 1 - distToBottom;

		cNewDens[ID] = distToRight * (distToTop*cOldDens[cGetArrayPos(left, bottom)] + distToBottom * cOldDens[cGetArrayPos(left, top)])
			+ distToLeft * (distToTop*cOldDens[cGetArrayPos(right, bottom)] + distToBottom * cOldDens[cGetArrayPos(right, top)]);
	}
}

__device__ void cFinalProjection(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	float h = 1.0f / RES;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cNewVelX[ID] -= 0.5f*(cOldVelX[ID + 1] - cOldVelX[ID - 1]) / h;
		cNewVelY[ID] -= 0.5f*(cOldVelX[ID + RES + 2] - cOldVelX[ID - RES - 2]) / h;

		cSetBoundary(1, cNewVelX);
		cSetBoundary(2, cNewVelY);
	}
}
__device__ void cProjectionInY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	float h = 1.0f / RES;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cOldVelY[ID] = -0.5f * h * (cNewVelX[ID + 1] - cNewVelX[ID - 1]
			+ cNewVelY[ID + RES + 2] - cNewVelY[ID - RES - 2]);
		cOldVelX[ID] = 0;

		cSetBoundary(0, cOldVelY);
		cSetBoundary(0, cOldVelX);
	}
}
__device__ void cProjectionInX(float* cOldVelX, float* cOldVelY) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cOldVelX[ID] = (cOldVelY[ID] + cOldVelX[ID - 1]
			+ cOldVelX[ID + 1] + cOldVelX[ID - RES - 2] + cOldVelX[ID + RES + 2]) / 4;

		cSetBoundary(0, cOldVelX);
	}
}

__device__ void cSetBoundary(int b, float* boundArray) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	int x, y;
	x = cGetX(ID);
	y = cGetY(ID);

	if (x > 0 && x <= RES && y == 0) {
		boundArray[ID] = b == 2 ? -boundArray[ID + RES + 2] : boundArray[ID + RES + 2];
	}
	else if (x > 0 && x <= RES && y == RES + 1) {
		boundArray[ID] = b == 2 ? -boundArray[ID - RES - 2] : boundArray[ID - RES - 2];
	}
	else if (x == 1 && y > 0 && y <= RES) {
		boundArray[ID] = b == 1 ? -boundArray[ID + 1] : boundArray[ID + 1];
	}
	else if (x == RES + 1 && y > 0 && y <= RES) {
		boundArray[ID] = b == 1 ? -boundArray[ID - 1] : boundArray[ID - 1];
	}

	else if (x == 0 && y == 0) {
		boundArray[ID] = 0.5f *(boundArray[ID + 1] + boundArray[ID + RES + 2]);
	}
	else if (x == 0 && y == RES + 1) {
		boundArray[ID] = 0.5f *(boundArray[ID + 1] + boundArray[ID - RES - 2]);
	}
	else if (x == RES + 1 && y == 0) {
		boundArray[ID] = 0.5f *(boundArray[ID - 1] + boundArray[ID + RES + 2]);
	}
	else if (x == RES + 1 && y == RES + 1) {
		boundArray[ID] = 0.5f *(boundArray[ID - 1] + boundArray[ID - RES - 2]);
	}
}

__device__ void cSwap(float** arrayOne, float** arrayTwo) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if (ID == 0) {
		float* temp = *arrayOne;
		*arrayOne = *arrayTwo;
		*arrayTwo = temp;
	}
}

__device__ int cGetX(int arrayPos) {
	return (arrayPos % (RES + 2));
}

__device__ int cGetY(int arrayPos) {
	return (arrayPos / (RES + 2));
}

__device__ int cGetArrayPos(int xPos, int yPos) {
	return xPos + (RES + 2)*yPos;
}
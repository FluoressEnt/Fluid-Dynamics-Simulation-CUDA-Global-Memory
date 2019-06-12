#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "Solver.h"

///function to swap the pointers of two arrays
void swap(float **a, float **b) {
	float *t = *a;
	*a = *b;
	*b = t;
}

///Creates kernels that run in parallel on the GPU
void Solver::CalculateWrapper() {
	//diffusion part

	// Add source 
	{
		cAddSource << <NBLOCK, TPBLOCK >> > (cSDens, cNewDens);
		cudaDeviceSynchronize();
	}

	swap(&cNewDens, &cOldDens);

	// Diffusion
	{
		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcDiffusion << <NBLOCK, TPBLOCK >> > (cNewDens, cOldDens);
			cudaDeviceSynchronize();
			cCalcBound << <NBLOCK, TPBLOCK >> > (1, cNewDens);
			cudaDeviceSynchronize();
		}
	}

	swap(&cNewDens, &cOldDens);

	// Advection
	{
		cCalcAdvection << <NBLOCK, TPBLOCK >> > (cNewDens, cOldDens, cNewVelX, cNewVelY);
		cudaDeviceSynchronize();
		cCalcBound << <NBLOCK, TPBLOCK >> > (2, cNewDens);
		cudaDeviceSynchronize();
	}

	//velocity part

	// Add Source 
	{
		cAddSource << <NBLOCK, TPBLOCK >> > (cSVelX, cNewVelX);
		cAddSource << <NBLOCK, TPBLOCK >> > (cSVelY, cNewVelY);
		cudaDeviceSynchronize();
	}

	swap(&cNewVelX, &cOldVelX);
	swap(&cNewVelY, &cOldVelY);

	// Diffusion
	{
		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcDiffusion << <NBLOCK, TPBLOCK >> > (cNewVelX, cOldVelX);
			cCalcDiffusion << <NBLOCK, TPBLOCK >> > (cNewVelY, cOldVelY);
			cudaDeviceSynchronize();
			cCalcBound << <NBLOCK, TPBLOCK >> > (1, cNewVelX);
			cCalcBound << <NBLOCK, TPBLOCK >> > (2, cNewVelY);
			cudaDeviceSynchronize();
		}
	}

	// Projection 
	{
		cCalcProjY << <NBLOCK, TPBLOCK >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <NBLOCK, TPBLOCK >> > (0, cOldVelY);
		cCalcBound << <NBLOCK, TPBLOCK >> > (0, cOldVelX);
		cudaDeviceSynchronize();

		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcProjX << <NBLOCK, TPBLOCK >> > (cOldVelX, cOldVelY);
			cudaDeviceSynchronize();
			cCalcBound << <NBLOCK, TPBLOCK >> > (0, cOldVelX);
			cudaDeviceSynchronize();
		}

		cCalcFinalProj << <NBLOCK, TPBLOCK >> > (cNewVelX, cNewVelY, cOldVelX);
		cudaDeviceSynchronize();
		cCalcBound << <NBLOCK, TPBLOCK >> > (1, cNewVelX);
		cCalcBound << <NBLOCK, TPBLOCK >> > (2, cNewVelY);
		cudaDeviceSynchronize();
	}

	swap(&cNewVelX, &cOldVelX);
	swap(&cNewVelY, &cOldVelY);

	// Advection 
	{
		cCalcAdvection << <NBLOCK, TPBLOCK >> > (cNewVelX, cOldVelX, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcAdvection << <NBLOCK, TPBLOCK >> > (cNewVelY, cOldVelY, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <NBLOCK, TPBLOCK >> > (1, cNewVelX);
		cCalcBound << <NBLOCK, TPBLOCK >> > (2, cNewVelY);
		cudaDeviceSynchronize();
	}

	// Projection
	{
		cCalcProjY << <NBLOCK, TPBLOCK >> > (cNewVelX, cNewVelY, cOldVelX, cOldVelY);
		cudaDeviceSynchronize();
		cCalcBound << <NBLOCK, TPBLOCK >> > (0, cOldVelY);
		cCalcBound << <NBLOCK, TPBLOCK >> > (0, cOldVelX);
		cudaDeviceSynchronize();

		for (int GaussItterator = 0; GaussItterator < 20; GaussItterator++) {
			cCalcProjX << <NBLOCK, TPBLOCK >> > (cOldVelX, cOldVelY);
			cudaDeviceSynchronize();
			cCalcBound << <NBLOCK, TPBLOCK >> > (0, cOldVelX);
			cudaDeviceSynchronize();
		}

		cCalcFinalProj << <NBLOCK, TPBLOCK >> > (cNewVelX, cNewVelY, cOldVelX);
		cudaDeviceSynchronize();
		cCalcBound << <NBLOCK, TPBLOCK >> > (1, cNewVelX);
		cCalcBound << <NBLOCK, TPBLOCK >> > (2, cNewVelY);
		cudaDeviceSynchronize();
	}
}

///kernel that adds a source to array values
__global__ void cAddSource(float *inputArr, float *arrayAffected) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	if (ID < ALENGTH) {
		float temp;
		temp = (DT * inputArr[ID]);
		arrayAffected[ID] += temp;
	}
}

///kernel that calculates the diffusion of the fluid on th GPU
__global__ void cCalcDiffusion(float* newArray, float* oldArray) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		//conservation of diffusivity
		float a = DT * VISC * RES * RES;
		newArray[ID] = (oldArray[ID] + a * (newArray[cGetArrayPos(xPos - 1, yPos)] + newArray[cGetArrayPos(xPos + 1, yPos)]
			+ newArray[cGetArrayPos(xPos, yPos - 1)] + newArray[cGetArrayPos(xPos, yPos + 1)])) / (1 + 4 * a);
	}
}

///kernel that calculates the advection of the fluid on the GPU
__global__ void cCalcAdvection(float* cNew, float* cOld, float* cVelX, float* cVelY) {
	int left, bottom, right, top, ID;
	float x, y, distToRight, distToTop, distToLeft, distToBottom;

	ID = blockIdx.x * blockDim.x + threadIdx.x;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {

		x = xPos - DT * RES * cVelX[ID];
		y = yPos - DT * RES * cVelY[ID];

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

		cNew[ID] = distToRight * (distToTop*cOld[cGetArrayPos(left, bottom)] + distToBottom * cOld[cGetArrayPos(left, top)])
			+ distToLeft * (distToTop*cOld[cGetArrayPos(right, bottom)] + distToBottom * cOld[cGetArrayPos(right, top)]);
	}
}

///kernel that calculates a part of the projection of the fluid on the GPU
__global__ void cCalcFinalProj(float* cNewVelX, float* cNewVelY, float* cOldVelX) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	float h = 1.0f / RES;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cNewVelX[ID] -= 0.5f*(cOldVelX[ID + 1] - cOldVelX[ID - 1]) / h;
		cNewVelY[ID] -= 0.5f*(cOldVelX[ID + RES + 2] - cOldVelX[ID - RES - 2]) / h;
	}
}

///kernel that calculates a part of the projection of the fluid on the GPU
__global__ void cCalcProjY(float* cNewVelX, float* cNewVelY, float* cOldVelX, float* cOldVelY) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	float h = 1.0f / RES;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cOldVelY[ID] = -0.5f * h * (cNewVelX[ID + 1] - cNewVelX[ID - 1]
			+ cNewVelY[ID + RES + 2] - cNewVelY[ID - RES - 2]);
		cOldVelX[ID] = 0;
	}
}

///kernel that calculates a part of the projection of the fluid on the GPU
__global__ void cCalcProjX(float* cOldVelX, float* cOldVelY) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;

	int xPos = cGetX(ID);
	int yPos = cGetY(ID);

	if ((xPos > 0) && (xPos <= RES) && (yPos > 0) && (yPos <= RES)) {
		cOldVelX[ID] = (cOldVelY[ID] + cOldVelX[ID - 1]
			+ cOldVelX[ID + 1] + cOldVelX[ID - RES - 2] + cOldVelX[ID + RES + 2]) / 4;
	}
}

///kernel that sets the boundary of the fluid to one where fluid does not leave the grid
__global__ void cCalcBound(int b, float* boundArray) {
	int ID = blockIdx.x * blockDim.x + threadIdx.x;
	int x, y;
	x = ID % (RES + 2);
	y = ID / (RES + 2);

	//setting sides
	if (x > 0 && x <= RES && y == 0) {
		boundArray[ID] = b == 2 ? -boundArray[ID + RES + 2] : boundArray[ID + RES + 2];
	}
	else if (x > 0 && x <= RES && y == RES + 1) {
		boundArray[ID] = b == 2 ? -boundArray[ID - RES - 2] : boundArray[ID - RES - 2];
	}
	else if (x == 0 && y > 0 && y <= RES) {
		boundArray[ID] = b == 1 ? -boundArray[ID + 1] : boundArray[ID + 1];
	}
	else if (x == RES + 1 && y > 0 && y <= RES) {
		boundArray[ID] = b == 1 ? -boundArray[ID - 1] : boundArray[ID - 1];
	}

	//setting corners
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

///A function that runs on the GPU that calculates the 2D X coordinate given the 1D array position
__device__ int cGetX(int arrayPos) {
	return (arrayPos % (RES + 2));
}
///A function that runs on the GPU that calculates the 2D Y coordinate given the 1D array position
__device__ int cGetY(int arrayPos) {
	return (arrayPos / (RES + 2));
}
///A function that runs on the GPU that calculates the 1D array position given the 2D X&Y coordinates
__device__ int cGetArrayPos(int xPos, int yPos) {
	return xPos + (RES + 2)*yPos;
}
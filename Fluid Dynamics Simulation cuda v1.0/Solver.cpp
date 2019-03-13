#include "Solver.h"
#include <iostream>


///Initialise Arrays on CPU with value 0
Solver::Solver() {

	//create arrays
	sDens = new float[ALENGTH];
	sVelX = new float[ALENGTH];
	sVelY = new float[ALENGTH];
	newDens = new float[ALENGTH];
	oldDens = new float[ALENGTH];
	newVelX = new float[ALENGTH];
	newVelY = new float[ALENGTH];
	oldVelX = new float[ALENGTH];
	oldVelY = new float[ALENGTH];

	//set arrays to 0
	for (auto i = 0u; i < ALENGTH; i++) {
		sDens[i] = 0;
		sVelX[i] = 0;
		sVelY[i] = 0;
		newDens[i] = 0;
		oldDens[i] = 0;
		newVelX[i] = 0;
		newVelY[i] = 0;
		oldVelX[i] = 0;
		oldVelY[i] = 0;
	}

	//allocate some memory in in shared memory and return errors if it fails
	CUDA_CHECK(cudaMalloc((void**)&cNewDens, ALENGTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&cOldDens, ALENGTH * sizeof(float)));

	CUDA_CHECK(cudaMalloc((void**)&cNewVelX, ALENGTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&cNewVelY, ALENGTH * sizeof(float)));

	CUDA_CHECK(cudaMalloc((void**)&cOldVelX, ALENGTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&cOldVelY, ALENGTH * sizeof(float)));

	CUDA_CHECK(cudaMalloc((void**)&cSDens, ALENGTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&cSVelX, ALENGTH * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&cSVelY, ALENGTH * sizeof(float)));

	//define what and how much you want to copy, and the direction of the copying - CPU to GPU
	CUDA_CHECK(cudaMemcpy(cNewDens, newDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cOldDens, oldDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(cNewVelX, newVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cNewVelY, newVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(cOldVelX, oldVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cOldVelY, oldVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(cSDens, sDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cSVelX, sVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cSVelY, sVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
}

///Release memory on the GPU on destruction of the Solver object
Solver::~Solver(void)
{
	CUDA_CHECK(cudaFree(cNewDens));
	CUDA_CHECK(cudaFree(cOldDens));
	CUDA_CHECK(cudaFree(cNewVelX));
	CUDA_CHECK(cudaFree(cNewVelY));
	CUDA_CHECK(cudaFree(cOldVelX));
	CUDA_CHECK(cudaFree(cOldVelY));
	CUDA_CHECK(cudaFree(cSDens));
	CUDA_CHECK(cudaFree(cSVelX));
	CUDA_CHECK(cudaFree(cSVelY));
}

///Retreive density array from the GPU
float* Solver::GetDensityArray() {
	CUDA_CHECK(cudaMemcpy(newDens, cNewDens, ALENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	return newDens;
}
///Revreive velocityX array from the GPU
float* Solver::GetVelXArray() {
	CUDA_CHECK(cudaMemcpy(newVelX, cNewVelX, ALENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	return newVelX;
}
///Retreive velocityY array from the GPU
float* Solver::GetVelYArray() {
	CUDA_CHECK(cudaMemcpy(newVelY, cNewVelY, ALENGTH * sizeof(float), cudaMemcpyDeviceToHost));
	return newVelY;
}

///Set inputDensity array value to 1.0f then copy to the GPU
void Solver::SetInputDens(int arrayValue) {
	sDens[arrayValue] = 1.0f;
	CUDA_CHECK(cudaMemcpy(cSDens, sDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
}
///Set inputVelocity array values to input value and then copy to the GPU
void Solver::SetInputVel(int arrayValue, int xVel, int yVel) {
	sVelX[arrayValue] = xVel;
	sVelY[arrayValue] = yVel;
	CUDA_CHECK(cudaMemcpy(cSVelX, sVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cSVelY, sVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
}

///Set every inputDensity array element to 0 and copy to the GPU
void Solver::RefreshDensIn() {
	for (auto i = 0u; i < ALENGTH; i++) {
		sDens[i] = 0;
	}
	CUDA_CHECK(cudaMemcpy(cSDens, sDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
}
///Set every inputVelocityX&Y array element to 0 and copy to the GPU
void Solver::RefreshVelIn() {
	for (auto i = 0u; i < ALENGTH; i++) {
		sVelX[i] = 0;
		sVelY[i] = 0;
	}
	CUDA_CHECK(cudaMemcpy(cSVelX, sVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cSVelY, sVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
}
///Set every array element to 0 and copy to the GPU
void Solver::RefreshAll() {
	//set arrays to 0
	for (auto i = 0u; i < ALENGTH; i++) {
		sDens[i] = 0;
		sVelX[i] = 0;
		sVelY[i] = 0;
		newDens[i] = 0;
		oldDens[i] = 0;
		newVelX[i] = 0;
		newVelY[i] = 0;
		oldVelX[i] = 0;
		oldVelY[i] = 0;
	}

	//copy values over to GPU
	CUDA_CHECK(cudaMemcpy(cNewDens, newDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cOldDens, oldDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(cNewVelX, newVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cNewVelY, newVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(cOldVelX, oldVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cOldVelY, oldVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaMemcpy(cSDens, sDens, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cSVelX, sVelX, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(cSVelY, sVelY, ALENGTH * sizeof(float), cudaMemcpyHostToDevice));
}
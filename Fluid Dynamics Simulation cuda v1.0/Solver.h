#include "defines.h"


class Solver {
public:
	Solver();
	~Solver(void);

	void CalculateWrapper();

	float* GetDensityArray();
	float* GetVelXArray();
	float* GetVelYArray();

	void RefreshDensIn();
	void RefreshVelIn();
	void RefreshAll();

	void SetInputDens(int);
	void SetInputVel(int, int, int);

private:

	//input arrays
	float *sDens;
	float* sVelX;
	float* sVelY;
	//CPU arrays
	float* newDens;
	float* oldDens;
	float* newVelX;
	float* newVelY;
	float* oldVelX;
	float* oldVelY;

	//pointers to arrays on gpu
	float* cSDens;
	float* cSVelX;
	float* cSVelY;
	float* cNewDens;
	float* cOldDens;
	float* cNewVelX;
	float* cNewVelY;
	float* cOldVelX;
	float* cOldVelY;
};

#ifdef __cplusplus
extern "C" {
#endif
	//kernels
	__global__ void cAddSource(float*, float*);
	__global__ void cCalcDiffusion(float*, float*);
	__global__ void cCalcAdvection(float*, float*, float*, float*);
	__global__ void cCalcProjY(float*, float*, float*, float*);
	__global__ void cCalcProjX(float*, float*);
	__global__ void cCalcFinalProj(float*, float*, float*);
	__global__ void cCalcBound(int, float*);

	void swap(float**, float**);

	__device__ int cGetX(int);
	__device__ int cGetY(int);
	__device__ int cGetArrayPos(int, int);

#ifdef __cplusplus
}
#endif
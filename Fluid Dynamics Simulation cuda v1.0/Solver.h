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

	__global__ void cCalculate(float*, float*, float*, float*, float*, float*, float*, float*, float*);

	__device__ void cFinalProjection(float*, float*, float*);
	__device__ void cProjectionInX(float*, float*);
	__device__ void cProjectionInY(float*, float*, float*, float*);
	__device__ void cAdvection(float *, float *, float *, float *);
	__device__ void cDiffuse(int, float *, float *);
	__device__ void cAddSource(float *, float *);

	__device__ void cSetBoundary(int, float*);
	__device__ void cSwap(float**, float**);

	__device__ int cGetX(int);
	__device__ int cGetY(int);
	__device__ int cGetArrayPos(int, int);

#ifdef __cplusplus
}
#endif
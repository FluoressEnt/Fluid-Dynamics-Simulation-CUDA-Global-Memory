#include "Colour3.h"
#include "Solver.h"

#pragma once

typedef struct _mSolver {
	Solver solver = Solver();
	bool mouseButtonDown = false;
}mSolver;

class Visualisation {
public:
	static mSolver fSolver;
	static void OnKeyDown(unsigned char, int, int);
	static void OnMouseClick(int, int, int, int);
	static void OnMouseDrag(int, int);
	static void Render();
	static void Calculate();

private:
	static void setMouseButtonState(bool);
	static bool const isMouseButtonDown();
	static Colour3 DetermineColour(float);
};
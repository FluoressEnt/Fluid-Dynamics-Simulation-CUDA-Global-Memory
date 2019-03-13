#include "Visualisation.h"
#include "ConversionTools.h"
#include "Defines.h"
#include <iostream>
#include <gl\freeglut.h>


using namespace std;

mSolver Visualisation::fSolver;
int mouseX;
int mouseY;
int oldMouseX;
int oldMouseY;
bool mouseButtonDown = false;
bool diffuseDisplay = true;

///A function that catches specific key presses
void Visualisation::OnKeyDown(unsigned char key, int x, int y) {
	switch (key) {
	case 32:									//spacebar - swap visualisation to render
		diffuseDisplay = !diffuseDisplay;

	case 82:									//R - refreshes all arrays in simulation
		fSolver.solver.RefreshAll();

	case 114:									//r - refreshes all arrays in simulation
		fSolver.solver.RefreshAll();
	}

}
///A function that updates the X&Y position of the mouse or refreshes the input arrays on the GPU
void Visualisation::OnMouseClick(int button, int state, int xPos, int yPos) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		Visualisation::setMouseButtonState(true);
		mouseX = xPos;
		mouseY = yPos;

		oldMouseX = mouseX;
		oldMouseY = mouseY;
	}
	if (button == GLUT_LEFT_BUTTON && state == GLUT_UP) {
		Visualisation::setMouseButtonState(false);
		if (diffuseDisplay)
			fSolver.solver.RefreshDensIn();
		else {
			fSolver.solver.RefreshVelIn();
		}
	}
}
///A function that refreshes the input arrays to 0, updates the X&Y position of the mouse
///Then assigns that frames values for the inputof either density or velocity depenant on the current visualisation selected
void Visualisation::OnMouseDrag(int xPos, int yPos) {
	if (Visualisation::isMouseButtonDown() && xPos > 0 && xPos < RES) {

		if (diffuseDisplay)
			fSolver.solver.RefreshDensIn();
		else {
			fSolver.solver.RefreshVelIn();
		}

		mouseX = xPos;
		mouseY = yPos;

		int arrayValue = ConversionTools::ConvertCoordToArray(xPos, yPos);

		if (arrayValue < ALENGTH && arrayValue > 0) {
			if (diffuseDisplay)
				fSolver.solver.SetInputDens(arrayValue);
			else {
				fSolver.solver.SetInputVel(arrayValue, mouseX - oldMouseX, oldMouseY - mouseY);
			}
		}

		oldMouseX = mouseX;
		oldMouseY = mouseY;
	}
}

void Visualisation::setMouseButtonState(bool value) {
	mouseButtonDown = value;
}
bool const Visualisation::isMouseButtonDown() {
	return mouseButtonDown;
}

///A function that determines the colour of a value using a clamped scale
Colour3 Visualisation::DetermineColour(float value)
{
	//ensure not 0 or will return error
	if (value == 0) {
		return Colour3(0.0f, 0.0f, 0.0f);
	}

	//ensure there are no negatives due to logarithm
	if (value < 0)
	{
		value *= -1;

		float logValue = log10(value);

		if (logValue < -40) {					//pruple
			return Colour3(0.3f, 0.0f, 0.3f);
		}
		else if (logValue < -30) {				//purple-blue
			return Colour3(0.5f, 0.0f, 0.8f);
		}
		else if (logValue < -20) {				//blue
			return Colour3(0.0f, 0.0f, 1.0f);
		}
		else if (logValue < -10) {				//blue-green
			return Colour3(0.0f, 0.5f, 0.5f);
		}
		else if (logValue < 0) {				//green
			return Colour3(0.0f, 1.0f, 0.0f);
		}
	}
	else {

		float logValue = log10(value);

		if (logValue < -40) {					//yellow
			return Colour3(1.0f, 1.0f, 0.0f);
		}
		else if (logValue < -30) {				//yellow-orange
			return Colour3(1.0f, 0.8f, 0.0f);
		}
		else if (logValue < -20) {				//orange
			return Colour3(1.0f, 0.5f, 0.0f);
		}
		else if (logValue < -10) {				//orange-red
			return Colour3(1.0f, 0.2f, 0.0f);
		}
		else if (logValue < 0) {				//red
			return Colour3(1.0f, 0.0f, 0.0f);
		}

	}
	return Colour3(0.0f, 0.0f, 0.0f);
}

///A function that renders either the representation of diffusion or velocity field 
///Calculated from on the most recent values from the GPU
void Visualisation::Render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_ACCUM_BUFFER_BIT);

	if (diffuseDisplay) {
		glBegin(GL_POINTS);
		//retreive the solver's most recent density array from the GPU
		float* calculatedDensity = fSolver.solver.GetDensityArray();

		for (int i = 0; i < ALENGTH; i++) {

			//determine point's colour based on a clamped scale
			Colour3 colourValue = DetermineColour(calculatedDensity[i]);

			tuple<int, int> coords = ConversionTools::ConvertArraytoCoord(i);
			float x = ConversionTools::ConvertWindowToGL(get<0>(coords), false);
			float y = ConversionTools::ConvertWindowToGL(get<1>(coords), true);

			glColor3f(colourValue.getX(), colourValue.getY(), colourValue.getZ());
			glVertex3f(x, y, 0.0f);

		}
		glEnd();
		glutSwapBuffers();
	}
	else {
		glBegin(GL_LINES);
		glLineWidth(1.0f);
		glColor3f(1.0f, 1.0f, 1.0f);

		//retreive solver's most recent velocity arrays from the GPU
		float* calculatedVelocityX = fSolver.solver.GetVelXArray();
		float* calculatedVelocityY = fSolver.solver.GetVelYArray();

		for (int i = 5; i <= RES; i = i + 10) {
			for (int j = 5; j <= RES; j = j + 10) {

				int arrayPos = ConversionTools::ConvertCoordToArray(i, j);
				float velX = calculatedVelocityX[arrayPos];
				float velY = calculatedVelocityY[arrayPos];

				float magnitude = sqrt(velX*velX + velY * velY);

				if (magnitude != 0) {
					//find unit components
					float unitX = velX / magnitude;
					float unitY = velY / magnitude;

					//scale components with log and scalar to standardise the size & make visible
					float scale = log(magnitude*1e20 + 1.0);
					float newX = unitX * scale;
					float newY = unitY * scale;

					//creating coordinates in window space where line is drawn around the origin i,j
					tuple<float, float> startWindowPos = ConversionTools::ConvertCoordtoWindow(i - newX / 2, j - newY / 2);
					tuple<float, float> endWindowPos = ConversionTools::ConvertCoordtoWindow(i + newX / 2, j + newY / 2);

					//start of line
					glVertex2f(get<0>(startWindowPos), get<1>(startWindowPos));
					//end of line
					glVertex2f(get<0>(endWindowPos), get<1>(endWindowPos));
				}
			}
		}
		glEnd();
		glutSwapBuffers();
	}
}

///A function that calls the solver's calculate which calls the GPU kernel
///Performs a GL redisplay after the algorithm has completed 1 itteration
void Visualisation::Calculate() {
	fSolver.solver.CalculateWrapper();
	glutPostRedisplay();
}
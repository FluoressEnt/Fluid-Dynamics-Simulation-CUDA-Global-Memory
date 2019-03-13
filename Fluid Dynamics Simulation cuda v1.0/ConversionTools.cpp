#include "ConversionTools.h"
#include "Defines.h"
#include <gl/freeglut.h>

///Converts a 2D coordinate to a 1D array position
int ConversionTools::ConvertCoordToArray(int xPos, int yPos) {
	int arrayValue = ((xPos)+(RES + 2)*yPos);
	return arrayValue;
}

///Converts a 1D array position to a 2D coordinate
std::tuple<int, int> ConversionTools::ConvertArraytoCoord(int arrayValue) {
	int xPos, yPos;
	xPos = arrayValue % (RES + 2);
	yPos = arrayValue / (RES + 2);

	std::tuple<int, int> coordinates(xPos, yPos);
	return coordinates;
}

///Converts a 2D array position to Freeglut Window position
std::tuple<float, float> ConversionTools::ConvertCoordtoWindow(int xPos, int yPos) {
	float newX = (float) xPos;
	float newY = (float) yPos;

	if (xPos < (RES / 2)) {
		newX = (1 - (newX / (RES / 2)))*-1;
	}
	else {
		newX = (newX - (RES / 2)) / (RES / 2);
	}

	if (yPos < (RES / 2)) {
		newY = 1 - (newY / (RES / 2));
	}
	else {
		newY = ((newY - (RES / 2)) / (RES / 2))*-1;
	}

	std::tuple<float, float> WindowSpace(newX, newY);
	return	WindowSpace;
}

///Converts a Freeglut Window position to a Gl position
float ConversionTools::ConvertWindowToGL(int number, bool isHeight) {
	float windowDimension;
	float newNumber = (float)number;
	if (isHeight) {
		windowDimension = (float)(glutGet(GLUT_WINDOW_HEIGHT) / 2);
		newNumber -= windowDimension;
		newNumber = newNumber / windowDimension;
		newNumber *= -1;
	}
	else {
		windowDimension = (float)(glutGet(GLUT_WINDOW_WIDTH) / 2);
		newNumber -= windowDimension;
		newNumber = newNumber / windowDimension;
	}
	return newNumber;
}
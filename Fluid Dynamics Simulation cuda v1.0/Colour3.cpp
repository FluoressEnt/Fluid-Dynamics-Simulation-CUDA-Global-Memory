#include "Colour3.h"

Colour3::Colour3():X(1.0), Y(1.0), Z(1.0)
{
}

Colour3::Colour3(float x, float y, float z) :X(x), Y(y), Z(z)
{
}

float Colour3::getX() {
	return X;
}

float Colour3::getY() {
	return Y;
}

float Colour3::getZ() {
	return Z;
}
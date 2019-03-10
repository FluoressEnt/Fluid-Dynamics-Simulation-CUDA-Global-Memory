#pragma once

class Colour3 {
public:
	Colour3();
	Colour3(float, float, float);
	float getX();
	float getY();
	float getZ();

private:
	//deleting assignment operator
	Colour3& operator=(const Colour3&);

	float X;
	float Y;
	float Z;
};
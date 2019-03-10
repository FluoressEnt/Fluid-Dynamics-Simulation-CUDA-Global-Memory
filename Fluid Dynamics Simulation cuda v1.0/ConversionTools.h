#include <tuple>
#pragma once

class ConversionTools {
public:
	static int ConvertCoordToArray(int, int);
	static std::tuple<int, int> ConvertArraytoCoord(int);
	static std::tuple<float, float> ConvertCoordtoWindow(int, int);
	static float ConvertWindowToGL(int, bool);
};
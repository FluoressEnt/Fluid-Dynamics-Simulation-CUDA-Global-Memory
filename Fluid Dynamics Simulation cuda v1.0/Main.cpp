#include "Visualisation.h"
#include "Defines.h"
#include <gl/freeglut.h>


int main(int argc, char* argv[]) {

	//setup window
	glutInit(&argc, argv);
	glutInitWindowSize(RES, RES);
	glutInitWindowPosition(800, 100);
	glutCreateWindow("Fluid Dynamics Simulation");

	//register callback functions
	glutMouseFunc(Visualisation::OnMouseClick);
	glutMotionFunc(Visualisation::OnMouseDrag);
	glutKeyboardFunc(Visualisation::OnKeyDown);
	glutDisplayFunc(Visualisation::Render);
	glutIdleFunc(Visualisation::Calculate);

	//initialise GL setup
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glPointSize(2.0f);

	// Enter the event-processing loop
	glutMainLoop();

	return 0;
}

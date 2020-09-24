#include "GUI/WorklistWindow.h"
#include "GUI/CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "ASAP/ASAP_Window.h"

using namespace ASAP;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	// Creates the composite window, that'll hold all other required tabs
	CompositeWindow main_window;
	main_window.show();

	// Creates the worklist and viewer tabs
	WorklistWindow worklist;
	ASAP_Window workstation;

	// Adds the tabs to the composite window and then connects the viewer to the worklist
	main_window.addTab(&worklist, "Worklist");
	int viewer_tab = main_window.addTab(&workstation, "Viewer");
	worklist.attachWorkstation(workstation, viewer_tab);

	return a.exec();
}
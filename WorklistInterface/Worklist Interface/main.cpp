#include "WorklistWindow.h"
#include "CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "../../ASAP/pathologyworkstation.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	ASAP::Worklist::GUI::CompositeWindow main_window;
	main_window.show();

	ASAP::Worklist::GUI::WorklistWindow worklist;	
	PathologyWorkstation workstation;
	

	main_window.AddTab(&worklist, "Worklist");
	int viewer_tab = main_window.AddTab(&workstation, "Viewer");

	worklist.AttachWorkstation(workstation, viewer_tab);

	return a.exec();
}
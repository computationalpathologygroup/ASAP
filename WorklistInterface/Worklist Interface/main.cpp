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
	worklist.AttachWorkstation(workstation);

	main_window.AddTab(&worklist, "Worklist");
	main_window.AddTab(&workstation, "Viewer");

	return a.exec();
}
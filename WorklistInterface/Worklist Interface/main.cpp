#include "WorklistWindow.h"
#include "CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "../../ASAP/pathologyworkstation.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	//uri_info.base_url = L"http://127.0.0.1:8000/api";

	ASAP::Worklist::GUI::CompositeWindow main_window;
	ASAP::Worklist::GUI::WorklistWindow worklist;	
	PathologyWorkstation workstation;

	worklist.AttachWorkstation(workstation);

	main_window.AddTab(&worklist, "Worklist");
	main_window.AddTab(&workstation, "Viewer");

	main_window.show();

	return a.exec();
}
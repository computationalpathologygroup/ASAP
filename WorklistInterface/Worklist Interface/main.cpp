#include "WorklistWindow.h"
#include "CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "../../ASAP/pathologyworkstation.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	// Creates the composite window, that'll hold all other required tabs
	ASAP::Worklist::GUI::CompositeWindow main_window;
	main_window.show();

	// Creates the worklist and viewer tabs
	ASAP::Worklist::GUI::WorklistWindow worklist;	
	PathologyWorkstation workstation;

	// Creates the worklist shortcuts.
	std::vector<ASAP::Worklist::GUI::ShortcutAction> worklist_shortcuts;
	worklist_shortcuts.push_back(
	{
		std::function<void(void)>(std::bind(&ASAP::Worklist::GUI::WorklistWindow::MoveSelectionLeft, &worklist)),
		QKeySequence(Qt::CTRL + Qt::Key_Left)
	});

	// Adds the tabs to the composite window and then connects the viewer to the worklist
	main_window.AddTab(&worklist, "Worklist", worklist_shortcuts);
	int viewer_tab = main_window.AddTab(&workstation, "Viewer");
	worklist.AttachWorkstation(workstation, viewer_tab);

	

	return a.exec();
}
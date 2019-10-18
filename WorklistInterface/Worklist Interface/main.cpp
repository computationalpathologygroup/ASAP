#include "GUI/WorklistWindow.h"
#include "GUI/CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "ASAP/ASAP_Window.h"

using namespace ASAP;

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	// Creates the composite window, that'll hold all other required tabs
	GUI::CompositeWindow main_window;
	main_window.show();

	// Creates the worklist and viewer tabs
	GUI::WorklistWindow worklist;
	ASAP_Window workstation;

	// Creates the worklist shortcuts.
	std::vector<GUI::ShortcutAction> worklist_shortcuts;
	worklist_shortcuts.push_back(
	{
		std::function<void(void)>(std::bind(&GUI::WorklistWindow::MoveImageSelectionLeft, &worklist)),
		QKeySequence(Qt::CTRL + Qt::Key_Left)
	});

	worklist_shortcuts.push_back(
	{
		std::function<void(void)>(std::bind(&GUI::WorklistWindow::MoveImageSelectionRight, &worklist)),
		QKeySequence(Qt::CTRL + Qt::Key_Right)
	});

	// Adds the tabs to the composite window and then connects the viewer to the worklist
	main_window.AddTab(&worklist, "Worklist", worklist_shortcuts);
	int viewer_tab = main_window.AddTab(&workstation, "Viewer");
	worklist.AttachWorkstation(workstation, viewer_tab);

	return a.exec();
}
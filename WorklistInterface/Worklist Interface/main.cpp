#include "ASAP_GUI_Window.h"
#include <QtWidgets/QApplication>

#include "worklist_csv_example.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	Worklist_CSV_Example* example(new Worklist_CSV_Example(
	{
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/worklist_items.csv",
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/patient_items.csv",
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/study_items.csv",
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/image_items.csv"
	}));

	ASAP_GUI_Window w(example);
	w.show();
	return a.exec();
}
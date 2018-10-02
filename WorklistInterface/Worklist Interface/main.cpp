#include "WorklistWindow.h"
#include "CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "../../ASAP/pathologyworkstation.h"

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	/*Worklist_CSV_Example* example(new Worklist_CSV_Example(
	{
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/worklist_items.csv",
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/patient_items.csv",
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/study_items.csv",
		"C:/Users/Karel Gerbrands/Documents/Visual Studio 2017/Projects/Worklist Interface/Worklist Interface/image_items.csv"
	}));*/

	ASAP::Worklist::Data::DjangoRestURI uri_info;
	uri_info.base_url = L"http://127.0.0.1:8000/api";
	uri_info.worklist_addition = L"worklists";
	uri_info.patient_addition = L"patients";
	uri_info.study_addition = L"studies";
	uri_info.image_addition = L"images";
	uri_info.worklist_patient_relation_addition = L"worklist_patient_relations";

	ASAP::Worklist::Data::DjangoDataAcquisition example(uri_info);

	ASAP::Worklist::GUI::CompositeWindow main_window;
	ASAP::Worklist::GUI::WorklistWindow worklist(&example);	
	PathologyWorkstation workstation;

	worklist.workstation_window = &workstation;

	main_window.AddTab(&worklist, "Worklist");
	main_window.AddTab(&workstation, "Viewer");

	main_window.show();

	return a.exec();
}
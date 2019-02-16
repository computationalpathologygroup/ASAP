#include "GUI/WorklistWindow.h"
#include "GUI/CompositeWindow.h"
#include <QtWidgets/QApplication>

#include "ASAP/ASAP_Window.h"

#include <cpprest/http_client.h>

#include "Networking/Django_Connection.h"
#include "Serialization/JSON.h"

#include <algorithm>
#include <codecvt>
#include <iostream>
#include <string> 

using namespace ASAP;



#include "Networking/HTTP_File_Download.h"
#include "Misc/TemporaryDirectoryTracker.h"


int main(int argc, char *argv[])
{
	web::http::client::http_client_config config;
	config.set_validate_certificates(false);
	

	// image: e27b6442-0e5d-4bd9-80f7-f6d26b4e1993
	std::wstring body;
	Networking::Django_Connection connection(L"https://localhost/", Networking::Django_Connection::AUTHENTICATION_TYPE::SESSION, Networking::Django_Connection::CreateCredentials(L"admin", L"admin", L"accounts/signin/", L"api/auth/"), config);
	if (connection.GetAuthenticationStatus() == Networking::Django_Connection::AUTHENTICATED)
	{
		web::http::http_request request(web::http::methods::GET);
	//	request.set_request_uri(L"api/v1/cases/images/0f754cd0-8f17-4b9a-ad0a-437c67b35d29");
		request.set_request_uri(L"media/images/0f754cd0-8f17-4b9a-ad0a-437c67b35d29/Patient4.tif");

		Networking::FileDownloadResults results;
		web::http::http_response reppy;
		connection.SendRequest(request).then([&results, &reppy](const web::http::http_response& rep)
		{
			reppy = rep;
		}).wait();

	/*	std::wstring json_result;
		reppy.extract_json().then([&json_result](web::json::value val)
		{
			json_result = val.to_string();
		}).wait();


		connection.SendRequest(request).then([&results, &reppy](const web::http::http_response& rep)
		{
			//results = Networking::HTTP_File_Download(rep, boost::filesystem::path("D:/"));
			reppy = rep;
		}).wait();
		*/
		
		
		results = Networking::HTTP_File_Download(reppy, boost::filesystem::path("D:/"));
	}
	
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
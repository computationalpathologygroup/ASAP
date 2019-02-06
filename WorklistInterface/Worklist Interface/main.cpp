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

#include <cpprest/filestream.h>

using namespace ASAP::Worklist;

enum DOWNLOAD_STATUS { DOWNLOAD_FAILURE, FILE_CREATION_FAILURE, NO_ATTACHEMENT, SUCCESS };

std::string WideStringToString(const std::wstring string)
{
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	std::string new_string(converter.to_bytes(string));
	return new_string;
}

std::wstring StringToWideString(const std::string string)
{
	std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
	std::wstring new_string(converter.from_bytes(string));
	return new_string;
}

DOWNLOAD_STATUS HTTP_File_Download(const web::http::http_response& response, const std::string directory)
{
	web::http::http_headers headers(response.headers());
	auto content_disposition = headers.find(L"Content-Disposition");

	if (response.status_code == web::http::status_codes::OK && content_disposition != headers.end())
	{
		std::wstring lower_case_disposition(content_disposition->second);
		std::transform(lower_case_disposition.begin(), lower_case_disposition.end(), lower_case_disposition.begin(), ::tolower);

		if (lower_case_disposition.find(L"filename=") != std::string::npos)
		{
			size_t filename_start = lower_case_disposition.find_last_of('"');
			std::wstring filepath(StringToWideString(directory) + L"/" + lower_case_disposition.substr(filename_start, lower_case_disposition.size() - filename_start));
			std::shared_ptr<concurrency::streams::ostream> filestream(std::make_shared<concurrency::streams::ostream>());
			concurrency::streams::fstream::open_ostream(filepath);
			
			if (filestream->is_open())
			{
				try
				{
					response.body().read_to_end(filestream->streambuf()).wait();
					filestream->close();
					return DOWNLOAD_STATUS::SUCCESS;
				}
				// TODO: Replace with specific error catching
				catch (const std::exception& e)
				{
					filestream->close();
					// TODO: Remove failed file.
					return DOWNLOAD_STATUS::DOWNLOAD_FAILURE;
				}
			}
			return DOWNLOAD_STATUS::FILE_CREATION_FAILURE;
		}
	}

	return DOWNLOAD_STATUS::NO_ATTACHEMENT;
}

int main(int argc, char *argv[])
{
	web::http::client::http_client_config config;
	config.set_validate_certificates(false);
	

	// image: e27b6442-0e5d-4bd9-80f7-f6d26b4e1993
	std::wstring body;
	Networking::Django_Connection connection(L"https://localhost/", Networking::Django_Connection::AUTHENTICATION_TYPE::TOKEN, Networking::Django_Connection::CreateCredentials(L"6199821edd5097d1ec80af2b30839c9b26946153"), config);
	if (connection.GetAuthenticationStatus() == Networking::Django_Connection::AUTHENTICATED)
	{
		web::http::http_request request(web::http::methods::GET);
		//request.set_request_uri(L"api/v1/cases/images/d7acd10c-eb2c-450a-861c-f38d495d341f");
		request.set_request_uri(L"media/images/d7acd10c-eb2c-450a-861c-f38d495d341f/Patient4.tif");

		web::http::http_response reppy;
		connection.SendRequest(request).then([&reppy](const web::http::http_response& rep)
		{
			HTTP_File_Download(rep, "");
			reppy = rep;
		}).wait();
	}
	
	QApplication a(argc, argv);

	// Creates the composite window, that'll hold all other required tabs
	ASAP::Worklist::GUI::CompositeWindow main_window;
	main_window.show();

	// Creates the worklist and viewer tabs
	ASAP::Worklist::GUI::WorklistWindow worklist;
	ASAP_Window workstation;

	// Creates the worklist shortcuts.
	std::vector<ASAP::Worklist::GUI::ShortcutAction> worklist_shortcuts;
	worklist_shortcuts.push_back(
	{
		std::function<void(void)>(std::bind(&ASAP::Worklist::GUI::WorklistWindow::MoveImageSelectionLeft, &worklist)),
		QKeySequence(Qt::CTRL + Qt::Key_Left)
	});

	worklist_shortcuts.push_back(
	{
		std::function<void(void)>(std::bind(&ASAP::Worklist::GUI::WorklistWindow::MoveImageSelectionRight, &worklist)),
		QKeySequence(Qt::CTRL + Qt::Key_Right)
	});

	// Adds the tabs to the composite window and then connects the viewer to the worklist
	main_window.AddTab(&worklist, "Worklist", worklist_shortcuts);
	int viewer_tab = main_window.AddTab(&workstation, "Viewer");
	worklist.AttachWorkstation(workstation, viewer_tab);

	return a.exec();
}
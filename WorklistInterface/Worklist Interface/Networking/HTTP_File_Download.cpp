#include "HTTP_File_Download.h"

#include "../Misc/StringConversions.h"

namespace ASAP::Worklist::Networking
{
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
				std::wstring filepath(Misc::StringToWideString(directory) + L"/" + lower_case_disposition.substr(filename_start, lower_case_disposition.size() - filename_start));
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
}
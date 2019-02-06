#include "HTTP_File_Download.h"

#include "../Misc/StringConversions.h"

namespace ASAP::Worklist::Networking
{
	FileDownloadResults HTTP_File_Download(const web::http::http_response& response, const boost::filesystem::path& output_directory)
	{
		// Fails if the path doesn't point towards a directory.
		if (!boost::filesystem::is_directory(output_directory))
		{
			return { boost::filesystem::path(), DOWNLOAD_STATUS::FILE_CREATION_FAILURE };
		}

		// Fails if the response wasn't a HTTP 200 message, or lacks the content disposition header.
		web::http::http_headers headers(response.headers());
		auto content_disposition = headers.find(L"Content-Disposition");
		if (response.status_code == web::http::status_codes::OK && content_disposition != headers.end())
		{
			// Fails if the content disposition doesn't list a filename.
			std::wstring disposition(content_disposition->second);
			std::transform(disposition.begin(), disposition.end(), disposition.begin(), ::tolower);
			if (disposition.find(L"filename=") != std::string::npos)
			{
				// Appends the filename to the output directory.
				boost::filesystem::path output_file(output_directory);
				output_file.append(boost::filesystem::path(disposition.substr(disposition.find_last_of('=')).c_str()));

				// Fails if the file can't be created and opened.
				std::shared_ptr<concurrency::streams::ostream> filestream(std::make_shared<concurrency::streams::ostream>());
				concurrency::streams::fstream::open_ostream(output_file.wstring());
				if (filestream->is_open())
				{
					try
					{
						response.body().read_to_end(filestream->streambuf()).wait();
						filestream->close();
						return { boost::filesystem::absolute(output_file), DOWNLOAD_STATUS::SUCCESS };
					}
					// TODO: Replace with specific error catching
					catch (const std::exception& e)
					{
						filestream->close();
						if (boost::filesystem::exists(output_file))
						{
							boost::filesystem::remove(output_file);
						}
						return { boost::filesystem::path(), DOWNLOAD_STATUS::DOWNLOAD_FAILURE };
					}
				}
				return { boost::filesystem::path(), DOWNLOAD_STATUS::FILE_CREATION_FAILURE };
			}
		}
		return { boost::filesystem::path(), DOWNLOAD_STATUS::NO_ATTACHMENT };
	}
}
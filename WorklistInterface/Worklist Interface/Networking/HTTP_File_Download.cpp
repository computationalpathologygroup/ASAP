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
		auto content_disposition	= headers.find(L"Content-Disposition");
		auto content_length			= headers.find(L"Content-Length");
		if (response.status_code() == web::http::status_codes::OK && content_disposition != headers.end() && content_length != headers.end())
		{
			// Fails if the content disposition doesn't list a filename.
			std::wstring disposition(content_disposition->second);
			size_t length(std::stoi(content_length->second));
			std::transform(disposition.begin(), disposition.end(), disposition.begin(), ::tolower);

			if (disposition.find(L"filename=") != std::string::npos)
			{
				// Appends the filename to the output directory.
				boost::filesystem::path output_file(output_directory);
				output_file.append(disposition.substr(disposition.find_last_of('=') + 1));

				// Checks if the file has already been downloaded.
				if (FileIsUnique(output_file, length))
				{
					// Changes filename if the binary size is unique, but the filename isn't.
					FixFilepath(output_file);

					// Fails if the file can't be created and opened.
					concurrency::streams::ostream stream;
					concurrency::streams::fstream::open_ostream(output_file.wstring()).then([&stream](concurrency::streams::ostream open_stream)
					{
						stream = open_stream;
					}).wait();

					if (stream.is_open())
					{
						response.body().read_to_end(stream.streambuf()).wait();
						stream.close().wait();

						if (FileHasCorrectSize(output_file, length))
						{
							return { boost::filesystem::absolute(output_file), DOWNLOAD_STATUS::SUCCESS };
						}
						return { boost::filesystem::path(), DOWNLOAD_STATUS::DOWNLOAD_FAILURE };
					}
					return { boost::filesystem::path(), DOWNLOAD_STATUS::FILE_CREATION_FAILURE };
				}
				// File has already been downloaded.
				{
					return { boost::filesystem::absolute(output_file), DOWNLOAD_STATUS::SUCCESS };
				}
			}
		}
		return { boost::filesystem::path(), DOWNLOAD_STATUS::NO_ATTACHMENT };
	}

	namespace
	{
		bool FileHasCorrectSize(const boost::filesystem::path& filepath, size_t size)
		{
			return boost::filesystem::exists(filepath) && boost::filesystem::file_size(filepath) == size;
		}

		bool FileIsUnique(const boost::filesystem::path& filepath, size_t size)
		{
			if (boost::filesystem::exists(filepath) && boost::filesystem::file_size(filepath) == size)
			{
				return false;
			}
			return true;
		}

		void FixFilepath(boost::filesystem::path& filepath)
		{
			while (boost::filesystem::exists(filepath))
			{
				std::string filename = filepath.leaf().string();

				size_t version = 0;
				if (filename.find('('))
				{
					size_t value_start	= filename.find_last_of('(') + 1;
					size_t value_end	= filename.find_last_of(')');
					version = std::stoi(filename.substr(value_start, value_end - value_start));
				}

				size_t dot_location = filename.find_first_of('.');
				std::string new_filename = filename.substr(0, dot_location) + "(" + std::to_string(version) + ")" + filename.substr(dot_location);
				
				filepath.remove_leaf() /= new_filename;
			}
		}
	}
}
#ifndef __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#define __ASAP_NETWORKING_HTTPFILEDOWNLOAD__

#include <functional>
#include <string> 
#include <thread>

#include <boost/filesystem.hpp>
#include <cpprest/filestream.h>
#include <cpprest/http_client.h>

namespace ASAP::Networking
{
	enum DownloadStatus { DOWNLOAD_FAILURE, FILE_CREATION_FAILURE, NO_ATTACHMENT, SUCCESS };

	struct FileDownloadResults
	{
		boost::filesystem::path filepath;
		DownloadStatus			status;
	};

	/// <summary>
	/// Downloads a file that may be tied to a HTTP response.
	/// </summary>
	/// <param name="response">The response holding the file.</param>
	/// <param name="output_directory">The directory to write the image to.</param>
	/// <return>A struct containing the absolute path to the downloaded file (empty on failure) and the state of the download.</return>
	FileDownloadResults HTTP_File_Download(const web::http::http_response& response, const boost::filesystem::path& output_directory, std::function<void(uint8_t)> observer = std::function<void(uint8_t)>());

	namespace
	{
		bool FileHasCorrectSize(const boost::filesystem::path& filepath, size_t size);
		bool FileIsUnique(const boost::filesystem::path& filepath, size_t size);
		void FixFilepath(boost::filesystem::path& filepath);
		std::thread StartMonitorThread(const bool& stop, const size_t length, concurrency::streams::ostream& stream, std::function<void(uint8_t)>& observer);
	}
}
#endif // __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#ifndef __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#define __ASAP_NETWORKING_HTTPFILEDOWNLOAD__

#include <string> 

#include <boost/filesystem.hpp>
#include <cpprest/filestream.h>
#include <cpprest/http_client.h>

namespace ASAP::Worklist::Networking
{
	enum DOWNLOAD_STATUS { DOWNLOAD_FAILURE, FILE_CREATION_FAILURE, NO_ATTACHMENT, SUCCESS };

	struct FileDownloadResults
	{
		boost::filesystem::path filepath;
		DOWNLOAD_STATUS			status;
	};

	/// <summary>
	/// Downloads a file that may be tied to a HTTP response.
	/// </summary>
	/// <param name="response">The response holding the file.</param>
	/// <param name="output_directory">The directory to write the image to.</param>
	/// <return>A struct containing the absolute path to the downloaded file (empty on failure) and the state of the download.</return>
	FileDownloadResults HTTP_File_Download(const web::http::http_response& response, const boost::filesystem::path& output_directory);
}
#endif // __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#ifndef __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#define __ASAP_NETWORKING_HTTPFILEDOWNLOAD__

#include <string> 

#include <boost/filesystem.hpp>
#include <cpprest/filestream.h>
#include <cpprest/http_client.h>

namespace ASAP::Worklist::Networking
{
	struct FileDownloadResults
	{
		boost::filesystem::path filepath;
		DOWNLOAD_STATUS			status;
	};

	enum DOWNLOAD_STATUS { DOWNLOAD_FAILURE, FILE_CREATION_FAILURE, NO_ATTACHMENT, SUCCESS };

	/// <summary>
	/// 
	/// </summary>
	/// <param name="response"></param>
	/// <param name="directory"></param>
	/// <return></return>
	FileDownloadResults HTTP_File_Download(const web::http::http_response& response, const boost::filesystem::path& output_directory);
}
#endif // __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#ifndef __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#define __ASAP_NETWORKING_HTTPFILEDOWNLOAD__

#include <string> 


#include <cpprest/filestream.h>
#include <cpprest/http_client.h>

namespace ASAP::Worklist::Networking
{
	enum DOWNLOAD_STATUS { DOWNLOAD_FAILURE, FILE_CREATION_FAILURE, NO_ATTACHEMENT, SUCCESS };

	/// <summary>
	/// 
	/// </summary>
	/// <param name="response"></param>
	/// <param name="directory"></param>
	/// <return></return>
	DOWNLOAD_STATUS HTTP_File_Download(const web::http::http_response& response, const std::string directory);
}
#endif // __ASAP_NETWORKING_HTTPFILEDOWNLOAD__
#ifndef __ASAP_SERIALIZATION_JSON__
#define __ASAP_SERIALIZATION_JSON__

#include <cpprest/http_client.h>

#include "Data/DataTable.h"

namespace ASAP::Serialization::JSON
{
	web::json::object GetTagRecursive(std::wstring tag, const web::json::value& json);

	int ParseJsonResponseToRecords(const web::http::http_response& response, Data::DataTable& table);
	int ParseJsonResponseToTableSchema(const web::http::http_response& response, Data::DataTable& table);
}
#endif // __ASAP_SERIALIZATION_JSON__
#pragma once
#include <cpprest/http_client.h>

#include "Data/DataTable.h"

namespace ASAP::Worklist::Serialization::JSON
{
	web::json::object GetTagRecursive(std::wstring tag, const web::json::value& json);

	int ParseJsonResponseToRecords(const web::http::http_response& response, DataTable& table);
	DataTable ParseJsonResponseToTable(const web::http::http_response& response);
	int ParseJsonResponseToTableSchema(const web::http::http_response& response, DataTable& table);
}
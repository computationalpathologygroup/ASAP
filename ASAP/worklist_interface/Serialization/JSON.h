#ifndef __ASAP_SERIALIZATION_JSON__
#define __ASAP_SERIALIZATION_JSON__

#include <functional>

#include <cpprest/http_client.h>

#include "../Data/DataTable.h"
#include "../Networking/Django_Connection.h"

namespace ASAP { namespace JSON
{
		web::json::object GetTagRecursive(std::wstring tag, const web::json::value& json);
		std::vector<std::string> ParseJsonObjectToHeaderVector(const web::json::object& object);
		std::vector<std::string> ParseJsonObjectToValueVector(const web::json::object& object);

		//int ParseJsonFieldsToVector(Django_Connection& connection, const web::http::http_response& response, const std::vector<std::string> fields, std::vector<std::vector<std::string>>& results);
		//int ParseJsonResponseToRecords(const web::http::http_response& response, DataTable& table);
		//int ParseJsonResponseToTable(const web::http::http_response& response, DataTable& table);
		int OptionsResponseToTableSchema(const web::http::http_response& response, DataTable& table);

		int ResponseToFilteredTable(Django_Connection& connection, const web::http::http_response& response, DataTable& table, std::vector<std::string>& fields);
		int ResponseToTable(Django_Connection& connection, const web::http::http_response& response, DataTable& table);

		namespace
		{
			int ParseJsonResponse(Django_Connection& connection, const web::http::http_response& response, DataTable& table, std::function<void(web::json::value&, DataTable&)> parser);
		}
} }
#endif // __ASAP_SERIALIZATION_JSON__
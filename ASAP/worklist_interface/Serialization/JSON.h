#ifndef __ASAP_SERIALIZATION_JSON__
#define __ASAP_SERIALIZATION_JSON__

#include <functional>

#include <cpprest/http_client.h>

#include "../Data/DataTable.h"
#include "../Networking/Django_Connection.h"

namespace ASAP { namespace JSON
{
		web::json::object getTagRecursive(std::wstring tag, const web::json::value& json);
		std::vector<std::string> parseJsonObjectToHeaderVector(const web::json::object& object);
		std::vector<std::string> parseJsonObjectToValueVector(const web::json::object& object);

		//int ParseJsonFieldsToVector(Django_Connection& connection, const web::http::http_response& response, const std::vector<std::string> fields, std::vector<std::vector<std::string>>& results);
		//int ParseJsonResponseToRecords(const web::http::http_response& response, DataTable& table);
		//int ParseJsonResponseToTable(const web::http::http_response& response, DataTable& table);
		int optionsResponseToTableSchema(const web::http::http_response& response, DataTable& table);

		int responseToFilteredTable(Django_Connection& connection, const web::http::http_response& response, DataTable& table, std::vector<std::string>& fields);
		int responseToTable(Django_Connection& connection, const web::http::http_response& response, DataTable& table);

		namespace
		{
			int parseJsonResponse(Django_Connection& connection, const web::http::http_response& response, DataTable& table, std::function<void(web::json::value&, DataTable&)> parser);
		}
} }
#endif // __ASAP_SERIALIZATION_JSON__
#include "JSON.h"

#include "core/stringconversion.h"

namespace ASAP { namespace JSON
{
	web::json::object getTagRecursive(std::wstring tag, const web::json::value& json)
	{
		web::json::object object(json.as_object());
		for (auto it = object.cbegin(); it != object.cend(); ++it)
		{
			if (it->first == tag)
			{
				return it->second.as_object();
			}
			else if (it->second.size() > 0)
			{
				return getTagRecursive(tag, it->second);
			}
		}

		throw std::runtime_error("Tag not found.");
	}

	std::vector<std::string> parseJsonObjectToHeaderVector(const web::json::object& object)
	{
		std::vector<std::string> header;
		for (auto it = object.cbegin(); it != object.cend(); ++it)
		{
			std::string column(core::wideStringToString(it->first));
			column.erase(std::remove(column.begin(), column.end(), '"'), column.end());
			header.push_back(column);
		}
		return header;
	}

	std::vector<std::string> parseJsonObjectToValueVector(const web::json::object& object)
	{
		std::vector<std::string> values;

		for (auto it = object.cbegin(); it != object.cend(); ++it)
		{
			std::string value;
			if (!it->second.is_null())
			{
				if (it->second.is_array())
				{
					std::wstringstream value_stream;
					web::json::array arr(it->second.as_array());
					for (auto val = arr.begin(); val != arr.end(); ++val)
					{
						value_stream << val->as_string();
						if (val != --arr.end())
						{
							value_stream << ',';
						}
					}
					value = core::wideStringToString(value_stream.str());
				}
				else
				{
					value = core::wideStringToString(it->second.to_string());
				}
			}
			value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
			values.push_back(value);
		}
		return values;
	}

	int responseToFilteredTable(DjangoConnection& connection, const web::http::http_response& response, DataTable& table, std::vector<std::string>& fields)
	{
		return parseJsonResponse(connection, response, table, [&fields](web::json::value& json, DataTable& table)
		{
			std::vector<std::wstring> conv_fields(core::stringsToWideStrings(fields));
			for (size_t obj = 0; obj < json.size(); ++obj)
			{
				auto object = json[obj];
				std::vector<std::string> record;
				for (const std::wstring& field : conv_fields)
				{
					std::string field(core::wideStringToString(object.at(field).serialize()));
					field.erase(std::remove(field.begin(), field.end(), '"'), field.end());
					record.push_back(field);
				}
				table.insert(record);
			}
		});

	}

	int optionsResponseToTableSchema(const web::http::http_response& response, DataTable& table)
	{
		int error_code = 0;
		response.extract_json().then([&table, &error_code](pplx::task<web::json::value> previous_task)
		{
			try
			{
				web::json::value json_object(previous_task.get());
				try
				{
					web::json::object post_actions(getTagRecursive(L"POST", json_object));

					std::vector<std::string> columns;
					for (auto it = post_actions.cbegin(); it != post_actions.cend(); ++it)
					{
						columns.push_back(core::wideStringToString(it->first));
					}

					table = DataTable(columns);
				}
				catch (const std::exception& e)
				{
					// Indicates a parsing error.
					error_code = -1;
				}
			}
			catch (const web::http::http_exception& e)
			{
				error_code = e.error_code().value();
			}
		}).wait();
		return error_code;
	}

	int responseToTable(DjangoConnection& connection, const web::http::http_response& response, DataTable& table)
	{
		return parseJsonResponse(connection, response, table, [](web::json::value& json, DataTable& table)
		{
			if (json.is_array())
			{
				if (table.getColumnCount() == 0)
				{
					table = DataTable(parseJsonObjectToHeaderVector(json[0].as_object()));
				}
				for (size_t o = 0; o < json.size(); ++o)
				{
					table.insert(parseJsonObjectToValueVector(json[o].as_object()));
				}
			}
			else
			{
				table = DataTable(parseJsonObjectToHeaderVector(json.as_object()));
				table.insert(parseJsonObjectToValueVector(json.as_object()));
			}
		});
	}

	int parseJsonResponse(DjangoConnection& connection, const web::http::http_response& response, DataTable& table, std::function<void(web::json::value&, DataTable&)> parser)
		{
			int error_code = 0;

			try
			{
				web::json::value json(response.extract_json().get());
				if (json.has_field(L"count") && json.has_field(L"next") && json.has_field(L"results"))
				{
					parser(json[L"results"], table);
					while (json[L"next"].to_string() != L"null")
					{
						web::http::http_request page_request(web::http::methods::GET);
						page_request.set_request_uri(json[L"next"].as_string());
						json	= connection.sendRequest(page_request).get().extract_json().get();
						parser(json[L"results"], table);
					}
				}
				else
				{
					parser(json, table);
				}
			}
			catch (const web::http::http_exception& e)
			{
				error_code = e.error_code().value();
			}
			catch (const std::exception& e)
			{
				error_code = -1;
			}

			return error_code;
		}
} }
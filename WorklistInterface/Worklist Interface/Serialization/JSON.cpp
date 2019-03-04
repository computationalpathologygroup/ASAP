#include "JSON.h"

#include "../Misc/StringConversions.h"

namespace ASAP::Serialization::JSON
{
	web::json::object GetTagRecursive(std::wstring tag, const web::json::value& json)
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
				return GetTagRecursive(tag, it->second);
			}
		}

		throw std::runtime_error("Tag not found.");
	}

	int ParseJsonFieldsToVector(const web::http::http_response& response, const std::vector<std::string> fields, std::vector<std::vector<std::string>>& results)
	{
		std::vector<std::wstring> converted_fields(Misc::StringsToWideStrings(fields));

		int error_code = 0;
		try
		{
			web::json::value json_response = response.extract_json().get();;
			try
			{
				for (size_t obj = 0; obj < json_response.size(); ++obj)
				{
					auto object = json_response[obj];
					results.push_back(std::vector<std::string>());

					for (const std::wstring& field : converted_fields)
					{
						results.back().push_back(Misc::WideStringToString(object.at(field).serialize()));
					}
				}
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

		return error_code;
	}

	int ParseJsonResponseToRecords(const web::http::http_response& response, Data::DataTable& table)
	{
		int error_code = 0;
		response.extract_json().then([&table, &error_code](pplx::task<web::json::value> previousTask)
		{
			try
			{
				web::json::value json_response(previousTask.get());
				if (json_response.size() > 0)
				{
					std::vector<std::string> values;
					values.reserve(json_response[0].as_object().size());

					for (size_t o = 0; o < json_response.size(); ++o)
					{
						web::json::object object(json_response[o].as_object());
						for (auto it = object.cbegin(); it != object.cend(); ++it)
						{
							values.push_back(Misc::WideStringToString(it->second.to_string()));

							if (values.back()[0] == '"' && values.back()[values.back().size() - 1] == '"')
							{
								values.back() = values.back().substr(1, values.back().size() - 2);
							}
							else if (values.back() == "null")
							{
								values.back().clear();
								values.back().shrink_to_fit();
							}
						}

						table.Insert(values);
						values.clear();
					}
				}
			}
			catch (const web::http::http_exception& e)
			{
				// Indicates a parsing error.
				error_code = e.error_code().value();
			}
			catch (const std::exception& e)
			{
				// Indicates a parsing error.
				error_code = -1;
			}
		}).wait();
		return error_code;
	}

	int ParseJsonResponseToTableSchema(const web::http::http_response& response, Data::DataTable& table)
	{
		int error_code = 0;
		response.extract_json().then([&table, &error_code](pplx::task<web::json::value> previous_task)
		{
			try
			{
				web::json::value json_object(previous_task.get());
				try
				{
					web::json::object post_actions(GetTagRecursive(L"POST", json_object));

					std::vector<std::string> columns;
					for (auto it = post_actions.cbegin(); it != post_actions.cend(); ++it)
					{
						columns.push_back(Misc::WideStringToString(it->first));
					}

					table = Data::DataTable(columns);
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
}
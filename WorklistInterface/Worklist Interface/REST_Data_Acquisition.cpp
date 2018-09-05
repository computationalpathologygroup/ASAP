#include "REST_Data_Acquisition.h"

#include <codecvt>
#include <stdexcept>
#include <fstream>
#include <locale>
#include <cpprest/json.h>

namespace ASAP::Worklist::Data
{
	// Located within the namespace
	void ParseJsonResponse(web::http::http_response& response, const std::function<void(DataTable, const web::http::http_exception&)>& table_receiver)
	{
		response.extract_json().then([table_receiver](pplx::task<web::json::value> previousTask)
		{
			try
			{
				DataTable table;
				web::json::array objects(previousTask.get().as_array());
				if (objects.size() > 0)
				{
					std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
					// Builds the table with the correct headers.
					{
						std::vector<std::string> column_names;
						web::json::object concrete_object(objects[0].as_object());
						for (auto it = concrete_object.cbegin(); it != concrete_object.cend(); ++it)
						{
							column_names.push_back(converter.to_bytes(it->first));
						}

						table = DataTable(column_names);
					}

					std::vector<std::string> values;
					values.reserve(objects[0].size());

					for (web::json::value& object : objects)
					{
						web::json::object concrete_object(object.as_object());
						for (auto it = concrete_object.cbegin(); it != concrete_object.cend(); ++it)
						{
							values.push_back(converter.to_bytes(it->second.as_string()));
						}

						table.Insert(values);
						values.clear();
					}
				}

				table_receiver(table, web::http::http_exception("0"));
			}
			catch (const web::http::http_exception& e)
			{
				// Return an empty DataTable to indicate an error occurred
				table_receiver(DataTable(), e);
			}
		}).wait();
	}

	DjangoDataAcquisition::DjangoDataAcquisition(const DjangoRestURI uri_info) : m_client_(uri_info.base_url), m_rest_uri_(uri_info)
	{
	}

	size_t DjangoDataAcquisition::GetWorklistRecords(const std::function<void(DataTable, const web::http::http_exception&)>& table_receiver)
	{
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.worklist_addition);
		ProcessRequest_(request, [table_receiver](web::http::http_response& response)
		{
			ParseJsonResponse(response, table_receiver);
		});
	}

	size_t DjangoDataAcquisition::GetPatientRecords(const std::function<void(DataTable, const web::http::http_exception&)>& table_receiver, const size_t worklist_index)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(worklist_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.patient_addition + L"/" + index_wide_char);
		ProcessRequest_(request, [table_receiver](web::http::http_response& response)
		{
			ParseJsonResponse(response, table_receiver);
		});
	}

	size_t DjangoDataAcquisition::GetStudyRecords(const std::function<void(DataTable, const web::http::http_exception&)>& table_receiver, const size_t patient_index)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(patient_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.study_addition + L"/" + index_wide_char);
		ProcessRequest_(request, [table_receiver](web::http::http_response& response)
		{
			ParseJsonResponse(response, table_receiver);
		});
	}

	size_t DjangoDataAcquisition::GetImageRecords(const std::function<void(DataTable, const web::http::http_exception&)>& table_receiver, const size_t study_index)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(study_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.image_addition + L"/" + index_wide_char);
		ProcessRequest_(request, [table_receiver](web::http::http_response& response)
		{
			ParseJsonResponse(response, table_receiver);
		});
	}

	std::vector<std::string> DjangoDataAcquisition::GetPatientHeaders(void)
	{
	}

	std::vector<std::string> DjangoDataAcquisition::GetStudyHeaders(void)
	{
	}

	std::vector<std::string> DjangoDataAcquisition::GetImageHeaders(void)
	{
	}


	void DjangoDataAcquisition::CancelTask(const size_t task_id)
	{
		auto task = m_active_tasks_.find(task_id);
		if (task != m_active_tasks_.end())
		{
			if (!task->second.task.is_done())
			{
				task->second.token.cancel();
			}
			m_active_tasks_.erase(task_id);
		}
	}

	void DjangoDataAcquisition::ProcessRequest_(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer)
	{
		size_t token_id = m_token_counter_;
		m_active_tasks_.insert({ m_token_counter_, TokenTaskPair() });
		auto inserted_pair(m_active_tasks_.find(token_id));
		++m_token_counter_;

		// Catches the response so the attached token can be removed.
		inserted_pair->second.task = m_client_.request(request, inserted_pair->second.token.get_token()).then([observer, token_id, this](web::http::http_response& response)
		{
			// Remove token
			this->CancelTask(token_id);

			// Passes the response to the observer
			observer(response);
		});
	}
}
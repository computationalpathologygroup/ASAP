#include "REST_Data_Acquisition.h"

#include <codecvt>
#include <stdexcept>
#include <fstream>
#include <locale>
#include <cpprest/json.h>

#include <system_error>

namespace ASAP::Worklist::Data
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

	// Located within the namespace
	int ParseJsonToRecords(const web::http::http_response& response, DataTable& table)
	{
		int error_code = 0;
		response.extract_json().then([&table, &error_code](pplx::task<web::json::value> previousTask)
		{
			try
			{
				web::json::array objects(previousTask.get().as_array());
				if (objects.size() > 0)
				{
					std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

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
			}
			catch (const web::http::http_exception& e)
			{
				// Leave table empty to notify an error.
				table = DataTable();
				error_code = e.error_code().value();
			}
		}).wait();
		return error_code;
	}

	int ParseJsonToTable(const web::http::http_response& response, DataTable& table)
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

					std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
					std::vector<std::string> columns;
					for (auto it = post_actions.cbegin(); it != post_actions.cend(); ++it)
					{
						columns.push_back(converter.to_bytes(it->first));
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

	DjangoDataAcquisition::DjangoDataAcquisition(const DjangoRestURI uri_info) : m_client_(uri_info.base_url), m_rest_uri_(uri_info), m_tables_(5)
	{
		InitializeTables_();
	}

	size_t DjangoDataAcquisition::GetWorklistRecords(const std::function<void(DataTable, const int)>& receiver)
	{
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.worklist_addition);

		DataTable* table(&m_tables_[TableEntry::WORKLIST]);
		return ProcessRequest_(request, [receiver, table](web::http::http_response& response)
		{
			// Parses response into the data table, and then returns both the table and the error code to the receiver.
			receiver(*table, ParseJsonToRecords(response, *table));
		});
	}

	size_t DjangoDataAcquisition::GetPatientRecords(const std::function<void(DataTable, const int)>& receiver, const size_t worklist_index)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(worklist_index));
		
		// Acquires the many to many worklist <-> patient relations for the given index.
		web::http::http_request relations_request(web::http::methods::GET);
		relations_request.set_request_uri(L"/" + m_rest_uri_.worklist_patient_relation_addition + L"/?worklist=" + index_wide_char);

		DataTable* relation_table(&m_tables_[TableEntry::WORKLIST_PATIENT_RELATION]);
		DataTable* patient_table(&m_tables_[TableEntry::PATIENT]);
		web::http::client::http_client* client(&m_client_);
		std::wstring* patient_addition(&m_rest_uri_.patient_addition);

		return ProcessRequest_(relations_request, [relation_table, patient_table, client, patient_addition, receiver](web::http::http_response& response)
		{
			ParseJsonToRecords(response, *relation_table);

			std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
			std::vector<std::string> field_selection({ "id" });

			std::wstring ids;
			for (size_t relation = 0; relation < relation_table->Size(); ++relation)
			{
				ids += converter.from_bytes(*relation_table->At(relation, field_selection)[0]);
				if (relation != relation_table->Size() - 1)
				{
					ids += L",";
				}
			}

			web::http::http_request patient_request(web::http::methods::GET);
			patient_request.set_request_uri(L"/" + *patient_addition + L"/ids=" + ids);
			client->request(patient_request).then([patient_table](web::http::http_response& response)
			{
				ParseJsonToRecords(response, *patient_table);
			}).wait();
		});
	}

	size_t DjangoDataAcquisition::GetStudyRecords(const std::function<void(DataTable, const int)>& receiver, const size_t patient_index)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(patient_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.study_addition + L"/?patient=" + index_wide_char);

		DataTable* table(&m_tables_[TableEntry::WORKLIST]);
		table->Clear();
		return ProcessRequest_(request, [receiver, table](web::http::http_response& response)
		{
			// Parses response into the data table, and then returns both the table and the error code to the receiver.
			receiver(*table, ParseJsonToRecords(response, *table));
		});
	}

	size_t DjangoDataAcquisition::GetImageRecords(const std::function<void(DataTable, int)>& receiver, const size_t study_index)
	{
		std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(study_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.image_addition + L"/?study=" + index_wide_char);

		DataTable* table(&m_tables_[TableEntry::WORKLIST]);
		table->Clear();
		return ProcessRequest_(request, [receiver, table](web::http::http_response& response)
		{
			// Parses response into the data table, and then returns both the table and the error code to the receiver.
			receiver(*table, ParseJsonToRecords(response, *table));
		});
	}

	std::vector<std::string> DjangoDataAcquisition::GetPatientHeaders(void)
	{
		return m_tables_[TableEntry::PATIENT].GetColumnNames();
	}

	std::vector<std::string> DjangoDataAcquisition::GetStudyHeaders(void)
	{
		return m_tables_[TableEntry::STUDY].GetColumnNames();
	}

	std::vector<std::string> DjangoDataAcquisition::GetImageHeaders(void)
	{
		return m_tables_[TableEntry::IMAGE].GetColumnNames();
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

	void DjangoDataAcquisition::InitializeTables_(void)
	{
		std::vector<std::wstring> table_url_addition
		({
			m_rest_uri_.worklist_addition,
			m_rest_uri_.patient_addition,
			m_rest_uri_.study_addition,
			m_rest_uri_.image_addition,
			m_rest_uri_.worklist_patient_relation_addition
		});
		for (size_t table = 0; table < table_url_addition.size(); ++table)
		{
			web::http::http_request request(web::http::methods::OPTIONS);
			request.set_request_uri(L"/" + table_url_addition[table]);

			DataTable* datatable(&m_tables_[table]);
			ProcessRequest_(request, [datatable](web::http::http_response& response)
			{
				ParseJsonToTable(response, *datatable);
			});
		}
	}

	size_t DjangoDataAcquisition::ProcessRequest_(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer)
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

		return token_id;
	}
}
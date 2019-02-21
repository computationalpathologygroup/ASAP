#include "GrandChallengeDataAcquisition.h"

#include <codecvt>
#include <stdexcept>
#include <locale>
#include <system_error>
#include <cstdio>

#include "../Serialization/JSON.h"

namespace ASAP::Data
{
	GrandChallengeDataAcquisition::GrandChallengeDataAcquisition(const GrandChallengeURLInfo uri_info, const Networking::Django_Connection::Credentials credentials)
		: m_connection_(uri_info.base_url, Networking::Django_Connection::AuthenticationType::TOKEN, credentials), m_rest_uri_(uri_info), m_schemas_(5)
	{
		InitializeTables_();
	}

	WorklistDataAcquisitionInterface::SourceType GrandChallengeDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FULL_WORKLIST;
	}

	GrandChallengeURLInfo GrandChallengeDataAcquisition::GetStandardURI(const std::wstring base_url)
	{
		return { base_url, L"worklists/list/", L"worklists/set/", L"patients/patient/", L"studies/study/", L"api/cases/images", L"api/cases/images" };
	}

	size_t GrandChallengeDataAcquisition::AddWorklistRecord(const std::string& title, std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t GrandChallengeDataAcquisition::UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.worklist_set_addition + L"?token=");

		DataTable* worklist_schema		= &m_schemas_[TableEntry::WORKLIST_SET];
		DataTable* worklist_set_schema	= &m_schemas_[TableEntry::WORKLIST_SET];
		return m_connection_.QueueRequest(request, [receiver, worklist_schema, worklist_set_schema](web::http::http_response& response)
		{
			DataTable worklist_set(*worklist_set_schema);
			// Parses response into the data table, and then returns both the table and the error code to the receiver.
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, worklist_set);
			receiver(worklist_set, error_code);
		});
	}

	size_t GrandChallengeDataAcquisition::GetPatientRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		/*std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(worklist_index));
		
		// Acquires the many to many worklist <-> patient relations for the given index.
		web::http::http_request relations_request(web::http::methods::GET);
		relations_request.set_request_uri(L"/" + m_rest_uri_.worklist_patient_relation_addition + L"/?worklist=" + index_wide_char);

		DataTable* relation_table(&m_tables_[TableEntry::WORKLIST_PATIENT_RELATION]);
		DataTable* patient_table(&m_tables_[TableEntry::PATIENT]);
		Networking::HTTP_Connection* connection(&m_connection_);
		std::wstring* patient_addition(&m_rest_uri_.patient_addition);

		relation_table->Clear();
		patient_table->Clear();

		return m_connection_.QueueRequest(relations_request, [relation_table, patient_table, connection, patient_addition, receiver](web::http::http_response& response)
		{
			Serialization::JSON::ParseJsonResponseToRecords(response, *relation_table);

			std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
			std::vector<std::string> field_selection({ "patient" });

			std::wstring ids;
			for (size_t relation = 0; relation < relation_table->Size(); ++relation)
			{
				ids += converter.from_bytes(*relation_table->At(relation, field_selection)[0]);
				if (relation != relation_table->Size() - 1)
				{
					ids += L",";
				}
			}

			if (!ids.empty())
			{
				web::http::http_request patient_request(web::http::methods::GET);
				patient_request.set_request_uri(L"/" + *patient_addition + L"/?ids=" + ids);
				connection->SendRequest(patient_request).then([patient_table](web::http::http_response& response)
				{
					Serialization::JSON::ParseJsonResponseToRecords(response, *patient_table);
				}).wait();
			}

			receiver(*patient_table, 0);
		});*/

		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		/*std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(patient_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.study_addition + L"/?patient=" + index_wide_char);

		DataTable* table(&m_tables_[TableEntry::STUDY]);
		table->Clear();
		return m_connection_.QueueRequest(request, [receiver, table](web::http::http_response& response)
		{
			// Parses response into the data table, and then returns both the table and the error code to the receiver.
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, *table);
			receiver(*table, error_code);
		});*/
		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetImageRecords(const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		/*std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
		std::wstring index_wide_char = converter.from_bytes(std::to_string(study_index));

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.image_addition + L"/?study=" + index_wide_char);

		DataTable* table(&m_tables_[TableEntry::IMAGE]);
		table->Clear();
		return m_connection_.QueueRequest(request, [receiver, table](web::http::http_response& response)
		{
			// Parses response into the data table, and then returns both the table and the error code to the receiver.
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, *table);
			receiver(*table, error_code);
		});*/
		return 0;
	}

	std::set<std::string> GrandChallengeDataAcquisition::GetWorklistHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas_[TableEntry::WORKLIST].GetColumnNames(selection);
	}

	std::set<std::string> GrandChallengeDataAcquisition::GetPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas_[TableEntry::PATIENT].GetColumnNames(selection);
	}

	std::set<std::string> GrandChallengeDataAcquisition::GetStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas_[TableEntry::STUDY].GetColumnNames(selection);
	}

	std::set<std::string> GrandChallengeDataAcquisition::GetImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas_[TableEntry::IMAGE].GetColumnNames(selection);
	}

	size_t GrandChallengeDataAcquisition::GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver)
	{
		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver)
	{
		return 0;
	}

	void GrandChallengeDataAcquisition::CancelTask(size_t id)
	{
		m_connection_.CancelTask(id);
	}

	void GrandChallengeDataAcquisition::InitializeTables_(void)
	{
		std::vector<std::wstring> table_url_addition
		({
			m_rest_uri_.worklist_addition,
			m_rest_uri_.worklist_set_addition,
			m_rest_uri_.patient_addition,
			m_rest_uri_.study_addition,
			m_rest_uri_.image_addition,
		});

		for (size_t table = 0; table < table_url_addition.size(); ++table)
		{
			web::http::http_request request(web::http::methods::OPTIONS);
			request.set_request_uri(L"/" + table_url_addition[table] + L"/");

			DataTable* datatable(&m_schemas_[table]);
			m_connection_.SendRequest(request).then([datatable](web::http::http_response& response)
			{
				Serialization::JSON::ParseJsonResponseToTableSchema(response, *datatable);
			}).wait();
		}
	}
}
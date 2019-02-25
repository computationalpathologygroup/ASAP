#include "GrandChallengeDataAcquisition.h"

#include <codecvt>
#include <stdexcept>
#include <locale>
#include <system_error>
#include <cstdio>

#include "../Misc/StringConversions.h"
#include "../Serialization/JSON.h"

namespace ASAP::Data
{
	GrandChallengeDataAcquisition::GrandChallengeDataAcquisition(const GrandChallengeURLInfo uri_info, const Networking::Django_Connection::Credentials credentials, const web::http::client::http_client_config& config)
		: m_connection_(uri_info.base_url, Networking::Django_Connection::AuthenticationType::TOKEN, credentials, config), m_rest_uri_(uri_info), m_schemas_(4)
	{
		InitializeTables_();
	}

	WorklistDataAcquisitionInterface::SourceType GrandChallengeDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FULL_WORKLIST;
	}

	GrandChallengeURLInfo GrandChallengeDataAcquisition::GetStandardURI(const std::wstring base_url)
	{
		return { base_url, L"worklists/list/", L"worklists/set/", L"patients/patient/", L"studies/study/", L"api/v1/cases/images/" };
	}

	size_t GrandChallengeDataAcquisition::AddWorklistRecord(const std::string& title, std::function<void(const bool)>& observer)
	{
		web::http::http_request request(web::http::methods::GET);


		return 0;
	}

	size_t GrandChallengeDataAcquisition::UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		web::http::http_request set_request(web::http::methods::GET);
		set_request.set_request_uri(L"/" + m_rest_uri_.worklist_set_addition);

		Networking::Django_Connection* connection = &m_connection_;
		DataTable* worklist_schema					= &m_schemas_[TableEntry::WORKLIST];		
		GrandChallengeURLInfo* info					= &m_rest_uri_;
		return m_connection_.QueueRequest(set_request, [connection, receiver, worklist_schema, info](web::http::http_response& response)
		{
			// If there is a set, acquire worklists.
			web::json::value set_json(response.extract_json().get());
			if (set_json.size() > 0)
			{
				web::http::http_request list_request(web::http::methods::GET);
				list_request.set_request_uri(L"/" + info->worklist_addition);

				web::http::http_response list_response(connection->SendRequest(list_request).get());
				DataTable worklists(*worklist_schema);
				int error_code = Serialization::JSON::ParseJsonResponseToRecords(list_response, worklists);
				receiver(worklists, error_code);
			}
			// If there's no set, only create one.
			else
			{
			/*	std::wstringstream body;
				body << L"{ "

				web::http::http_request set_creation(web::http::methods::POST);
				set_creation.set_request_uri(L"/" + info->worklist_set_addition + L"/");
				set_creation.set_body()
				
				
				
				, info->worklist_set_addition, L"application/json");*/
			}
		});
	}

	size_t GrandChallengeDataAcquisition::GetPatientRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.patient_addition);

		DataTable* patient_schema = &m_schemas_[TableEntry::PATIENT];
		return m_connection_.QueueRequest(request, [receiver, patient_schema](web::http::http_response& response)
		{
			// Parses the worklist sets into a data table.
			DataTable patients(*patient_schema);
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, patients);
			receiver(patients, error_code);
		});
	}

	size_t GrandChallengeDataAcquisition::GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.patient_addition << L"?worklist=" << Misc::StringToWideString(worklist_index);

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		DataTable* patient_schema = &m_schemas_[TableEntry::PATIENT];
		return m_connection_.QueueRequest(request,[receiver, patient_schema](web::http::http_response& response)
		{
			// Parses the worklist sets into a data table.
			DataTable patients(*patient_schema);
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, patients);
			receiver(patients, error_code);
		});

		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.study_addition << L"?patient=" << Misc::StringToWideString(patient_index);

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		DataTable* study_schema = &m_schemas_[TableEntry::STUDY];
		m_connection_.SendRequest(request).then([receiver, study_schema](web::http::http_response& response)
		{
			// Parses the worklist sets into a data table.
			DataTable studies(*study_schema);
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, studies);
			receiver(studies, error_code);
		}).wait();

		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition << L"?study=" << Misc::StringToWideString(study_index);		
		if (worklist_index.empty() == false)
		{
			url << L",worklist=" << Misc::StringToWideString(worklist_index);
		}

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());
		
		// TOO: Add this section as json method that can extract certain fields from a json array
		DataTable* image_schema = &m_schemas_[TableEntry::IMAGE];
		return m_connection_.QueueRequest(request, [image_schema, receiver](web::http::http_response& response)
		{
			DataTable images(*image_schema);
			int error_code;

			try
			{
				web::json::value json_response = response.extract_json().get();;
				try
				{
					for (size_t obj = 0; obj < json_response.size(); ++obj)
					{
						auto object = json_response[obj];
						images.Insert(
						{
							Misc::WideStringToString(object.at(L"pk").serialize()),
							Misc::WideStringToString(object.at(L"name").serialize())
						});				
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

			receiver(images, error_code);
		});
	}

	size_t GrandChallengeDataAcquisition::GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(float)> observer)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition << L"/" << Misc::StringToWideString(image_index) << L"/";

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		DataTable* image_schema = &m_schemas_[TableEntry::IMAGE];
		return m_connection_.QueueRequest(request, [image_schema, receiver, observer](web::http::http_response& response)
		{

		});
	}

	size_t GrandChallengeDataAcquisition::GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(float)> observer)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition << L"/" << Misc::StringToWideString(image_index) << L"/";

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		DataTable* image_schema = &m_schemas_[TableEntry::IMAGE];
		return m_connection_.QueueRequest(request, [image_schema, receiver, observer](web::http::http_response& response)
		{

		});
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

	void GrandChallengeDataAcquisition::CancelTask(size_t id)
	{
		m_connection_.CancelTask(id);
	}

	void GrandChallengeDataAcquisition::InitializeTables_(void)
	{
		std::vector<std::wstring> table_url_addition
		({
			m_rest_uri_.worklist_addition,
			m_rest_uri_.patient_addition,
			m_rest_uri_.study_addition
		});

		for (size_t table = 0; table < table_url_addition.size(); ++table)
		{
			web::http::http_request request(web::http::methods::OPTIONS);
			request.set_request_uri(L"/" + table_url_addition[table]);

			DataTable* datatable(&m_schemas_[table]);
			m_connection_.SendRequest(request).then([datatable](web::http::http_response& response)
			{
				Serialization::JSON::ParseJsonResponseToTableSchema(response, *datatable);
			}).wait();
		}

		// The image table supplies the OPTIONS request differently, hence why it'll be treated differently.
		m_schemas_[TableEntry::IMAGE] = DataTable({ "id", "title" });
	}
}
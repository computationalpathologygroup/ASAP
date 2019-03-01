#include "GrandChallengeDataAcquisition.h"

#include <algorithm>
#include <codecvt>
#include <stdexcept>
#include <locale>
#include <system_error>
#include <cstdio>

#include "../Misc/StringConversions.h"
#include "../Networking/HTTP_File_Download.h"
#include "../Serialization/JSON.h"

namespace ASAP::Data
{
	GrandChallengeDataAcquisition::GrandChallengeDataAcquisition(const GrandChallengeURLInfo uri_info, Misc::TemporaryDirectoryTracker& temp_dir, const Networking::Django_Connection::Credentials credentials, const web::http::client::http_client_config& config)
		: m_connection_(uri_info.base_url, Networking::Django_Connection::AuthenticationType::TOKEN, credentials, config), m_rest_uri_(uri_info), m_schemas_(4), m_temporary_directory_(temp_dir)
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

				DataTable worklists(*worklist_schema);
				try
				{
					web::http::http_response list_response(connection->SendRequest(list_request).get());
					receiver(worklists, Serialization::JSON::ParseJsonResponseToRecords(list_response, worklists));
				}
				catch (...)
				{
					receiver(worklists, -1);
				}
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
	}

	size_t GrandChallengeDataAcquisition::GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.study_addition << L"?patient=" << Misc::StringToWideString(patient_index);

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		DataTable* study_schema = &m_schemas_[TableEntry::STUDY];
		return m_connection_.QueueRequest(request, [receiver, study_schema](web::http::http_response& response)
		{
			// Parses the worklist sets into a data table.
			DataTable studies(*study_schema);
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, studies);
			receiver(studies, error_code);
		});
	}

	size_t GrandChallengeDataAcquisition::GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition;

		// TODO: clean this up.
		if (!worklist_index.empty() || !study_index.empty())
		{
			url << L"?";

			if (!study_index.empty())
			{
				url << L"study=" << Misc::StringToWideString(study_index);
			}
			if (!worklist_index.empty())
			{
				if (!study_index.empty())
				{
					url << L"&";
				}

				url << L"worklist=" << Misc::StringToWideString(worklist_index);
			}
		}

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection_.QueueRequest(request, [image_schema=&m_schemas_[TableEntry::IMAGE], receiver](web::http::http_response& response)
		{
		std::vector<std::vector<std::string>> image_records;				
			int error_code = Serialization::JSON::ParseJsonFieldsToVector(response, { "pk", "name" }, image_records);

			DataTable images(*image_schema);
			if (error_code == 0)
			{
				for (const std::vector<std::string>& record : image_records)
				{
					images.Insert(record);
				}
			}
			receiver(images, error_code);
		});
	}

	size_t GrandChallengeDataAcquisition::GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition << L"/" << Misc::StringToWideString(image_index) << L"/";

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		DataTable* image_schema = &m_schemas_[TableEntry::IMAGE];
		return m_connection_.QueueRequest(request, [image_schema, receiver, observer](web::http::http_response& response)
		{
			receiver(boost::filesystem::path());
			observer(100);
		});
	}

	size_t GrandChallengeDataAcquisition::GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer)
	{
		std::string sanitized_index(image_index);
		sanitized_index.erase(std::remove(sanitized_index.begin(), sanitized_index.end(), '"'), sanitized_index.end());

		std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition << Misc::StringToWideString(sanitized_index) << L"/";

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection_.QueueRequest(request, [receiver, observer, connection=&m_connection_, temp_dir=&m_temporary_directory_](web::http::http_response& response)
		{
			// Acquires the URL to the file.
			Networking::FileDownloadResults results;
			if (response.status_code() == web::http::status_codes::OK)
			{
				std::wstring file_uri = response.extract_json().get().at(L"files").as_array()[0].at(L"file").serialize();
				file_uri.erase(std::remove(file_uri.begin(), file_uri.end(), L'"'), file_uri.end());

				// Creates a request to acquire the file.
				web::http::http_request image_file_request(web::http::methods::GET);
				image_file_request.set_request_uri(file_uri);

				// Acquire the response or return an empty path if it fails.
				web::http::http_response image_file_response;
				try
				{
					image_file_response = connection->SendRequest(image_file_request).get();
				}
				catch (...)
				{
					receiver(boost::filesystem::path());
				}

				// TODO: implement method to reveal errors to user.

				// Download the image, link the status to the observer and return the results.
				results = Networking::HTTP_File_Download(image_file_response, temp_dir->GetAbsolutePath(), observer);
			}
			receiver(results.filepath);
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
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
		AcquireWorklistSet_();
	}

	WorklistDataAcquisitionInterface::SourceType GrandChallengeDataAcquisition::GetSourceType(void)
	{
		return WorklistDataAcquisitionInterface::SourceType::FULL_WORKLIST;
	}

	GrandChallengeURLInfo GrandChallengeDataAcquisition::GetStandardURI(const std::wstring base_url)
	{
		return { base_url, L"worklists/list/", L"worklists/set/", L"patients/patient/", L"studies/study/", L"api/v1/cases/images/" };
	}

	size_t GrandChallengeDataAcquisition::AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		std::wstringstream body;
		body << L"{ \"title\": \"" << Misc::StringToWideString(title) << "\", \"set\":\"" << m_worklist_set_id_ << "\", \"images\": [] }";

		web::http::http_request request(web::http::methods::POST);
		request.set_request_uri(L"/" + m_rest_uri_.worklist_addition);
		request.set_body(body.str(), L"application/json");

		return m_connection_.QueueRequest(request, [observer](web::http::http_response& response)
		{
			observer(response.status_code() == web::http::status_codes::Created);
		});
	}

	size_t GrandChallengeDataAcquisition::UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, const std::function<void(const bool)>& observer)
	{
		return 0;
	}

	size_t GrandChallengeDataAcquisition::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri_.worklist_addition);

		return m_connection_.QueueRequest(request, [receiver, worklist_schema=&m_schemas_[TableEntry::WORKLIST]](web::http::http_response& response)
		{
			// Parses the worklist sets into a data table.
			DataTable worklists(*worklist_schema);
			int error_code = Serialization::JSON::ParseJsonResponseToRecords(response, worklists);
			receiver(worklists, error_code);
		});
	}

	size_t GrandChallengeDataAcquisition::GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri_.patient_addition;
		
		if (!worklist_index.empty())
		{
			url << L"?worklist=" << Misc::StringToWideString(worklist_index);
		}

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection_.QueueRequest(request,[receiver, patient_schema=&m_schemas_[TableEntry::PATIENT]](web::http::http_response& response)
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

		return m_connection_.QueueRequest(request, [receiver, study_schema=&m_schemas_[TableEntry::STUDY]](web::http::http_response& response)
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
		/*std::wstringstream url;
		url << L"/" << m_rest_uri_.image_addition << L"/" << Misc::StringToWideString(image_index) << L"/";

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());
		*/
		receiver(boost::filesystem::path());
		observer(100);
		return 0;
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

	void GrandChallengeDataAcquisition::AcquireWorklistSet_(void)
	{
		web::http::http_request set_request(web::http::methods::GET);
		set_request.set_request_uri(L"/" + m_rest_uri_.worklist_set_addition);

		m_connection_.SendRequest(set_request).then([connection=&m_connection_, info=&m_rest_uri_, &set_id=m_worklist_set_id_](web::http::http_response& response)
		{
			// If there is a set, acquire worklists.
			web::json::value set_json(response.extract_json().get());
			if (set_json.size() == 0)
			{
				web::http::http_request set_creation(web::http::methods::POST);
				set_creation.set_request_uri(L"/" + info->worklist_set_addition);
				set_creation.set_body(L"{ \"title\": \"user worklists\" }", L"application/json");

				connection->SendRequest(set_creation).then([&set_id](web::http::http_response& response)
				{
					web::json::value set_json = response.extract_json().get();
					set_id = set_json.at(L"id").to_string();
				}).wait();
			}
			else
			{
				set_id = set_json.as_array()[0].at(L"id").to_string();
			}

			set_id.erase(std::remove(set_id.begin(), set_id.end(), '"'), set_id.end());
		}).wait();
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

		// Defines which fields the user should be able to see. Only required for Patient and Study records
		// TODO: Clean this up
		std::set<std::string> patient_headers(m_schemas_[TableEntry::PATIENT].GetColumnNames());
		for (const std::string& header : patient_headers)
		{
			if (header != "name")
			{
				m_schemas_[TableEntry::PATIENT].SetColumnAsInvisible(header);
			}
		}

		std::set<std::string> study_headers(m_schemas_[TableEntry::STUDY].GetColumnNames());
		for (const std::string& header : study_headers)
		{
			if (header != "name")
			{
				m_schemas_[TableEntry::STUDY].SetColumnAsInvisible(header);
			}
		}
	}
}
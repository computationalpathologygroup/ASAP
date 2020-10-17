#include "GrandChallengeSource.h"

#include <algorithm>
#include <stdexcept>
#include <system_error>
#include <cstdio>

#include "core/stringconversion.h"
#include "../Networking/HTTPFileDownload.h"
#include "../Serialization/JSON.h"

// TODO: Grand Challenge doesn't offer schema's for it's API yet, which is why it's being loaded
//		 through actual records currently. Once schema's are made available through the API, the
//		 refresh tables function should be refactored.

namespace ASAP
{
	GrandChallengeSource::GrandChallengeSource(const GrandChallengeURLInfo uri_info, TemporaryDirectoryTracker& temp_dir, const DjangoConnection::Credentials credentials, const web::http::client::http_client_config& config)
		: m_connection(uri_info.base_url, DjangoConnection::AuthenticationType::TOKEN, credentials, config), m_rest_uri(uri_info), m_schemas(4), m_temporary_directory(temp_dir)
	{
		refreshTables();
	}

	WorklistSourceInterface::SourceType GrandChallengeSource::getSourceType(void)
	{
		return WorklistSourceInterface::SourceType::FULL_WORKLIST;
	}

	GrandChallengeURLInfo GrandChallengeSource::getStandardURI(const std::wstring base_url)
	{
		return { base_url, L"api/v1/worklists/", L"api/v1/patients/", L"api/v1/studies/", L"api/v1/cases/images/" };
	}

	size_t GrandChallengeSource::addWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		std::wstringstream body;
		body << L"{ \"title\": \"" << core::stringToWideString(title) << "\", \"images\": [] }";

		web::http::http_request request(web::http::methods::POST);
		request.set_request_uri(L"/" + m_rest_uri.worklist_addition);
		request.set_body(body.str(), L"application/json");

		return m_connection.queueRequest(request, [observer](web::http::http_response& response)
		{
			observer(response.status_code() == web::http::status_codes::Created);
		});
	}

	size_t GrandChallengeSource::updateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer)
	{
		std::wstringstream body;
		body << L"{ \"title\": \"" << core::stringToWideString(title) << "\", \"images\": [ ";

		for (auto it = images.begin(); it != images.end(); ++it)
		{
			body << L"\"" << core::stringToWideString(*it) << L"\"";
			if (it != --images.end())
			{
				body << L",";
			}
		}
		body << L" ] }";

		std::wstringstream url;
		url << L"/" << m_rest_uri.worklist_addition << core::stringToWideString(worklist_index) << L"/";

		web::http::http_request request(web::http::methods::PATCH);
		request.set_request_uri(url.str());
		request.set_body(body.str(), L"application/json");

		return m_connection.queueRequest(request, [observer](web::http::http_response& response)
		{
			observer(response.status_code() == web::http::status_codes::OK);
		});
	}

	size_t GrandChallengeSource::deleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri.worklist_addition << core::stringToWideString(worklist_index) << L"/";

		web::http::http_request request(web::http::methods::DEL);
		request.set_request_uri(url.str());

		return m_connection.queueRequest(request, [observer](web::http::http_response& response)
		{
			observer(response.status_code() == web::http::status_codes::NoContent);
		});
	}

	size_t GrandChallengeSource::getWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri.worklist_addition);

		return m_connection.queueRequest(request, [this, receiver](web::http::http_response& response)
		{
			// Parses the worklists into a data table.
			DataTable worklists(this->m_schemas[TableEntry::WORKLIST]);
			int error_code = JSON::responseToTable(this->m_connection, response, worklists);
			receiver(worklists, error_code);
		});
	}

	size_t GrandChallengeSource::getPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri.patient_addition;
		
		if (!worklist_index.empty())
		{
			url << L"?worklist=" << core::stringToWideString(worklist_index);
		}

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection.queueRequest(request,[this, receiver](web::http::http_response& response)
		{
			DataTable& patient_schema = this->m_schemas[TableEntry::PATIENT];
			DataTable patients(this->m_schemas[TableEntry::PATIENT]);

			// Parses the patients into a data table.
			int error_code = JSON::responseToTable(this->m_connection, response, patients);

			// TODO: Remove once Grand Challenge supports schema's
			if (patients.size() > 0 && patient_schema.getColumnCount() == 0)
			{
				this->refreshTables();
			}
			
			receiver(patients, error_code);
		});
	}

	size_t GrandChallengeSource::getStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri.study_addition << L"?patient=" << core::stringToWideString(patient_index);

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection.queueRequest(request, [this, receiver](web::http::http_response& response)
		{
			DataTable& study_schema = this->m_schemas[TableEntry::STUDY];
			DataTable studies(this->m_schemas[TableEntry::STUDY]);

			// Parses the studies into a data table.
			int error_code = JSON::responseToTable(this->m_connection, response, studies);

			// TODO: Remove once Grand Challenge supports schema's
			if (studies.size() > 0 && study_schema.getColumnCount() == 0)
			{
				this->refreshTables();
			}

			receiver(studies, error_code);
		});
	}

	size_t GrandChallengeSource::getImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, int)>& receiver)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri.image_addition;

		// TODO: clean this up.
		if (!worklist_index.empty() || !study_index.empty())
		{
			url << L"?";

			if (!study_index.empty())
			{
				url << L"study=" << core::stringToWideString(study_index);
			}
			if (!worklist_index.empty())
			{
				if (!study_index.empty())
				{
					url << L"&";
				}

				url << L"worklist=" << core::stringToWideString(worklist_index);
			}
		}

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection.queueRequest(request, [this, receiver](web::http::http_response& response)
		{
			DataTable images(this->m_schemas[TableEntry::IMAGE]);
			int error_code = JSON::responseToFilteredTable(this->m_connection, response, images, std::vector<std::string>({ "pk", "name"}));
			receiver(images, error_code);
		});
	}

	size_t GrandChallengeSource::getImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
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

	size_t GrandChallengeSource::getImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		std::wstringstream url;
		url << L"/" << m_rest_uri.image_addition << core::stringToWideString(image_index) << L"/";

		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(url.str());

		return m_connection.queueRequest(request, [receiver, observer, connection=&m_connection, temp_dir=&m_temporary_directory](web::http::http_response& response)
		{
			// Acquires the URL to the file.
			if (response.status_code() == web::http::status_codes::OK)
			{
				web::json::value json(response.extract_json().get());

				std::wstring file_name	= json.at(L"name").serialize();
				std::wstring file_uri	= json.at(L"files").as_array()[0].at(L"file").serialize();
				file_name.erase(std::remove(file_name.begin(), file_name.end(), L'"'), file_name.end());
				file_uri.erase(std::remove(file_uri.begin(), file_uri.end(), L'"'), file_uri.end());

				// Creates a request to acquire the file.
				web::http::http_request image_file_request(web::http::methods::GET);
				image_file_request.set_request_uri(file_uri);

				// Acquire the image or return an empty path if it fails.
				try
				{
					web::http::http_response image_file_response = connection->sendRequest(image_file_request).get();

					// Download the image, link the status to the observer and return the results.
					receiver(httpFileDownload(image_file_response, temp_dir->getAbsolutePath(), core::wideStringToString(file_name), observer));
				}
				catch (const std::exception& e)
				{
					// TODO: implement method to reveal errors to user.
					receiver(boost::filesystem::path());
				}
			}
		});
	}

	std::set<std::string> GrandChallengeSource::getWorklistHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas[TableEntry::WORKLIST].getColumnNames(selection);
	}

	std::set<std::string> GrandChallengeSource::getPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas[TableEntry::PATIENT].getColumnNames(selection);
	}

	std::set<std::string> GrandChallengeSource::getStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas[TableEntry::STUDY].getColumnNames(selection);
	}

	std::set<std::string> GrandChallengeSource::getImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_schemas[TableEntry::IMAGE].getColumnNames(selection);
	}

	void GrandChallengeSource::cancelTask(size_t id)
	{
		m_connection.cancelTask(id);
	}

	void GrandChallengeSource::refreshTables(void)
	{
		// Acquires Patients schema.
		web::http::http_request request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri.patient_addition);

		m_connection.sendRequest(request).then([this](web::http::http_response& response)
		{
			JSON::responseToTable(this->m_connection, response, m_schemas[TableEntry::PATIENT]);
			m_schemas[TableEntry::PATIENT].clear();
		}).wait();

		// Acquires the Studies schema
		request = web::http::http_request(web::http::methods::GET);
		request.set_request_uri(L"/" + m_rest_uri.study_addition);

		m_connection.sendRequest(request).then([this](web::http::http_response& response)
		{
			JSON::responseToTable(this->m_connection, response, m_schemas[TableEntry::STUDY]);
			m_schemas[TableEntry::STUDY].clear();
		}).wait();

		// Acquires Worklists schema.
		request = web::http::http_request(web::http::methods::OPTIONS);
		request.set_request_uri(L"/" + m_rest_uri.worklist_addition);

		DataTable* datatable(&m_schemas[TableEntry::WORKLIST]);
		m_connection.sendRequest(request).then([datatable](web::http::http_response& response)
		{
			JSON::optionsResponseToTableSchema(response, *datatable);
		}).wait();

		// The image table supplies the OPTIONS request differently, hence why it'll be treated differently.
		m_schemas[TableEntry::IMAGE] = DataTable({ "id", "title" });

		// Defines which fields the user should be able to see. Only required for Patient and Study records
		// TODO: Clean this up
		std::set<std::string> patient_headers(m_schemas[TableEntry::PATIENT].getColumnNames());
		for (const std::string& header : patient_headers)
		{
			if (header != "name")
			{
				m_schemas[TableEntry::PATIENT].setColumnAsInvisible(header);
			}
		}

		std::set<std::string> study_headers(m_schemas[TableEntry::STUDY].getColumnNames());
		for (const std::string& header : study_headers)
		{
			if (header != "name")
			{
				m_schemas[TableEntry::STUDY].setColumnAsInvisible(header);
			}
		}
	}
}
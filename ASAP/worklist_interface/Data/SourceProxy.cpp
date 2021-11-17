#include "SourceProxy.h"

#include <stdexcept>
#include <sstream>

#include <filesystem>
#include "core/stringconversion.h"
#ifdef BUILD_GRANDCHALLENGE_INTERFACE
#include "GrandChallengeSource.h"
#endif 

#include "DirectorySource.h"
#include "FilelistSource.h"

namespace ASAP
{
	SourceProxy::SourceProxy(TemporaryDirectoryTracker& temp_dir) : m_source(nullptr), m_temporary_directory(temp_dir), m_max_number_previous_sources(5)
	{
	}

	void SourceProxy::close(void)
	{
		m_source.reset(nullptr);
	}

	void SourceProxy::loadSource(const std::string& source)
	{
		if (source.empty())
		{
			throw std::runtime_error("No source selected.");
		}

		std::pair<std::string, std::unordered_map<std::string, std::string>> deserialized_source(deserializeSource(source));
		std::string& source_path(deserialized_source.first);
		std::unordered_map<std::string, std::string>& parameters(deserialized_source.second);

		try
		{
			fs::path potential_system_path(source_path);
			if (fs::is_regular_file(potential_system_path) && checkParameters(parameters, FilelistSource::getRequiredParameterFields()))
			{
				m_source = std::unique_ptr<WorklistSourceInterface>(new FilelistSource(source_path));
			}
			else if (fs::is_directory(potential_system_path) && checkParameters(parameters, DirectorySource::getRequiredParameterFields()))
			{
				m_source = std::unique_ptr<DirectorySource>(new DirectorySource(source_path));
			}
#ifdef BUILD_GRANDCHALLENGE_INTERFACE
			else if (checkParameters(parameters, GrandChallengeSource::getRequiredParameterFields()))
			{
				web::http::client::http_client_config config;
				config.set_validate_certificates(!static_cast<bool>(parameters.find("ignore_certificate")->second[0]));

				GrandChallengeURLInfo uri_info = GrandChallengeSource::getStandardURI(core::stringToWideString(source_path));
				DjangoConnection::Credentials credentials(DjangoConnection::CreateCredentials(core::stringToWideString(parameters.find("token")->second), L""));
				m_source = std::unique_ptr<WorklistSourceInterface>(new GrandChallengeSource(uri_info, m_temporary_directory, credentials, config));
			}
#endif
			// Adds the new source to the previous sources.
			m_current_source = source;

			auto already_added(std::find(m_previous_sources.begin(), m_previous_sources.end(), m_current_source));
			if (already_added != m_previous_sources.end())
			{
				m_previous_sources.erase(already_added);
			}
			else if (m_previous_sources.size() == m_max_number_previous_sources)
			{
				m_previous_sources.pop_back();
			}
			m_previous_sources.push_front(m_current_source);
		}
		catch (const std::exception& e)
		{
			throw std::runtime_error("Unable to open source: " + source_path + "\n" + e.what());
		}
	}

	bool SourceProxy::isInitialized(void)
	{
		return m_source != nullptr;
	}

	const std::string& SourceProxy::getCurrentSource(void)
	{
		return m_current_source;
	}

	const std::deque<std::string>& SourceProxy::getPreviousSources(void)
	{
		return m_previous_sources;
	}

	void SourceProxy::setSourceInformation(const std::string& current_source, const std::vector<std::string>& previous_sources)
	{
		m_current_source = current_source;

		unsigned int total_previous_sources = this->m_max_number_previous_sources > previous_sources.size() ? previous_sources.size() : this->m_max_number_previous_sources;
		for (unsigned int i = 0; i < total_previous_sources; ++i) 
		{
			m_previous_sources.push_back(previous_sources[i]);
		}
	}

	std::string SourceProxy::serializeSource(const std::string& location, const std::unordered_map<std::string, std::string>& parameters)
	{
		std::stringstream serialized_source;
		serialized_source << location << "&";
		for (const auto& entry : parameters)
		{
			serialized_source << entry.first << "=" << entry.second << "|";
		}
		return serialized_source.str();
	}

	std::pair<std::string, std::unordered_map<std::string, std::string>> SourceProxy::deserializeSource(const std::string& source)
	{
		std::string location(source);
		std::unordered_map<std::string, std::string> parameters;

		std::vector<std::string> source_elements;
		core::split(source, source_elements, "&");
		if (source_elements.size() > 0)
		{
			location = source_elements[0];
			std::vector<std::string> keyValueList;
			core::split(source_elements[1], keyValueList, "|");
			if (!keyValueList.empty()) {
				for (size_t parameter = 0; parameter < keyValueList.size(); ++parameter)
				{
					std::vector<std::string> key_value;
					core::split(keyValueList[parameter], key_value, "=");
					if (key_value.size() > 1) {
						parameters.insert({ key_value[0], key_value[1] });
					}
				}
			}
		}

		return { location, parameters };
	}

	bool SourceProxy::checkParameters(const std::unordered_map<std::string, std::string> additional_params, const std::vector<std::string> required_params)
	{
		for (const std::string& param : required_params)
		{
			auto it = additional_params.find(param);
			if (it == additional_params.end())
			{
				return false;
			}
		}
		return true;
	}

	/// ########################################### Proxy Calls ########################################### ///
	void SourceProxy::cancelTask(size_t id)
	{
		m_source->cancelTask(id);
	}

	WorklistSourceInterface::SourceType SourceProxy::getSourceType(void)
	{
		return m_source->getSourceType();
	}

	size_t SourceProxy::addWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		return m_source->addWorklistRecord(title, observer);
	}

	size_t SourceProxy::updateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer)
	{
		return m_source->updateWorklistRecord(worklist_index, title, images, observer);
	}

	size_t SourceProxy::deleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer)
	{
		return m_source->deleteWorklistRecord(worklist_index, observer);
	}

	size_t SourceProxy::getWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source->getWorklistRecords(receiver);
	}

	size_t SourceProxy::getPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source->getPatientRecords(worklist_index, receiver);
	}

	size_t SourceProxy::getStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source->getStudyRecords(patient_index, receiver);
	}

	size_t SourceProxy::getImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source->getImageRecords(worklist_index, study_index, receiver);
	}

	size_t SourceProxy::getImageThumbnailFile(const std::string& image_index, const std::function<void(fs::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		return m_source->getImageThumbnailFile(image_index, receiver, observer);
	}

	size_t SourceProxy::getImageFile(const std::string& image_index, const std::function<void(fs::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		return m_source->getImageFile(image_index, receiver, observer);
	}

	std::set<std::string> SourceProxy::getWorklistHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source->getWorklistHeaders(selection);
	}

	std::set<std::string> SourceProxy::getPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source->getPatientHeaders(selection);
	}

	std::set<std::string> SourceProxy::getStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source->getStudyHeaders(selection);
	}

	std::set<std::string> SourceProxy::getImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source->getImageHeaders(selection);
	}
}
#include "SourceProxy.h"

#include <codecvt>
#include <stdexcept>
#include <sstream>

#include <boost/filesystem.hpp>

#ifdef BUILD_GRANDCHALLENGE_INTERFACE
#include "GrandChallengeSource.h"
#endif 

#include "DirectorySource.h"
#include "FilelistSource.h"
#include "../Misc/StringManipulation.h"
#include "../Misc/StringConversions.h"

namespace ASAP
{
	SourceProxy::SourceProxy(TemporaryDirectoryTracker& temp_dir) : m_source_(nullptr), m_temporary_directory_(temp_dir), m_number_previous_sources_(5)
	{
	}

	void SourceProxy::Close(void)
	{
		m_source_.reset(nullptr);
	}

	void SourceProxy::LoadSource(const std::string& source)
	{
		if (source.empty())
		{
			throw std::runtime_error("No source selected.");
		}

		std::pair<std::string, std::unordered_map<std::string, std::string>> deserialized_source(DeserializeSource(source));
		std::string& source_path(deserialized_source.first);
		std::unordered_map<std::string, std::string>& parameters(deserialized_source.second);

		try
		{
			boost::filesystem::path potential_system_path(source_path);
			if (boost::filesystem::is_regular_file(potential_system_path) && CheckParameters_(parameters, FilelistSource::GetRequiredParameterFields()))
			{
				m_source_ = std::unique_ptr<WorklistSourceInterface>(new FilelistSource(source_path));
			}
			else if (boost::filesystem::is_directory(potential_system_path) && CheckParameters_(parameters, DirectorySource::GetRequiredParameterFields()))
			{
				m_source_ = std::unique_ptr<DirectorySource>(new DirectorySource(source_path));
			}
#ifdef BUILD_GRANDCHALLENGE_INTERFACE
			else if (CheckParameters_(parameters, GrandChallengeSource::GetRequiredParameterFields()))
			{
				web::http::client::http_client_config config;
				config.set_validate_certificates(!static_cast<bool>(parameters.find("ignore_certificate")->second[0]));

				GrandChallengeURLInfo uri_info = GrandChallengeSource::GetStandardURI(Misc::StringToWideString(source_path));
				Django_Connection::Credentials credentials(Django_Connection::CreateCredentials(Misc::StringToWideString(parameters.find("token")->second), L""));
				m_source_ = std::unique_ptr<WorklistSourceInterface>(new GrandChallengeSource(uri_info, m_temporary_directory_, credentials, config));
			}
#endif
			// Adds the new source to the previous sources.
			m_current_source_ = source;

			auto already_added(std::find(m_previous_sources_.begin(), m_previous_sources_.end(), m_current_source_));
			if (already_added != m_previous_sources_.end())
			{
				m_previous_sources_.erase(already_added);
			}
			else if (m_previous_sources_.size() == m_number_previous_sources_)
			{
				m_previous_sources_.pop_back();
			}
			m_previous_sources_.push_front(m_current_source_);
		}
		catch (const std::exception& e)
		{
			throw std::runtime_error("Unable to open source: " + source_path + "\n" + e.what());
		}
	}

	bool SourceProxy::IsInitialized(void)
	{
		return m_source_ != nullptr;
	}

	const std::string& SourceProxy::GetCurrentSource(void)
	{
		return m_current_source_;
	}

	const std::deque<std::string>& SourceProxy::GetPreviousSources(void)
	{
		return m_previous_sources_;
	}

	void SourceProxy::SetSourceInformation(const std::string& current_source, const std::vector<std::string>& previous_sources)
	{
		m_current_source_ = current_source;

		unsigned int total_previous_sources = this->m_number_previous_sources_ > previous_sources.size() ? previous_sources.size() : this->m_number_previous_sources_;
		for (unsigned int i = 0; i < total_previous_sources; ++i) 
		{
			m_previous_sources_.push_back(previous_sources[i]);
		}
	}

	std::string SourceProxy::SerializeSource(const std::string& location, const std::unordered_map<std::string, std::string>& parameters)
	{
		std::stringstream serialized_source;
		serialized_source << location << "|";
		for (const auto& entry : parameters)
		{
			serialized_source << entry.first << "=" << entry.second << "|";
		}
		return serialized_source.str();
	}

	std::pair<std::string, std::unordered_map<std::string, std::string>> SourceProxy::DeserializeSource(const std::string& source)
	{
		std::string location(source);
		std::unordered_map<std::string, std::string> parameters;

		std::vector<std::string> source_elements(Misc::Split(source, '|'));
		if (source_elements.size() > 0)
		{
			location = source_elements[0];
			for (size_t element = 1; element < source_elements.size(); ++element)
			{
				std::vector<std::string> key_value(Misc::Split(source_elements[element], '='));
				parameters.insert({ key_value[0], key_value[1] });
			}
		}

		return { location, parameters };
	}

	bool SourceProxy::CheckParameters_(const std::unordered_map<std::string, std::string> additional_params, const std::vector<std::string> required_params)
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
	void SourceProxy::CancelTask(size_t id)
	{
		m_source_->CancelTask(id);
	}

	WorklistSourceInterface::SourceType SourceProxy::GetSourceType(void)
	{
		return m_source_->GetSourceType();
	}

	size_t SourceProxy::AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer)
	{
		return m_source_->AddWorklistRecord(title, observer);
	}

	size_t SourceProxy::UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer)
	{
		return m_source_->UpdateWorklistRecord(worklist_index, title, images, observer);
	}

	size_t SourceProxy::DeleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer)
	{
		return m_source_->DeleteWorklistRecord(worklist_index, observer);
	}

	size_t SourceProxy::GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source_->GetWorklistRecords(receiver);
	}

	size_t SourceProxy::GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source_->GetPatientRecords(worklist_index, receiver);
	}

	size_t SourceProxy::GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source_->GetStudyRecords(patient_index, receiver);
	}

	size_t SourceProxy::GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver)
	{
		return m_source_->GetImageRecords(worklist_index, study_index, receiver);
	}

	size_t SourceProxy::GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		return m_source_->GetImageThumbnailFile(image_index, receiver, observer);
	}

	size_t SourceProxy::GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer)
	{
		return m_source_->GetImageFile(image_index, receiver, observer);
	}

	std::set<std::string> SourceProxy::GetWorklistHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source_->GetWorklistHeaders(selection);
	}

	std::set<std::string> SourceProxy::GetPatientHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source_->GetPatientHeaders(selection);
	}

	std::set<std::string> SourceProxy::GetStudyHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source_->GetStudyHeaders(selection);
	}

	std::set<std::string> SourceProxy::GetImageHeaders(const DataTable::FIELD_SELECTION selection)
	{
		return m_source_->GetImageHeaders(selection);
	}
}
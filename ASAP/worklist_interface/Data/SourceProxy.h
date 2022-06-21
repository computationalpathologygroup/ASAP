#ifndef __ASAP_DATA_SOURCEPROXY__
#define __ASAP_DATA_SOURCEPROXY__

#include <memory>
#include <string>
#include <unordered_map>
#include <deque>

#include "WorklistSourceInterface.h"
#include "../Misc/TemporaryDirectoryTracker.h"

namespace ASAP
{
	class SourceProxy : public WorklistSourceInterface
	{
		public:
			SourceProxy(TemporaryDirectoryTracker& temp_dir);

			void close(void);
			void loadSource(const std::string& source);
			bool isInitialized(void);

			const std::string& getCurrentSource(void);
			const std::deque<std::string>& getPreviousSources(void);

			void setSourceInformation(const std::string& current_source, const std::vector<std::string>& previous_sources);

			static std::string serializeSource(const std::string& location, const std::unordered_map<std::string, std::string>& parameters);
			static std::pair<std::string, std::unordered_map<std::string, std::string>> deserializeSource(const std::string& source);

			/// Proxy Calls ///
			void cancelTask(size_t id);
			WorklistSourceInterface::SourceType getSourceType(void);

			size_t addWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer);
			size_t updateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer);
			size_t deleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer);

			size_t getWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t getPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t getStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t getImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver);

			size_t getImageThumbnailFile(const std::string& image_index, const std::function<void(fs::path)>& receiver, const std::function<void(uint8_t)>& observer);
			size_t getImageFile(const std::string& image_index, const std::function<void(fs::path)>& receiver, const std::function<void(uint8_t)>& observer);

			std::set<std::string> getWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> getPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> getStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> getImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);

		private:
			TemporaryDirectoryTracker&					m_temporary_directory;
			std::unique_ptr<WorklistSourceInterface>	m_source;

			std::string				m_current_source;
			std::deque<std::string>	m_previous_sources;
			unsigned int			m_max_number_previous_sources;

			bool checkParameters(const std::unordered_map<std::string, std::string> additional_params, const std::vector<std::string> required_params);
	};
}
#endif // __ASAP_DATA_SOURCEPROXY__
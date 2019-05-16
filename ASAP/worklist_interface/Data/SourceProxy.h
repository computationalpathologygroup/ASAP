#ifndef __ASAP_DATA_SOURCEPROXY__
#define __ASAP_DATA_SOURCEPROXY__

#include <memory>
#include <string>
#include <unordered_map>

#include "WorklistSourceInterface.h"
#include "../Misc/TemporaryDirectoryTracker.h"

namespace ASAP::Data
{
	class SourceProxy : public WorklistSourceInterface
	{
		public:
			SourceProxy(Misc::TemporaryDirectoryTracker& temp_dir);

			void Close(void);
			void LoadSource(const std::string& source);
			bool IsInitialized(void);

			const std::string& GetCurrentSource(void);
			const std::deque<std::string>& GetPreviousSources(void);

			void SetSourceInformation(const std::string& current_source, const std::vector<std::string>& previous_sources);

			static std::string SerializeSource(const std::string& location, const std::unordered_map<std::string, std::string>& parameters);
			static std::pair<std::string, std::unordered_map<std::string, std::string>> DeserializeSource(const std::string& source);

			/// Proxy Calls ///
			void CancelTask(size_t id);
			WorklistSourceInterface::SourceType GetSourceType(void);

			size_t AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer);
			size_t UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, const std::function<void(const bool)>& observer);
			size_t DeleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer);

			size_t GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver);

			size_t GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer);
			size_t GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer);

			std::set<std::string> GetWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);

		private:
			Misc::TemporaryDirectoryTracker&				m_temporary_directory_;
			std::unique_ptr<Data::WorklistSourceInterface>	m_source_;

			std::string				m_current_source_;
			std::deque<std::string>	m_previous_sources_;

			bool CheckParameters_(const std::unordered_map<std::string, std::string> additional_params, const std::vector<std::string> required_params);
	};
}
#endif // __ASAP_DATA_SOURCEPROXY__
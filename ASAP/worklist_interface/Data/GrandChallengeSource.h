#ifndef __ASAP_DATA_GRANDCHALLENGESOURCE__
#define __ASAP_DATA_GRANDCHALLENGESOURCE__

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataTable.h"
#include "WorklistSourceInterface.h"
#include "../Networking/DjangoConnection.h"
#include "../Misc/TemporaryDirectoryTracker.h"

namespace ASAP
{
	struct GrandChallengeURLInfo
	{
		std::wstring base_url;
		std::wstring worklist_addition;
		std::wstring patient_addition;
		std::wstring study_addition;
		std::wstring image_addition;
	};

	class GrandChallengeSource : public WorklistSourceInterface
	{
		public:
			GrandChallengeSource(const GrandChallengeURLInfo uri_info, TemporaryDirectoryTracker& temp_dir, const DjangoConnection::Credentials credentials, const web::http::client::http_client_config& config = web::http::client::http_client_config());

			static GrandChallengeURLInfo getStandardURI(const std::wstring base_url);
			WorklistSourceInterface::SourceType getSourceType(void);

			size_t addWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer);
			size_t updateWorklistRecord(const std::string& worklist_index, const std::string title, const std::set<std::string> images, const std::function<void(const bool)>& observer);
			size_t deleteWorklistRecord(const std::string& worklist_index, const std::function<void(const bool)>& observer);

			size_t getWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t getPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t getStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t getImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver);

			size_t getImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer);
			size_t getImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)>& observer);

			std::set<std::string> getWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> getPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> getStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> getImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);

			void cancelTask(size_t id);

		private:
			enum TableEntry { WORKLIST, PATIENT, STUDY, IMAGE };

			DjangoConnection			m_connection;
			GrandChallengeURLInfo		m_rest_uri;
			std::vector<DataTable>		m_schemas;
			TemporaryDirectoryTracker&	m_temporary_directory;

			void refreshTables(void);
	};
}
#endif // __ASAP_DATA_GRANDCHALLENGESOURCE__
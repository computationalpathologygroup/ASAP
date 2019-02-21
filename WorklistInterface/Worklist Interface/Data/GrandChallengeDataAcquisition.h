#ifndef __ASAP_DATA_GRANDCHALLENGEDATAACQUISITION__
#define __ASAP_DATA_GRANDCHALLENGEDATAACQUISITION__

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataTable.h"
#include "WorklistDataAcquisitionInterface.h"
#include "../Networking/Django_Connection.h"

namespace ASAP::Data
{
	struct GrandChallengeURLInfo
	{
		std::wstring base_url;
		std::wstring worklist_addition;
		std::wstring worklist_set_addition;
		std::wstring patient_addition;
		std::wstring study_addition;
		std::wstring image_addition;
		std::wstring image_file_addition;
	};

	class GrandChallengeDataAcquisition : public WorklistDataAcquisitionInterface
	{
		public:
			GrandChallengeDataAcquisition(const GrandChallengeURLInfo uri_info, const Networking::Django_Connection::Credentials credentials);

			static GrandChallengeURLInfo GetStandardURI(const std::wstring base_url);
			WorklistDataAcquisitionInterface::SourceType GetSourceType(void);

			size_t AddWorklistRecord(const std::string& title, std::function<void(const bool)>& observer);
			size_t UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, std::function<void(const bool)>& observer);

			size_t GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t GetPatientRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetImageRecords(const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver);

			size_t GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver);
			size_t GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver);

			std::set<std::string> GetWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);

			void CancelTask(size_t id);

		private:
			enum TableEntry { WORKLIST, WORKLIST_SET, PATIENT, STUDY, IMAGE, IMAGE_FILE };

			Networking::Django_Connection	m_connection_;
			GrandChallengeURLInfo			m_rest_uri_;
			std::vector<DataTable>			m_schemas_;

			void InitializeTables_(void);
	};
}
#endif // __ASAP_DATA_GRANDCHALLENGEDATAACQUISITION__
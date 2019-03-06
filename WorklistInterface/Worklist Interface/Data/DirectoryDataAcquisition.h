#ifndef __ASAP_DATA_DIRECTORYDATAACQUISITION__ 
#define __ASAP_DATA_DIRECTORYDATAACQUISITION__

#include "WorklistDataAcquisitionInterface.h"

namespace ASAP::Data
{
	class DirectoryDataAcquisition : public WorklistDataAcquisitionInterface
	{
		public:
			DirectoryDataAcquisition(const std::string directory_path);

			WorklistDataAcquisitionInterface::SourceType GetSourceType(void);

			size_t AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer);
			size_t UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, const std::function<void(const bool)>& observer);

			size_t GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver);

			size_t GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer);
			size_t GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer);

			std::set<std::string> GetWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::set<std::string> GetImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);

		private:
			DataTable m_images_;

			DataTable GetImageFilelist_(const std::string directory_path);
	};
}
#endif // __ASAP_DATA_DIRECTORYDATAACQUISITION__
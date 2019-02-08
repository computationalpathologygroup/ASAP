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

			size_t GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t GetPatientRecords(const size_t worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetStudyRecords(const size_t patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetImageRecords(const size_t study_index, const std::function<void(DataTable&, const int)>& receiver);

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
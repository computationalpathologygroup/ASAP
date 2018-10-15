#ifndef __DIRECTORY_DATA_AQUISITION_H__
#define __DIRECTORY_DATA_AQUISITION_H__

#include <string>

#include "WorklistDataAcquisitionInterface.h"

namespace ASAP::Worklist::Data
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

			std::vector<std::string> GetVisiblePatientHeaders(void);
			std::vector<std::string> GetVisibleStudyHeaders(void);
			std::vector<std::string> GetVisibleImageHeaders(void);

		private:
			DataTable m_images_;

			DataTable GetImageFilelist_(const std::string directory_path);
	};
}
#endif // __DIRECTORY_DATA_AQUISITION_H__
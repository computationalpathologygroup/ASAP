#ifndef __ASAP_DATA_FILELISTSOURCE__
#define __ASAP_DATA_FILELISTSOURCE__

#include <string>

#include "WorklistSourceInterface.h"

namespace ASAP
{
	class FilelistSource : public WorklistSourceInterface
	{
		public:
			FilelistSource(const std::string filepath);

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
			DataTable m_images;

			DataTable getImageFilelist(const std::string filepath);
	};
}
#endif // __ASAP_DATA_FILELISTSOURCE__
#ifndef __ASAP_DATA_DJANGODATAAQUISITION__
#define __ASAP_DATA_DJANGODATAAQUISITION__

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataTable.h"
#include "WorklistDataAcquisitionInterface.h"
#include "../IO/HTTP_Connection.h"

namespace ASAP::Worklist::Data
{
	struct DjangoRestURI
	{
		std::wstring base_url;
		std::wstring worklist_addition;
		std::wstring patient_addition;
		std::wstring study_addition;
		std::wstring image_addition;
		std::wstring worklist_patient_relation_addition;
	};

	class DjangoDataAcquisition : public WorklistDataAcquisitionInterface
	{
		public:
			DjangoDataAcquisition(const DjangoRestURI uri_info);

			static DjangoRestURI GetStandardURI(void);
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
			enum TableEntry { WORKLIST, PATIENT, STUDY, IMAGE, WORKLIST_PATIENT_RELATION };

			IO::HTTP_Connection		m_connection_;
			DjangoRestURI			m_rest_uri_;
			std::vector<DataTable>	m_tables_;

			void InitializeTables_(void);
	};
}
#endif // __ASAP_DATA_DJANGODATAAQUISITION__
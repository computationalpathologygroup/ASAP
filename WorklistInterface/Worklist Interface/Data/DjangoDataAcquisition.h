#ifndef __DJANGO_DATA_AQUISITION_H__
#define __DJANGO_DATA_AQUISITION_H__

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "WorklistDataAcquisitionInterface.h"
#include "DataTable.h"

#include <cpprest/http_client.h>

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
			~DjangoDataAcquisition(void);

			static DjangoRestURI GetStandardURI(void);
			WorklistDataAcquisitionInterface::SourceType GetSourceType(void);

			size_t GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver);
			size_t GetPatientRecords(const size_t worklist_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetStudyRecords(const size_t patient_index, const std::function<void(DataTable&, const int)>& receiver);
			size_t GetImageRecords(const size_t study_index, const std::function<void(DataTable&, const int)>& receiver);

			std::unordered_set<std::string> GetWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::unordered_set<std::string> GetPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::unordered_set<std::string> GetStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);
			std::unordered_set<std::string> GetImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL);

			void CancelTask(const size_t task_id);

		private:
			struct TokenTaskPair
			{
				pplx::task<void>						task;
				concurrency::cancellation_token_source	token;
			};

			enum TableEntry { WORKLIST, PATIENT, STUDY, IMAGE, WORKLIST_PATIENT_RELATION };

			web::http::client::http_client				m_client_;
			DjangoRestURI								m_rest_uri_;
			size_t										m_token_counter_;
			std::vector<DataTable>						m_tables_;
			std::unordered_map<size_t, TokenTaskPair>	m_active_tasks_;

			void InitializeTables_(void);
			size_t ProcessRequest_(const web::http::http_request& request, std::function<void(web::http::http_response&)> observer);
	};
}
#endif // __DJANGO_DATA_AQUISITION_H__
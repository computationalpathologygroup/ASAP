#ifndef WORKLIST_CSV_EXAMPLE_H
#define WORKLIST_CSV_EXAMPLE_H

#include <string>
#include <unordered_map>
#include <vector>

#include "WorklistDataAcquisitionInterface.h"
#include "DataTable.h"

class Worklist_CSV_Example : public AbstractWorklistDataAcquisition
{
    public:
        Worklist_CSV_Example(const std::vector<std::string> paths);

        DataTable GetWorklistRecords(void);
        DataTable GetPatientRecords(const size_t worklist_index);
        DataTable GetStudyRecords(const size_t patient_index);
        DataTable GetImageRecords(const size_t study_index);

        std::vector<std::string> GetPatientHeaders(void);
        std::vector<std::string> GetStudyHeaders(void);
        std::vector<std::string> GetImageHeaders(void);

    private:
        std::vector<DataTable> m_tables_;

        std::unordered_map<size_t, std::vector<std::string>> m_worklist_items_;
        std::unordered_map<size_t, std::vector<std::string>> m_patient_items_;
        std::unordered_map<size_t, std::vector<std::string>> m_study_items_;
        std::unordered_map<size_t, std::vector<std::string>> m_image_items_;

        void Parse_(const std::string& location,
                   std::vector<std::vector<std::string>>& headers,
                   std::vector<std::vector<bool>>& is_meta,
                   std::unordered_map<size_t, std::vector<std::string>>& items);
        std::vector<bool> ParseBools_(std::vector<std::string> values);

        std::vector<std::string> ParseLine_(const std::string& source);
};

#endif // WORKLIST_CSV_EXAMPLE_H

 #ifndef ABSTRACTWORKLISTDATAACQUISITION_H
#define ABSTRACTWORKLISTDATAACQUISITION_H

#include "datatable.h"

class AbstractWorklistDataAcquisition
{
    public:
        virtual DataTable GetWorklistRecords(void) = 0;
        virtual DataTable GetPatientRecords(const size_t worklist_index) = 0;
        virtual DataTable GetStudyRecords(const size_t patient_index) = 0;
        virtual DataTable GetImageRecords(const size_t study_index) = 0;

        virtual std::vector<std::string> GetPatientHeaders(void) = 0;
        virtual std::vector<std::string> GetStudyHeaders(void) = 0;
        virtual std::vector<std::string> GetImageHeaders(void) = 0;
};
#endif // ABSTRACTWORKLISTDATAACQUISITION_H

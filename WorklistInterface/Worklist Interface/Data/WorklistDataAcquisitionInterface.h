 #ifndef __WORKLIST_DATA_ACQUISITION_INTERFACE_H__
#define __WORKLIST_DATA_ACQUISITION_INTERFACE_H__

#include "datatable.h"
#include <functional>
namespace ASAP::Worklist::Data
{
	/// <summary>
	/// Provides a basic interface that all Worklist GUI operations are based upon. Because it assumes
	/// a asynchronous environment, the acquirement of actual DataTables is done through lambda
	/// functions, while the called method itself only returns a task or token id that can be used to
	/// cancel the methods execution.
	///
	/// If a source isn't required to deal with asynchronous actions, it can simply ignore the
	/// implementation of the CancelTask method.
	/// </summary>
	class WorklistDataAcquisitionInterface
	{
	public:
		/// <summary>
		/// Describes the amount of information the source can provide.
		/// </summary>
		enum SourceType { FILELIST, FULL_WORKLIST };

		/// <summary>
		/// Returns a SourceType that describes the amount of information this source can provide.
		/// </summary>
		/// <return>The SourceType of this source.</return>
		virtual SourceType GetSourceType(void) = 0;

		/// <summary>
		/// Acquires the worklist records in a asynchronous manner, offering them to the receiver lambda.
		/// </summary>
		/// <param name="receiver">A lamba that accepts a DataTable, which holds the requested items and an integer that describes potential errors.</param>
		/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
		virtual size_t GetWorklistRecords(const std::function<void(DataTable&, const int)>& receiver) = 0;
		/// <summary>
		/// Acquires the patient records in a asynchronous manner, offering them to the receiver lambda.
		/// </summary>
		/// <param name="worklist_index">The id from the selected worklist.</param>
		/// <param name="receiver">A lamba that accepts a DataTable, which holds the requested items and an integer that describes potential errors.</param>
		/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
		virtual size_t GetPatientRecords(const size_t worklist_index, const std::function<void(DataTable&, const int)>& receiver) = 0;
		/// <summary>
		/// Acquires the study records in a asynchronous manner, offering them to the receiver lambda.
		/// </summary>
		/// <param name="patient_index">The id from the selected patient.</param>
		/// <param name="receiver">A lamba that accepts a DataTable, which holds the requested items and an integer that describes potential errors.</param>
		/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
		virtual size_t GetStudyRecords(const size_t patient_index, const std::function<void(DataTable&, const int)>& receiver) = 0;
		/// <summary>
		/// Acquires the image records in a asynchronous manner, offering them to the receiver lambda.
		/// </summary>
		/// <param name="study_index">The id from the selected study.</param>
		/// <param name="receiver">A lamba that accepts a DataTable, which holds the requested items and an integer that describes potential errors.</param>
		/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
		virtual size_t GetImageRecords(const size_t study_index, const std::function<void(DataTable&, const int)>& receiver) = 0;

		/// <summary>
		/// Returns the headers for the Patient table.
		/// </summary>
		/// <return>A vector with the headers for the patient records.</return>
		virtual std::vector<std::string> GetVisiblePatientHeaders(void) = 0;
		/// <summary>
		/// Returns the headers for the Study table.
		/// </summary>
		/// <return>A vector with the headers for the study records.</return>
		virtual std::vector<std::string> GetVisibleStudyHeaders(void) = 0;
		/// <summary>
		/// Returns the headers for the Image table.
		/// </summary>
		/// <return>A vector with the headers for the study records.</return>
		virtual std::vector<std::string> GetVisibleImageHeaders(void) = 0;

		/// <summary>
		/// Cancels the asynchronous task if it hasn't finished yet.
		/// </summary>
		/// <param name="id">The id of the task to cancel.</param>
		void CancelTask(size_t id);
	};
}
#endif // __WORKLIST_DATA_ACQUISITION_INTERFACE_H__
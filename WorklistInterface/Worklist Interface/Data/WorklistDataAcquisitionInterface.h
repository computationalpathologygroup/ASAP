#ifndef __ASAP_DATA_WORKLISTDATAACQUISITIONINTERFACE__
#define __ASAP_DATA_WORKLISTDATAACQUISITIONINTERFACE__

#include <functional>

#include <boost/filesystem.hpp>

#include "datatable.h"

namespace ASAP::Data
{
	/// <summary>
	/// Provides a basic interface that all Worklist GUI operations are based upon. Because t assumes
	/// a asynchronous environment, the acquirement of actual DataTables is done through lambda
	/// functions, while the called method itself only returns a task or token id that can be used to
	/// cancel the methods execution.
	///
	/// The DataTable indices are used to reference records for specifics tables, which requires
	/// implementations to translate these to potential id's.
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
			/// Returns a list of parameters required to access this source that can't be covered by the location alone.
			/// </summary>
			/// <returns>Returns a vector of strings that define the required parameter fields.</returns>
			static std::vector<std::string> GetRequiredParameterFields(void);

			/// <summary>
			/// Returns a SourceType that describes the amount of information this source can provide.
			/// </summary>
			/// <return>The SourceType of this source.</return>
			virtual SourceType GetSourceType(void) = 0;

			/// <summary>
			/// Adds a new worklist.
			/// </summary>
			/// <param name="title">The title for the new worklist. Must be unique.</param>
			/// <param name="observer">A lamba that accepts a boolean which details whether or not the task was succesful.</param>
			/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
			virtual size_t AddWorklistRecord(const std::string& title, const std::function<void(const bool)>& observer) = 0;

			/// <summary>
			/// Updates a worklist with a new name or list of images.
			/// </summary>
			/// <param name="worklist_index">A string containing the id of the worklist to update.</param>
			/// <param name="title">The (new) title for the worklist. Must be unique.</param>
			/// <param name="images">A vector containing the ids of the images the worklist should list.</param>
			/// <param name="observer">A lamba that accepts a boolean which details whether or not the task was succesful.</param>
			/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
			virtual size_t UpdateWorklistRecord(const std::string& worklist_index, const std::string title, const std::vector<std::string> images, const std::function<void(const bool)>& observer) = 0;

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
			virtual size_t GetPatientRecords(const std::string& worklist_index, const std::function<void(DataTable&, const int)>& receiver) = 0;
			/// <summary>
			/// Acquires the study records in a asynchronous manner, offering them to the receiver lambda.
			/// </summary>
			/// <param name="patient_index">The id from the selected patient.</param>
			/// <param name="receiver">A lamba that accepts a DataTable, which holds the requested items and an integer that describes potential errors.</param>
			/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
			virtual size_t GetStudyRecords(const std::string& patient_index, const std::function<void(DataTable&, const int)>& receiver) = 0;
			/// <summary>
			/// Acquires the image records in a asynchronous manner, offering them to the receiver lambda.
			/// </summary>
			/// <param name="worklist_index">The id from the overlaying worklist.</param>
			/// <param name="study_index">The id from the selected study.</param>
			/// <param name="receiver">A lamba that accepts a DataTable, which holds the requested items and an integer that describes potential errors.</param>
			/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
			virtual size_t GetImageRecords(const std::string& worklist_index, const std::string& study_index, const std::function<void(DataTable&, const int)>& receiver) = 0;

			/// <summary>
			/// Offers a thumbnail image, or an image that can be utilized to generate one through a filepath.
			/// </summary>
			/// <param name="image_index">The id of the image to acquire the thumbnail for.</param>
			/// <param name="receiver">A lamba that accepts a filepath pointing towards the thumbnail.</param>
			/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
			virtual size_t GetImageThumbnailFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer) = 0;
			/// <summary>
			/// Offers an image through a filepath.
			/// </summary>
			/// <param name="image_index">The id of the image to acquire.</param>
			/// <param name="receiver">A lamba that accepts a filepath pointing towards the image.</param>
			/// <return>The task id, which can be used to cancel asynchronous tasks.</return>
			virtual size_t GetImageFile(const std::string& image_index, const std::function<void(boost::filesystem::path)>& receiver, const std::function<void(uint8_t)> observer) = 0;

			/// <summary>
			/// Returns the headers for the Worklist table.
			/// </summary>
			/// <param name="selection">The selection criteria for which header fields to return.</param>
			/// <return>A vector with the headers for the worklist records.</return>
			virtual std::set<std::string> GetWorklistHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL) = 0;
			/// <summary>
			/// Returns the headers for the Patient table.
			/// </summary>
			/// <param name="selection">The selection criteria for which header fields to return.</param>
			/// <return>A vector with the headers for the patient records.</return>
			virtual std::set<std::string> GetPatientHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL) = 0;
			/// <summary>
			/// Returns the headers for the Study table.
			/// </summary>
			/// <param name="selection">The selection criteria for which header fields to return.</param>
			/// <return>A vector with the headers for the study records.</return>
			virtual std::set<std::string> GetStudyHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL) = 0;
			/// <summary>
			/// Returns the headers for the Image table.
			/// </summary>
			/// <param name="selection">The selection criteria for which header fields to return.</param>
			/// <return>A vector with the headers for the study records.</return>
			virtual std::set<std::string> GetImageHeaders(const DataTable::FIELD_SELECTION selection = DataTable::FIELD_SELECTION::ALL) = 0;

			/// <summary>
			/// Cancels the asynchronous task if it hasn't finished yet.
			/// </summary>
			/// <param name="id">The id of the task to cancel.</param>
			void CancelTask(size_t id);
	};
}
#endif // __ASAP_DATA_WORKLISTDATAACQUISITIONINTERFACE__
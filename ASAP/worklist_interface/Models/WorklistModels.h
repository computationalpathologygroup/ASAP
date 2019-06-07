#ifndef __ASAP_MODELS_WORKLISTMODELS__
#define __ASAP_MODELS_WORKLISTMODELS__

#include <set>
#include <QAbstractItemView>
#include <QStandardItemModel>

#include "../Data/DataTable.h"

namespace ASAP
{
	// Forward declaration
	class WorklistWindow;

	class WorklistModels
	{
		public:
			WorklistModels(void);

			QStandardItemModel* images;
			QStandardItemModel* patients;
			QStandardItemModel* studies;
			QStandardItemModel* worklists;

			void SetWorklistItems(const DataTable& items);
			void SetPatientsItems(const DataTable& items);
			void SetStudyItems(const DataTable& items);
			void SetImageItems(const DataTable& items, WorklistWindow* window, bool& continue_loading);

			void UpdateHeaders(std::vector<std::pair<std::set<std::string>, QAbstractItemView*>>& header_view_couple);

		private:
			void SetHeaders_(const std::set<std::string> headers, QStandardItemModel* model, QAbstractItemView* view);
	};
}
#endif // __ASAP_MODELS_WORKLISTMODELS__
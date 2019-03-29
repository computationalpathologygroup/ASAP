#ifndef __ASAP_MODELS_WORKLISTMODELS__
#define __ASAP_MODELS_WORKLISTMODELS__

#include <set>
#include <QStandardItemModel>

#include "Data/DataTable.h"
#include "../GUI/WorklistWindow.h"

namespace ASAP::Models
{
	class WorklistModels
	{
		Q_OBJECT

		public:
			WorklistModels(void);

			QStandardItemModel* images;
			QStandardItemModel* patients;
			QStandardItemModel* studies;
			QStandardItemModel* worklists;

			void SetWorklistItems(const Data::DataTable& items);
			void SetPatientsItems(const Data::DataTable& items);
			void SetStudyItems(const Data::DataTable& items);
			void SetImageItems(const Data::DataTable& items, GUI::WorklistWindow* window, bool& continue_loading);

			void UpdateHeaders(std::vector<std::pair<std::set<std::string>, QAbstractItemView*>>& header_view_couple);

		private:
			void SetHeaders_(const std::set<std::string> headers, QStandardItemModel* model, QAbstractItemView* view);
	};
}
#endif // __ASAP_MODELS_WORKLISTMODELS__
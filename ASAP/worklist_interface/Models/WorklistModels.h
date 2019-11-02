#ifndef __ASAP_MODELS_WORKLISTMODELS__
#define __ASAP_MODELS_WORKLISTMODELS__

#include <set>
#include <QAbstractItemView>
#include <QStandardItemModel>
#include "../Data/DataTable.h"
#include "../GUI/IconCreator.h"
#include <QtConcurrent/qtconcurrentmap.h>
#include <QFutureWatcher>

namespace ASAP
{
	// Forward declaration
	class WorklistWindow;

	class WorklistModels
	{
		public:
			WorklistModels(void);

			enum ModelEnum { WORKLISTS, PATIENTS, STUDIES, IMAGES };

			QStandardItemModel* images;
			QStandardItemModel* patients;
			QStandardItemModel* studies;
			QStandardItemModel* worklists;

			QStandardItemModel* GetModel(const ModelEnum model);
			ModelEnum GetModelEnum(QStandardItemModel* model);
			void SetWorklistItems(const DataTable& items);
			void SetPatientsItems(const DataTable& items);
			void SetStudyItems(const DataTable& items);
			std::unique_ptr<QFutureWatcher<void> > SetImageItems(const DataTable& items, WorklistWindow* window);

			void UpdateHeaders(std::vector<std::pair<std::set<std::string>, QAbstractItemView*>>& header_view_couple);

		private:
			void SetHeaders_(const std::set<std::string> headers, QStandardItemModel* model, QAbstractItemView* view);
			IconCreator m_creator;
	};
}
#endif // __ASAP_MODELS_WORKLISTMODELS__
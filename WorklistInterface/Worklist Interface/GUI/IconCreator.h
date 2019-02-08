#ifndef __ASAP_GUI_ICONCREATOR__
#define __ASAP_GUI_ICONCREATOR__

#include <mutex>
#include <string>
#include <vector>

#include <QObject>
#include <QStandardItemModel>
#include <qlistview.h>
#include <qstatusbar.h>

#include <Data/DataTable.h>

namespace ASAP::GUI
{
	/// <summary>
	/// Creates thumbnail icons based on WSIs readable by the multiresolution reader.
	/// </summary>
	class IconCreator : public QObject
	{
		Q_OBJECT

		public:
			IconCreator(void);
	
			void InsertIcons(const Data::DataTable& image_items, QStandardItemModel* image_model, const size_t size, bool& stop_loading);
			QIcon CreateIcon_(const std::string& filepath, const size_t size);
		
		private:
			QIcon CreateBlankIcon_(const size_t size);
			QIcon CreateInvalidIcon_(const size_t size);

		signals:
			void RequiresStatusBarChange(const QString& message);
			void RequiresItemRefresh(void);
	};
}
#endif // __ASAP_GUI_ICONCREATOR__
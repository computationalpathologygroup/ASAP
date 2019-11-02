#ifndef __ASAP_GUI_ICONCREATOR__
#define __ASAP_GUI_ICONCREATOR__

#include <mutex>
#include <string>
#include <vector>

#include <QObject>
#include <QStandardItemModel>

namespace ASAP
{
	/// <summary>
	/// Creates thumbnail icons based on WSIs readable by the multiresolution reader.
	/// </summary>
	class IconCreator : public QObject
	{
		Q_OBJECT

		public:
			IconCreator(void);
	
			bool InsertIcon(const std::pair<int, std::string>& index_location);
			QIcon CreateIcon_(const std::string& filepath, const size_t size);
		
		private:
			QIcon CreateBlankIcon_();
			QIcon CreateInvalidIcon_();
			QIcon m_placeholder_icon;
			QIcon m_invalid_icon;
			static int m_icon_size;


		signals:
			void RequiresStatusBarChange(const QString& message);
			void RequiresItemRefresh(int, const QIcon&);
	};
}
#endif // __ASAP_GUI_ICONCREATOR__
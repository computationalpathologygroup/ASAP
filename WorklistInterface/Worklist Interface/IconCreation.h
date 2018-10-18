#pragma once
#include <mutex>
#include <string>
#include <vector>

#include <QObject>
#include <QStandardItemModel>
#include <qlistview.h>
#include <qstatusbar.h>

#include <Data/DataTable.h>

namespace ASAP::Worklist::GUI
{
	/// <summary>
	/// Creates thumbnail icons based on WSIs readable by the multiresolution reader.
	/// </summary>
	class IconCreator : public QObject
	{
		Q_OBJECT

		public:
			IconCreator(void);
			~IconCreator(void);
	
			void InsertIcons(const DataTable& image_items, QStandardItemModel* image_model, QStatusBar* status_bar, const size_t size);

		private:
			bool		m_message_shown_;
			std::mutex	m_next_message_access_;
			QString		m_next_message_;
			QStatusBar*	m_status_bar_;

			QIcon CreateIcon_(const std::string& filepath, const size_t size);

		private slots:
			void OnMessageChanged_(const QString& text);
	};
}
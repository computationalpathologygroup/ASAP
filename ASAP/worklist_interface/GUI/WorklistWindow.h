#ifndef __ASAP_GUI_WORKLISTDATAACQUISITIONINTERFACE__
#define __ASAP_GUI_WORKLISTDATAACQUISITIONINTERFACE__

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <QDropEvent>
#include <QtWidgets/QMainWindow>
#include <QStandardItemModel>
#include <QFutureWatcher>
#include <QSettings>

#include "CompositeChild.h"
#include "ui_WorklistWindowLayout.h"
#include "ASAP/ASAP_Window.h"
#include "../Data/SourceProxy.h"
#include "../Misc/TemporaryDirectoryTracker.h"
#include "../Models/WorklistModels.h"
#include "../GUI/ThumbnailCache.h"

namespace ASAP
{
	// TODO: Decouple source management from worklist window
	class WorklistWindow : public CompositeChild
	{
		Q_OBJECT

		public:
			explicit WorklistWindow(QWidget* parent = nullptr);
			~WorklistWindow(void);

			void attachWorkstation(ASAP_Window& workstation, const int tab_id);

			void setDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> parameters);
			void setDataSource(const std::string& source);

		public slots:
			void updateImageIcons(int itemRow, const QIcon& newIcon);
			void updateStatusBar(const QString& message);

		signals:
			void showMessageBox(const QString message);
			void requestStatusBarUpdate(const QString& message);
			void requestOpenImage(const QString& filepath);
			void requestWorklistRefresh(void);

		private:
			SourceProxy									m_source;
			std::unique_ptr<Ui::WorklistWindowLayout>	m_ui;
			std::unique_ptr<QFutureWatcher<void> >      m_thumbnail_loader;
			std::mutex									m_image_switch_access;
			std::mutex									m_status_bar_access;
			TemporaryDirectoryTracker					m_storage_directory;
			std::vector<std::unique_ptr<QAction>>		m_history_actions;
			ASAP_Window*								m_workstation; // Todo: Clean up or perhaps combine in struct
			int											m_workstation_tab_id;
			WorklistModels								m_models;
			QSettings*									m_settings;

			bool checkSchema(void);
			void readSettings(void);
			void writeSettings(void);
			void stopThumbnailLoading(void);
			void updatePreviousSources(void);
			void updateSourceViews(void);

			std::vector<std::string> getImagesForItem(const std::string& id, const WorklistModels::ModelEnum model);
			//void deleteFromWorklist(QStandardItem* item, const WorklistModels::ModelEnum model);
			void updateWorklist(const QStandardItem* worklist_item, const std::vector<std::string>& image_list, bool remove);

			void setModels(void);
			void setSlots(void);

			void keyPressEvent(QKeyEvent* event);
			bool eventFilter(QObject* obj, QEvent* event);
			void getImageFromIndex(const QModelIndex& index);


		private slots:
			void onWorklistClear(QModelIndex index, int row, int column);
			void onPatientsClear(QModelIndex index, int row, int column);
			void onStudyClear(QModelIndex index, int row, int column);

			void onWorklistSelect(QModelIndex index);
			void onPatientSelect(QModelIndex index);
			void onStudySelect(QModelIndex index);
			void onImageDoubleClicked(QModelIndex index);
			void onImageSelect(const bool checked);
			void onSelectFileSource(bool checked);
			void onSelectFolderSource(bool checked);
			void onSelectExternalSource(bool checked);

		//	void OnDeletePress_();
			void onShowMessageBox(const QString message);
			void onImageDrop(QDropEvent* drop_event);
			void onOpenImage(QString path);
			void onIconDoubleClicked(const QModelIndex& index);
			void onCreateWorklist(void);
			void onWorklistNameChange(QStandardItem* item);
			void onWorklistRefresh(void);
	};
}
#endif // __ASAP_GUI_WORKLISTDATAACQUISITIONINTERFACE__
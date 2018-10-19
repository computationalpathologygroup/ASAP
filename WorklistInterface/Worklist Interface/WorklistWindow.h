#pragma once
#include <deque>
#include <memory>
#include <unordered_map>
#include <QtWidgets/QMainWindow>
#include <QStandardItemModel>

#include "CompositeChild.h"
#include "ui_WorklistWindowLayout.h"
#include "Data/WorklistDataAcquisitionInterface.h"


#include "../../ASAP/pathologyworkstation.h"

namespace ASAP::Worklist::GUI
{
	struct WorklistWindowSettings
	{
		std::string				source_location;
		std::deque<std::string>	previous_sources;
	};

	class WorklistWindow : public CompositeChild // public QMainWindow,
	{
		Q_OBJECT

		public:
			explicit WorklistWindow(QWidget* parent = 0);
			~WorklistWindow(void);

			void AttachWorkstation(PathologyWorkstation& workstation, const int tab_id);
			WorklistWindowSettings GetStandardSettings(void);

			void SetDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> additional_params);
			void SetWorklistItems(const DataTable& items, QStandardItemModel* model);
			void SetPatientsItems(const DataTable& items, QStandardItemModel* model);
			void SetStudyItems(const DataTable& items, QStandardItemModel* model);
			void SetImageItems(const DataTable& items, QStandardItemModel* model);

		private:
			std::unique_ptr<Data::WorklistDataAcquisitionInterface>	m_data_acquisition_;
			std::unique_ptr<Ui::WorklistWindowLayout>				m_ui_;
			WorklistWindowSettings									m_settings_;
			PathologyWorkstation*									m_workstation_;
			int														m_workstation_tab_id_;
			std::vector<std::unique_ptr<QAction>>					m_history_actions_;

			QStandardItemModel* m_images_model_;
			QStandardItemModel* m_patients_model_;
			QStandardItemModel* m_studies_model_;
			QStandardItemModel* m_worklist_model_;

			bool CheckSchema_(Data::WorklistDataAcquisitionInterface* source);
			void LoadSettings_(void);
			void StoreSettings_(void);
			void UpdatePreviousSources_(void);
			void UpdateSourceViews_(void);

			void SetHeaders_(const std::unordered_set<std::string> headers, QStandardItemModel* model, QAbstractItemView* view);
			void SetModels_(void);
			void SetSlots_(void);

		private slots:
			void OnWorklistClear_(QModelIndex index, int row, int column);
			void OnPatientsClear_(QModelIndex index, int row, int column);
			void OnStudyClear_(QModelIndex index, int row, int column);

			void OnWorklistSelect_(QModelIndex index);
			void OnPatientSelect_(QModelIndex index);
			void OnStudySelect_(QModelIndex index);
			void OnImageSelect_(QModelIndex index);

			void OnSelectFileSource_(bool checked);
			void OnSelectFolderSource_(bool checked);
			void OnSelectExternalSource_(bool checked);
			void UpdateStatusBar_(const QString& message);
	};
}
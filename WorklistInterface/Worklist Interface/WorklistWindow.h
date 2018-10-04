#pragma once

#include <memory>
#include <QtWidgets/QMainWindow>
#include <QStandardItemModel>

#include "Data/WorklistDataAcquisitionInterface.h"
#include "Data/DataTable.h"
#include "ui_WorklistWindowLayout.h"

#include "../../ASAP/pathologyworkstation.h"

namespace ASAP::Worklist::GUI
{
	struct WorklistWindowSettings
	{
		std::string source_location;
	};

	class WorklistWindow : public QMainWindow
	{
		Q_OBJECT

		public:
			explicit WorklistWindow(QWidget* parent = 0);

			void AttachWorkstation(PathologyWorkstation& workstation);
			WorklistWindowSettings GetStandardSettings(void);

			void SetWorklistItems(const DataTable& items, QStandardItemModel* model);
			void SetPatientsItems(const DataTable& items, QStandardItemModel* model);
			void SetStudyItems(const DataTable& items, QStandardItemModel* model);
			void SetImageItems(const DataTable& items, QStandardItemModel* model);

		private:
			std::unique_ptr<Data::WorklistDataAcquisitionInterface>	m_data_acquisition_;
			std::unique_ptr<Ui::WorklistWindowLayout>				m_ui_;
			PathologyWorkstation*									m_workstation_;

			QStandardItemModel* m_images_model_;
			QStandardItemModel* m_patients_model_;
			QStandardItemModel* m_studies_model_;
			QStandardItemModel* m_worklist_model_;

			void AdjustGuiToSource_(const Data::WorklistDataAcquisitionInterface::SourceType type);
			QIcon CreateIcon_(const std::string absolute_filepath);
			void InitializeSource_(const std::string& source_path, const bool is_file);
			void LoadSettings_(void);

			void SetHeaders_(std::vector<std::string> headers, QStandardItemModel* model, QAbstractItemView* view);
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
	};
}
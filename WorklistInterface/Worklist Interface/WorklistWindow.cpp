#include "WorklistWindow.h"

#include <cctype>
#include <unordered_map>

#include "Data/SourceLoading.h"
#include "INI_Parsing.h"

namespace ASAP::Worklist::GUI
{
	WorklistWindow::WorklistWindow(QWidget *parent) :
		QMainWindow(parent),
		m_ui_(new Ui::WorklistWindowLayout),
		m_data_acquisition_(nullptr),
		m_images_model_(new QStandardItemModel(0, 0)),
		m_patients_model_(new QStandardItemModel(0, 0)),
		m_studies_model_(new QStandardItemModel(0, 0)),
		m_worklist_model_(new QStandardItemModel(0, 0))
	{
		m_ui_->setupUi(this);
		SetSlots_();
		SetModels_();
		LoadSettings_();
		SetDataSource(m_settings_.source_location);
	}

	void WorklistWindow::AttachWorkstation(PathologyWorkstation& workstation)
	{
		m_workstation_ = &workstation;
	}

	WorklistWindowSettings WorklistWindow::GetStandardSettings(void)
	{
		return WorklistWindowSettings();
	}

	void WorklistWindow::SetDataSource(const std::string source_path)
	{
		m_data_acquisition_ = Data::LoadDataSource(source_path);
		AdjustGuiToSource_();
	}

	void WorklistWindow::SetWorklistItems(const DataTable& items, QStandardItemModel* model)
	{
		std::unordered_map<std::string, QStandardItem*> inserted_items;
		model->removeRows(0, model->rowCount());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, std::vector<std::string>{"id", "title", "parent"}));
			QStandardItem* model_item(new QStandardItem(QString(record[1]->data())));

			model_item->setData(QVariant(std::stoi(*record[0])));
			inserted_items.insert({ *record[0], model_item });

			if (record[2]->empty())
			{
				model->setItem(m_worklist_model_->rowCount(), 0, model_item);
			}
			else
			{
				auto parent = inserted_items.find(*record[2]);
				if (parent != inserted_items.end())
				{
					parent->second->appendRow(model_item);
				}
			}
		}
	}

	void WorklistWindow::SetPatientsItems(const DataTable& items, QStandardItemModel* model)
	{
		model->removeRows(0, model->rowCount());
		model->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, DataTable::FIELD_SELECTION::VISIBLE));
			size_t record_id = std::stoi(*record[0]);

			for (size_t field = 0; field < record.size(); ++field)
			{
				QStandardItem* model_item = new QStandardItem(QString(record[field]->data()));
				model_item->setData(QVariant(record_id));
				model->setItem(item, field, model_item);
			}
		}
	}

	void WorklistWindow::SetStudyItems(const DataTable& items, QStandardItemModel* model)
	{
		model->removeRows(0, model->rowCount());
		model->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, DataTable::FIELD_SELECTION::VISIBLE));
			size_t record_id = std::stoi(*record[0]);

			for (size_t field = 0; field < record.size(); ++field)
			{
				QStandardItem* model_item = new QStandardItem(QString(record[field]->data()));
				model_item->setData(QVariant(record_id));
				model->setItem(item, field, model_item);
			}
		}
	}

	void WorklistWindow::SetImageItems(const DataTable& items, QStandardItemModel* model)
	{
		model->removeRows(0, model->rowCount());
		model->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, { "id", "location", "title" }));

			QStandardItem* model_item = new QStandardItem(CreateIcon_(*record[1]), QString(record[2]->data()));
			model_item->setData(QVariant(QString(record[1]->data())));
			model->setItem(item, 0, model_item);
		}
	}

	void WorklistWindow::AdjustGuiToSource_(void)
	{
		Data::WorklistDataAcquisitionInterface::SourceType type = Data::WorklistDataAcquisitionInterface::SourceType::FILELIST;

		if (m_data_acquisition_)
		{
			type = m_data_acquisition_->GetSourceType();

			if (type == Data::WorklistDataAcquisitionInterface::SourceType::FULL_WORKLIST)
			{
				SetHeaders_(m_data_acquisition_->GetPatientHeaders(), m_patients_model_, m_ui_->PatientView);
				SetHeaders_(m_data_acquisition_->GetStudyHeaders(), m_studies_model_, m_ui_->StudyView);

				QStandardItemModel* worklist_model = m_worklist_model_;
				m_data_acquisition_->GetWorklistRecords([this, worklist_model](DataTable table, int error)
				{
					if (error == 0)
					{
						this->SetWorklistItems(table, worklist_model);
					}
				});
			}
		}

		// Default is regarded as the FILELIST sourcetype
		switch (type)
		{
			case Data::WorklistDataAcquisitionInterface::SourceType::FULL_WORKLIST: 
				m_ui_->label_worklists->setVisible(true);
				m_ui_->label_patients->setVisible(true);
				m_ui_->label_studies->setVisible(true);
				m_ui_->WorklistView->setVisible(true);
				m_ui_->PatientView->setVisible(true);
				m_ui_->StudyView->setVisible(true);
				break;
			default:
				m_ui_->label_worklists->setVisible(false);
				m_ui_->label_patients->setVisible(false);
				m_ui_->label_studies->setVisible(false);
				m_ui_->WorklistView->setVisible(false);
				m_ui_->PatientView->setVisible(false);
				m_ui_->StudyView->setVisible(false);
				break;
		}
	}

	QIcon WorklistWindow::CreateIcon_(const std::string absolute_filepath)
	{
		QPixmap pixmap(QString(absolute_filepath.data()));
		if (pixmap.isNull())
		{
			pixmap = QPixmap(QString("./img/unavailable.png"));
		}
		QIcon icon(pixmap);
		return icon;
	}

	void WorklistWindow::LoadSettings_(void)
	{
		try
		{
			std::unordered_map<std::string, std::string> values(INI::ParseINI("worklist_config.ini"));
			auto source_value(values.find("source"));
			if (source_value != values.end())
			{
				m_settings_.source_location = source_value->second;
			}
		}
		catch (const std::runtime_error& e)
		{
			// Creates an INI file with standard settings.
			std::unordered_map<std::string, std::string> values;
			values.insert({ "source", "" });
			INI::WriteINI("worklist_config.ini", values);
		}
	}

	void WorklistWindow::SetHeaders_(std::vector<std::string> headers, QStandardItemModel* model, QAbstractItemView* view)
	{
		QStringList q_headers;
		for (std::string& header : headers)
		{
			header[0] = std::toupper(header[0]);
			q_headers.push_back(QString(header.c_str()));
		}
		model->setColumnCount(q_headers.size());
		model->setHorizontalHeaderLabels(q_headers);
		view->update();
	}

	void WorklistWindow::SetModels_(void)
	{
		m_ui_->ImageView->setModel(m_images_model_);
		m_ui_->PatientView->setModel(m_patients_model_);
		m_ui_->StudyView->setModel(m_studies_model_);
		m_ui_->WorklistView->setModel(m_worklist_model_);
	}

	//================================= Slots =================================//
	
	void WorklistWindow::SetSlots_(void)
	{
		connect(m_worklist_model_,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(OnWorklistClear_(QModelIndex, int, int)));

		connect(m_patients_model_,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(OnPatientsClear_(QModelIndex, int, int)));

		connect(m_studies_model_,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(OnStudyClear_(QModelIndex, int, int)));

		connect(m_ui_->WorklistView,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnWorklistSelect_(QModelIndex)));

		connect(m_ui_->PatientView,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnPatientSelect_(QModelIndex)));

		connect(m_ui_->StudyView,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnStudySelect_(QModelIndex)));

		connect(m_ui_->ImageView,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnImageSelect_(QModelIndex)));
	}

	void WorklistWindow::OnWorklistClear_(QModelIndex index, int, int)
	{
		m_patients_model_->removeRows(0, m_patients_model_->rowCount());
		m_ui_->PatientView->update();
	}

	void WorklistWindow::OnPatientsClear_(QModelIndex index, int, int)
	{
		m_studies_model_->removeRows(0, m_studies_model_->rowCount());
		m_ui_->StudyView->update();
	}

	void WorklistWindow::OnStudyClear_(QModelIndex index, int, int)
	{
		m_images_model_->removeRows(0, m_images_model_->rowCount());
		m_ui_->ImageView->update();
	}

	void WorklistWindow::OnWorklistSelect_(QModelIndex index)
	{
		QStandardItem* item(m_worklist_model_->itemFromIndex(index));
		int worklist_id = item->data().toInt();

		QStandardItemModel* patient_model = m_patients_model_;
		QTableView* patient_view = m_ui_->PatientView;
		m_data_acquisition_->GetPatientRecords(worklist_id, [this, patient_model, patient_view](DataTable table, int error)
		{
			if (error == 0)
			{
				this->SetPatientsItems(table, patient_model);
				patient_view->update();
			}
		});
	}

	void WorklistWindow::OnPatientSelect_(QModelIndex index)
	{
		QStandardItem* item(m_patients_model_->itemFromIndex(index));
		int patient_id = item->data().toInt();

		QStandardItemModel* study_model = m_studies_model_;
		QTableView* study_view = m_ui_->StudyView;
		m_data_acquisition_->GetStudyRecords(patient_id, [this, study_model, study_view](DataTable table, int error)
		{
			if (error == 0)
			{
				this->SetStudyItems(table, study_model);
				study_view->update();
			}
		});
	}

	void WorklistWindow::OnStudySelect_(QModelIndex index)
	{
		QStandardItem* item(m_studies_model_->itemFromIndex(index));
		int study_id = item->data().toInt();

		QStandardItemModel* image_model = m_images_model_;
		QListView* image_view = m_ui_->ImageView;
		m_data_acquisition_->GetImageRecords(study_id, [this, image_model, image_view](DataTable table, int error)
		{
			if (error == 0)
			{
				this->SetImageItems(table, image_model);
				image_view->update();
			}
		});
	}

	void WorklistWindow::OnImageSelect_(QModelIndex index)
	{
		QStandardItem* item(m_images_model_->itemFromIndex(index));
		QString image_handle = item->data().toString();

		if (m_workstation_)
		{
			m_workstation_->openFile(image_handle);
		}
	}
}
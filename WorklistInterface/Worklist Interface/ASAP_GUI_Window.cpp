#include "ASAP_GUI_Window.h"

#include <cctype>
#include <unordered_map>

namespace ASAP::Worklist::GUI
{
	ASAP_GUI_Window::ASAP_GUI_Window(Data::DjangoDataAcquisition* data_acquisition, QWidget *parent) :
		QMainWindow(parent),
		m_ui_(new Ui::ASAP_GUI_Layout),
		m_data_acquisition_(data_acquisition),
		m_images_model_(new QStandardItemModel(0, 0)),
		m_patients_model_(new QStandardItemModel(0, 0)),
		m_studies_model_(new QStandardItemModel(0, 0)),
		m_worklist_model_(new QStandardItemModel(0, 0))
	{
		m_ui_->setupUi(this);
		SetSlots_();

		m_ui_->ImageView->setModel(m_images_model_);
		m_ui_->PatientView->setModel(m_patients_model_);
		m_ui_->StudyView->setModel(m_studies_model_);
		m_ui_->WorklistView->setModel(m_worklist_model_);

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

	void ASAP_GUI_Window::SetWorklistItems(const DataTable& items, QStandardItemModel* model)
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

	void ASAP_GUI_Window::SetPatientsItems(const DataTable& items, QStandardItemModel* model)
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

	void ASAP_GUI_Window::SetStudyItems(const DataTable& items, QStandardItemModel* model)
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

	void ASAP_GUI_Window::SetImageItems(const DataTable& items, QStandardItemModel* model)
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

	QIcon ASAP_GUI_Window::CreateIcon_(const std::string absolute_filepath)
	{
		QPixmap pixmap(QString(absolute_filepath.data()));
		if (pixmap.isNull())
		{
			pixmap = QPixmap(QString("./img/unavailable.png"));
		}
		QIcon icon(pixmap);
		return icon;
	}

	void ASAP_GUI_Window::SetHeaders_(std::vector<std::string> headers, QStandardItemModel* model, QAbstractItemView* view)
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

	void ASAP_GUI_Window::SetSlots_(void)
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

	void ASAP_GUI_Window::OnWorklistClear_(QModelIndex index, int, int)
	{
		m_patients_model_->removeRows(0, m_patients_model_->rowCount());
		m_ui_->PatientView->update();
	}

	void ASAP_GUI_Window::OnPatientsClear_(QModelIndex index, int, int)
	{
		m_studies_model_->removeRows(0, m_studies_model_->rowCount());
		m_ui_->StudyView->update();
	}

	void ASAP_GUI_Window::OnStudyClear_(QModelIndex index, int, int)
	{
		m_images_model_->removeRows(0, m_images_model_->rowCount());
		m_ui_->ImageView->update();
	}

	void ASAP_GUI_Window::OnWorklistSelect_(QModelIndex index)
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

	void ASAP_GUI_Window::OnPatientSelect_(QModelIndex index)
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

	void ASAP_GUI_Window::OnStudySelect_(QModelIndex index)
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

	void ASAP_GUI_Window::OnImageSelect_(QModelIndex index)
	{
		QStandardItem* item(m_images_model_->itemFromIndex(index));
		QString image_handle = item->data().toString();

		if (workstation_window)
		{
			workstation_window->openFile(image_handle);
		}
	}
}
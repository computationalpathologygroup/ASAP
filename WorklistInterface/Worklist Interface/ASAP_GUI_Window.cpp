#include "ASAP_GUI_Window.h"

#include <cctype>
#include <unordered_map>

ASAP_GUI_Window::ASAP_GUI_Window(AbstractWorklistDataAcquisition* data_acquisition, QWidget *parent) :
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

	DataTable worklist_items(m_data_acquisition_->GetWorklistRecords());
	SetWorklistItems_(worklist_items, m_worklist_model_);
}

void ASAP_GUI_Window::SetWorklistItems_(const DataTable& items, QStandardItemModel* model)
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

void ASAP_GUI_Window::SetPatientsItems_(const DataTable& items, QStandardItemModel* model)
{
	model->removeRows(0, model->rowCount());
	model->setRowCount(items.Size());
	for (size_t item = 0; item < items.Size(); ++item)
	{
		std::vector<const std::string*> record(items.At(item, DataTable::FIELD_SELECTION::VISIBLE));
		for (size_t field = 0; field < record.size(); ++field)
		{
			model->setItem(item, field, new QStandardItem(QString(record[field]->data())));
		}
	}
}

void ASAP_GUI_Window::SetStudyItems_(const DataTable& items, QStandardItemModel* model)
{
	model->removeRows(0, model->rowCount());
	model->setRowCount(items.Size());
	for (size_t item = 0; item < items.Size(); ++item)
	{
		std::vector<const std::string*> record(items.At(item, DataTable::FIELD_SELECTION::VISIBLE));
		for (size_t field = 0; field < record.size(); ++field)
		{
			model->setItem(item, field, new QStandardItem(QString(record[field]->data())));
		}
	}
}

void ASAP_GUI_Window::SetImageItems_(const DataTable& items, QStandardItemModel* model)
{
	model->removeRows(0, model->rowCount());
	model->setRowCount(items.Size());
	for (size_t item = 0; item < items.Size(); ++item)
	{
		std::vector<const std::string*> record(items.At(item, { "location", "title" }));
		model->setItem(item, 0, new QStandardItem(QIcon(QPixmap(QString(record[0]->data()))), QString(record[1]->data())));
	}
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
	DataTable patient_items(m_data_acquisition_->GetPatientRecords(worklist_id));
	SetPatientsItems_(patient_items, m_patients_model_);
	m_ui_->PatientView->update();
}

void ASAP_GUI_Window::OnPatientSelect_(QModelIndex index)
{
	QStandardItem* item(m_patients_model_->itemFromIndex(index));

	int patient_id = item->data().toInt();
	DataTable study_items(m_data_acquisition_->GetStudyRecords(patient_id));
	SetStudyItems_(study_items, m_studies_model_);
	m_ui_->StudyView->update();
}

void ASAP_GUI_Window::OnStudySelect_(QModelIndex index)
{
	QStandardItem* item(m_studies_model_->itemFromIndex(index));

	int study_id = item->data().toInt();
	DataTable image_items(m_data_acquisition_->GetImageRecords(study_id));
	SetImageItems_(image_items, m_images_model_);
	m_ui_->StudyView->update();
}

void ASAP_GUI_Window::OnImageSelect_(QModelIndex index)
{

}

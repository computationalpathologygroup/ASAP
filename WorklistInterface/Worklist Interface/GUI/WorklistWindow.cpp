#include "WorklistWindow.h"

#include <cctype>
#include <sstream>
#include <unordered_map>

#include <boost/bind.hpp>
#include <qfiledialog.h>
#include <qinputdialog.h>
#include <qmessagebox.h>
#include <QtConcurrent\qtconcurrentrun.h>

#include "ExternalSourceDialog.h"
#include "IconCreator.h"
#include "Serialization/INI.h"
#include "Misc/StringManipulation.h"

using namespace ASAP::Data;

namespace ASAP::GUI
{
	WorklistWindow::WorklistWindow(QWidget* parent) :
		CompositeChild(parent),
		m_ui_(new Ui::WorklistWindowLayout),
		m_images_model_(new QStandardItemModel(0, 0)),
		m_patients_model_(new QStandardItemModel(0, 0)),
		m_studies_model_(new QStandardItemModel(0, 0)),
		m_worklist_model_(new QStandardItemModel(0, 0)),
		m_storage_directory_(boost::filesystem::path("/temp/"), Misc::TemporaryDirectoryTracker::GetStandardConfiguration()),
		m_source_(m_storage_directory_)
	{
		m_ui_->setupUi(this);
		SetSlots_();
		SetModels_();
		LoadSettings_();

		SetDataSource(m_source_.GetCurrentSource());
	}

	WorklistWindow::~WorklistWindow(void)
	{
		StoreSettings_();
	}

	void WorklistWindow::AttachWorkstation(ASAP_Window& workstation, const int tab_id)
	{
		m_workstation_			= &workstation;
		m_workstation_tab_id_	= tab_id;
	}

	void WorklistWindow::SetDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> parameters)
	{
		SetDataSource(SourceProxy::SerializeSource(source_path, parameters));
	}

	void WorklistWindow::SetDataSource(const std::string& source)
	{
		if (!m_source_.IsInitialized() || source != m_source_.GetCurrentSource())
		{
			std::pair<std::string, std::unordered_map<std::string, std::string>> deserialized_source(SourceProxy::DeserializeSource(source));

			// Halts any leftover thumbnail loading.
			StopThumbnailLoading_();

			try
			{
				// Attempts to load the data source and then confirms it has the required fields for the UI to function.
				m_ui_->status_bar->showMessage("Loading source: " + QString(deserialized_source.first.data()));
				m_source_.LoadSource(source);

				if (!m_source_.IsInitialized())
				{
					throw std::runtime_error("Unable to load source: " + deserialized_source.first);
				}
				if (!CheckSchema_())
				{
					m_source_.Close();
					throw std::runtime_error("Selected source has schema errors. Unable to open.");
				}

				

				UpdatePreviousSources_();
				UpdateSourceViews_();
				m_ui_->status_bar->showMessage("Succesfully loaded source: " + QString(deserialized_source.first.data()));
			}
			catch (const std::exception& e)
			{
				m_ui_->status_bar->showMessage("Unable to load source: " + QString(deserialized_source.first.data()));
				QMessageBox::question(this, "Error", e.what(), QMessageBox::Ok);
			}
		}
	}

	void WorklistWindow::SetWorklistItems(const DataTable& items, QStandardItemModel* model)
	{
		if (model->rowCount() == 0)
		{
			QStandardItem* model_item(new QStandardItem("All"));
			model->setItem(m_worklist_model_->rowCount(), 0, model_item);
		}

		model->removeRows(1, model->rowCount());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, std::vector<std::string>{ "id", "title" }));
			QStandardItem* model_item(new QStandardItem(QString(record[1]->data())));
			model_item->setData(QVariant(record[0]->data()));
			model->setItem(m_worklist_model_->rowCount(), 0, model_item);
		}
	}

	void WorklistWindow::SetPatientsItems(const Data::DataTable& items, QStandardItemModel* model)
	{
		model->removeRows(0, model->rowCount());
		model->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, Data::DataTable::FIELD_SELECTION::VISIBLE));
			for (size_t field = 0; field < record.size(); ++field)
			{
				QStandardItem* model_item = new QStandardItem(QString(record[field]->data()));
				model_item->setData(QVariant(items.At(item, { "id" })[0]->data()));
				model->setItem(item, field, model_item);
			}
		}
	}

	void WorklistWindow::SetStudyItems(const Data::DataTable& items, QStandardItemModel* model)
	{
		model->removeRows(0, model->rowCount());
		model->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, Data::DataTable::FIELD_SELECTION::VISIBLE));
			for (size_t field = 0; field < record.size(); ++field)
			{
				QStandardItem* model_item = new QStandardItem(QString(record[field]->data()));
				model_item->setData(QVariant(items.At(item, { "id" })[0]->data()));
				model->setItem(item, field, model_item);
			}
		}
	}

	// Todo: Loading and halting might be a bit too messy in terms of tasks and code calling, refactor
	void WorklistWindow::SetImageItems(const Data::DataTable& items, QStandardItemModel* model)
	{
		m_stop_loading_ = false;

		model->removeRows(0, model->rowCount());
		QtConcurrent::run([this, items, model, lock=&m_image_loading_access_, stop_loading=&m_stop_loading_, size=200]()
		{
			IconCreator creator;

			connect(&creator,
					&IconCreator::RequiresItemRefresh,
					this,
					&WorklistWindow::UpdateImageIcons);

			connect(&creator,
					&IconCreator::RequiresStatusBarChange,
					this,
					&WorklistWindow::UpdateStatusBar);

			lock->lock();
			creator.InsertIcons(items, model, size, *stop_loading);
			lock->unlock();
		});
	}

	// Todo: Decouple and create abstraction of schema's
	bool WorklistWindow::CheckSchema_(void)
	{
		if (m_source_.IsInitialized())
		{
			// No schema check is required for a filelist source.
			if (m_source_.GetSourceType() == Data::WorklistSourceInterface::FULL_WORKLIST)
			{
				std::set<std::string> worklist_headers(m_source_.GetWorklistHeaders());
				std::set<std::string> patient_headers(m_source_.GetPatientHeaders());
				std::set<std::string> study_headers(m_source_.GetStudyHeaders());
				std::set<std::string> image_headers(m_source_.GetImageHeaders());

				return	(worklist_headers.find("id") != worklist_headers.end()) &&
						(patient_headers.find("id") != worklist_headers.end()) &&
						(study_headers.find("id") != worklist_headers.end()) &&
						(image_headers.find("id") != worklist_headers.end()) &&
						(image_headers.find("title") != worklist_headers.end());
			}
			return true;
		}
		return false;
	}

	void WorklistWindow::LoadSettings_(void)
	{
		try
		{
			std::unordered_map<std::string, std::string> values(Serialization::INI::ParseINI("worklist_config.ini"));

			// Acquires the last known source.
			auto source_value(values.find("source"));
			std::string current_source;
			if (source_value != values.end())
			{
				current_source = source_value->second;
			}

			// Acquires the five most recent sources.
			auto previous_sources_value(values.find("previous_sources"));
			std::vector<std::string> previous_sources;
			if (previous_sources_value != values.end() && !previous_sources_value->second.empty())
			{
				std::vector<std::string> split_sources(Misc::Split(previous_sources_value->second));
				for (const std::string& source : split_sources)
				{
					previous_sources.push_back(source);
				}
			}

			m_source_.SetSourceInformation(current_source, previous_sources);
		}
		catch (const std::runtime_error& e)
		{
			// Creates an INI file with standard settings.
			std::unordered_map<std::string, std::string> values;
			values.insert({ "source", "" });
			values.insert({ "previous_sources", "" });
			Serialization::INI::WriteINI("worklist_config.ini", values);
		}
	}
	
	void WorklistWindow::StoreSettings_(void)
	{
		std::unordered_map<std::string, std::string> values;
		values.insert({ "source", m_source_.GetCurrentSource() });
		
		std::string previous_sources;
		for (const std::string& source : m_source_.GetPreviousSources())
		{
			previous_sources += "\"" + source + "\",";
		}

		// Removes the last comma and then inserts.
		values.insert({ "previous_sources", previous_sources.substr(0, previous_sources.size() - 1)});

		Serialization::INI::WriteINI("worklist_config.ini", values);
	}

	void WorklistWindow::StopThumbnailLoading_(void)
	{
		m_stop_loading_ = true;
		m_image_loading_access_.lock();
		m_image_loading_access_.unlock();
	}

	void WorklistWindow::UpdatePreviousSources_(void)
	{
		for (const std::unique_ptr<QAction>& action : m_history_actions_)
		{
			m_ui_->menu_history->removeAction(action.get());
		}
		m_history_actions_.clear();

		auto previous_source(m_source_.GetPreviousSources());
		for (const std::string& prev_source : previous_source)
		{
			std::pair<std::string, std::unordered_map<std::string, std::string>> source_parameters(SourceProxy::DeserializeSource(prev_source));

			m_history_actions_.push_back(std::unique_ptr<QAction>(new QAction(QString(source_parameters.first.data()), this)));
			m_ui_->menu_history->addAction(m_history_actions_.back().get());

			connect(m_history_actions_.back().get(),
				&QAction::triggered,
				this,
				[action = m_history_actions_.back().get(), this, source_parameters](bool checked)
				{
					this->SetDataSource(source_parameters.first, source_parameters.second);
				});
		}
	}

	void WorklistWindow::UpdateSourceViews_(void)
	{
		// Clears all previous source information.
		SetWorklistItems(Data::DataTable(), m_worklist_model_);
		SetPatientsItems(Data::DataTable(), m_patients_model_);
		SetStudyItems(Data::DataTable(), m_studies_model_);
		SetImageItems(Data::DataTable(), m_images_model_);

		// Resets the view to the Filelist standard.
		WorklistSourceInterface::SourceType type = WorklistSourceInterface::SourceType::FILELIST;

		// Adjusts the GUI to the actual new source, if it was initialized succesfully.
		if (m_source_.IsInitialized())
		{
			type = m_source_.GetSourceType();

			if (type == WorklistSourceInterface::SourceType::FILELIST)
			{
				QStandardItemModel* image_model = m_images_model_;
				m_source_.GetImageRecords(std::string(), std::string(), [this, image_model](Data::DataTable& table, int error)
				{
					if (error == 0)
					{
						this->SetImageItems(table, image_model);
					}
				});
			}
			else if (type == WorklistSourceInterface::SourceType::FULL_WORKLIST)
			{
				SetHeaders_(m_source_.GetPatientHeaders(Data::DataTable::FIELD_SELECTION::VISIBLE), m_patients_model_, m_ui_->view_patients);
				SetHeaders_(m_source_.GetStudyHeaders(Data::DataTable::FIELD_SELECTION::VISIBLE), m_studies_model_, m_ui_->view_studies);

				this->RequestWorklistRefresh();
			}
		}

		// Default is regarded as the FILELIST sourcetype
		switch (type)
		{
			case WorklistSourceInterface::SourceType::FULL_WORKLIST:
				m_ui_->label_worklists->setVisible(true);
				m_ui_->label_patients->setVisible(true);
				m_ui_->label_studies->setVisible(true);
				m_ui_->view_worklists->setVisible(true);
				m_ui_->view_patients->setVisible(true);
				m_ui_->view_studies->setVisible(true);
				m_ui_->button_create_worklist->setVisible(true);
				break;
			default:
				m_ui_->label_worklists->setVisible(false);
				m_ui_->label_patients->setVisible(false);
				m_ui_->label_studies->setVisible(false);
				m_ui_->view_worklists->setVisible(false);
				m_ui_->view_patients->setVisible(false);
				m_ui_->view_studies->setVisible(false);
				m_ui_->button_create_worklist->setVisible(false);
				break;
		}
	}

	void WorklistWindow::SetHeaders_(const std::set<std::string> headers, QStandardItemModel* model, QAbstractItemView* view)
	{
		QStringList q_headers;
		for (const std::string& column : headers)
		{
			std::string capital_column = column;
			capital_column[0] = std::toupper(capital_column[0]);
			q_headers.push_back(QString(capital_column.c_str()));
		}
		model->setColumnCount(q_headers.size());
		model->setHorizontalHeaderLabels(q_headers);
		view->update();
	}

	void WorklistWindow::SetModels_(void)
	{
		m_ui_->view_images->setModel(m_images_model_);
		m_ui_->view_patients->setModel(m_patients_model_);
		m_ui_->view_studies->setModel(m_studies_model_);
		m_ui_->view_worklists->setModel(m_worklist_model_);
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

		connect(m_ui_->view_worklists,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnWorklistSelect_(QModelIndex)));

		connect(m_ui_->view_patients,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnPatientSelect_(QModelIndex)));

		connect(m_ui_->view_studies,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnStudySelect_(QModelIndex)));

		connect(m_ui_->view_images,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(OnImageSelect_(QModelIndex)));

		connect(m_ui_->view_images,
			SIGNAL(activated(QModelIndex)),
			this,
			SLOT(OnImageSelect_(QModelIndex)));

		connect(m_ui_->button_create_worklist,
			&QPushButton::clicked,
			this,
			&WorklistWindow::OnCreateWorklist_);

		connect(m_ui_->action_open_file,
			&QAction::triggered,
			this,
			&WorklistWindow::OnSelectFileSource_);

		connect(m_ui_->action_open_folder,
			&QAction::triggered,
			this,
			&WorklistWindow::OnSelectFolderSource_);

		connect(m_ui_->action_open_external,
			&QAction::triggered,
			this,
			&WorklistWindow::OnSelectExternalSource_);

		connect(this,
			&WorklistWindow::RequestStatusBarUpdate,
			this,
			&WorklistWindow::UpdateStatusBar);
			
		connect(this,
			&WorklistWindow::RequestOpenImage,
			this,
			&WorklistWindow::OnOpenImage_);

		connect(this,
			&WorklistWindow::RequestWorklistRefresh,
			this,
			&WorklistWindow::OnWorklistRefresh);		
	}

	void WorklistWindow::MoveImageSelectionLeft(void)
	{
		if (m_image_switch_access_.try_lock())
		{
			QModelIndexList indexes(m_ui_->view_images->selectionModel()->selectedIndexes());
			if (indexes.size() > 0 && indexes[0].row() > 0)
			{
				m_ui_->view_images->selectionModel()->setCurrentIndex(m_images_model_->index(indexes[0].row() - 1, 0), QItemSelectionModel::SelectCurrent);
				OnImageSelect_(m_images_model_->index(indexes[0].row() - 1, 0));
			}
			m_image_switch_access_.unlock();
		}
	}

	void WorklistWindow::MoveImageSelectionRight(void)
	{
		if (m_image_switch_access_.try_lock())
		{
			QModelIndexList indexes(m_ui_->view_images->selectionModel()->selectedIndexes());
			if (indexes.size() > 0 && indexes[0].row() < m_images_model_->rowCount() - 1)
			{

				m_ui_->view_images->selectionModel()->setCurrentIndex(m_images_model_->index(indexes[0].row() + 1, 0), QItemSelectionModel::SelectCurrent);
				OnImageSelect_(m_images_model_->index(indexes[0].row() + 1, 0));
			}
			m_image_switch_access_.unlock();
		}
	}

	void WorklistWindow::UpdateImageIcons(void)
	{
		m_images_model_->layoutChanged();
	}

	void WorklistWindow::UpdateStatusBar(const QString& message)
	{
		m_status_bar_access_.lock();
		m_ui_->status_bar->showMessage(message);
		m_status_bar_access_.unlock();
	}

	void WorklistWindow::OnWorklistClear_(QModelIndex index, int, int)
	{
		m_patients_model_->removeRows(0, m_patients_model_->rowCount());
		m_ui_->view_patients->update();
	}

	void WorklistWindow::OnPatientsClear_(QModelIndex index, int, int)
	{
		m_studies_model_->removeRows(0, m_studies_model_->rowCount());
		m_ui_->view_studies->update();
	}

	void WorklistWindow::OnStudyClear_(QModelIndex index, int, int)
	{
		m_images_model_->removeRows(0, m_images_model_->rowCount());
		m_ui_->view_images->update();
	}

	void WorklistWindow::OnWorklistSelect_(QModelIndex index)
	{
		QStandardItem* item(m_worklist_model_->itemFromIndex(index));
		std::string worklist_id(item->data().toString().toUtf8().constData());

		QStandardItemModel* patient_model	= m_patients_model_;
		QTableView* patient_view			= m_ui_->view_patients;
		m_source_.GetPatientRecords(worklist_id, [this, patient_model, patient_view](Data::DataTable table, int error)
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
		std::string patient_id(item->data().toString().toUtf8().constData());

		QStandardItemModel* study_model = m_studies_model_;
		QTableView* study_view = m_ui_->view_studies;
		m_source_.GetStudyRecords(patient_id, [this, study_model, study_view](Data::DataTable table, int error)
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
		QModelIndexList selected_worklist(m_ui_->view_worklists->selectionModel()->selectedIndexes());

		QStandardItem* study_item(m_studies_model_->itemFromIndex(index));
		QStandardItem* worklist_item(m_worklist_model_->itemFromIndex(selected_worklist[0]));

		std::string study_id(study_item->data().toString().toUtf8().constData());
		std::string worklist_id(worklist_item->data().toString().toUtf8().constData());

		QStandardItemModel* image_model = m_images_model_;
		QListView* image_view = m_ui_->view_images;
		m_source_.GetImageRecords(worklist_id, study_id, [this, image_model, image_view](Data::DataTable table, int error)
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
		std::string image_index(item->data().toString().toUtf8().constData());

		if (m_workstation_)
		{
			m_ui_->status_bar->showMessage("Loading image: 0%");
			auto image_loading([this](const boost::filesystem::path& filepath)
			{
				this->RequestOpenImage(QString::fromStdString(filepath.string()));
			});

			auto acquisition_tracking([this, bar=m_ui_->status_bar](const uint8_t progress)
			{
				if (bar->currentMessage().endsWith("%"));
				{
					this->UpdateStatusBar("Loading image: " + QString(std::to_string(progress).data()) + "%");
				}
			});

			m_source_.GetImageFile(image_index, image_loading, acquisition_tracking);
		}
	}

	void WorklistWindow::OnSelectFileSource_(bool checked)
	{
		QFileDialog* dialog = new QFileDialog(this);
		dialog->setFileMode(QFileDialog::AnyFile);
		dialog->setNameFilter({ "*.txt" });
		dialog->exec();
		QStringList names = dialog->selectedFiles();
		
		if (names.size() > 0)
		{
			SetDataSource(dialog->selectedFiles()[0].toUtf8().constData(), std::unordered_map<std::string, std::string>());
		}
	}
	
	void WorklistWindow::OnSelectFolderSource_(bool checked)
	{
		QFileDialog* dialog = new QFileDialog(this);
		dialog->setFileMode(QFileDialog::Directory);
		dialog->exec();
		QStringList names = dialog->selectedFiles();

		if (names.size() > 0)
		{
			SetDataSource(dialog->selectedFiles()[0].toUtf8().constData(), std::unordered_map<std::string, std::string>());
		}
	}

	void WorklistWindow::OnSelectExternalSource_(bool checked)
	{
		ExternalSourceDialog* dialog = new ExternalSourceDialog(this);
		dialog->exec();

		if (dialog->HasValidCredentials())
		{
			ExternalSourceDialog::SourceDialogResults results(dialog->GetLoginDetails());

			std::unordered_map<std::string, std::string> params;
			params.insert({ "token", results.token.toUtf8().constData() });
			params.insert({ "ignore_certificate", std::to_string(results.ignore_certificate) });

			SetDataSource(std::string(results.location.toUtf8().constData()), params);
		}
	}

	void WorklistWindow::OnCreateWorklist_(void)
	{
		bool succesful;
		QString worklist = QInputDialog::getText(this, tr("Create New Worklist"), tr("Worklist title:"), QLineEdit::Normal, "", &succesful);
		if (succesful && !worklist.isEmpty())
		{
			m_source_.AddWorklistRecord(std::string(worklist.toUtf8().constData()), [this](const bool succesful)
			{
				if (succesful)
				{
					this->RequestWorklistRefresh();
				}
			});
		}
	}

	void WorklistWindow::OnOpenImage_(QString path)
	{
		this->UpdateStatusBar("Loaded file: " + path);
		m_workstation_->openFile(path);
		RequiresTabSwitch(m_workstation_tab_id_);
	}

	void WorklistWindow::OnWorklistRefresh(void)
	{
		m_source_.GetWorklistRecords([this, worklist_model = m_worklist_model_](Data::DataTable& table, int error)
		{
			if (error == 0)
			{
				this->SetWorklistItems(table, worklist_model);
			}
		});
	}
}
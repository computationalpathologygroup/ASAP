#include "WorklistWindow.h"

#include <cctype>
#include <condition_variable>
#include <sstream>
#include <unordered_map>

#include <boost/bind.hpp>
#include <qfiledialog.h>
#include <qinputdialog.h>
#include <qmessagebox.h>
#include <QMimeData>
#include <QtConcurrent\qtconcurrentrun.h>

#include "ExternalSourceDialog.h"
#include "IconCreator.h"
#include "../Serialization/INI.h"
#include "../Misc/StringManipulation.h"

namespace ASAP
{
	WorklistWindow::WorklistWindow(QWidget* parent) :
		CompositeChild(parent),
		m_ui_(new Ui::WorklistWindowLayout),
		m_storage_directory_(boost::filesystem::path("/temp/"), TemporaryDirectoryTracker::GetStandardConfiguration()),
		m_source_(m_storage_directory_)
	{
		m_ui_->setupUi(this);
		SetSlots_();
		SetModels_();
		LoadSettings_();
		UpdatePreviousSources_();

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
					throw std::runtime_error("Loaded source has schema errors or lacks Patient and Study data.");
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

	// Todo: Decouple and create abstraction of schema's
	bool WorklistWindow::CheckSchema_(void)
	{
		if (m_source_.IsInitialized())
		{
			// No schema check is required for a filelist source.
			if (m_source_.GetSourceType() == WorklistSourceInterface::FULL_WORKLIST)
			{
				std::set<std::string> worklist_headers(m_source_.GetWorklistHeaders());
				std::set<std::string> patient_headers(m_source_.GetPatientHeaders());
				std::set<std::string> study_headers(m_source_.GetStudyHeaders());
				std::set<std::string> image_headers(m_source_.GetImageHeaders());

				return	(worklist_headers.find("id") != worklist_headers.end()) &&
						(patient_headers.find("id") != patient_headers.end()) &&
						(study_headers.find("id") != study_headers.end()) &&
						(image_headers.find("id") != image_headers.end()) &&
						(image_headers.find("title") != image_headers.end());
			}
			return true;
		}
		return false;
	}

	std::vector<std::string> WorklistWindow::GetImagesForItem_(const std::string& id, const WorklistModels::ModelEnum& model)
	{
		std::vector<std::string> images;
		if (model > WorklistModels::WORKLISTS)
		{
			if (model == WorklistModels::IMAGES)
			{
				images.push_back(id);
			}
			else
			{
				// Acquires the studies attached to a passed patient id or only selects the passed id as study.
				std::vector<std::string> studies_to_acquire;
				std::condition_variable condition;
				std::mutex completed_flag;
				if (model == WorklistModels::STUDIES)
				{
					studies_to_acquire.push_back(id);
				}
				else if (model == WorklistModels::PATIENTS)
				{
					std::unique_lock<std::mutex> lock(completed_flag);
					m_source_.GetStudyRecords(id, [&studies_to_acquire, &condition](DataTable& table, int error_code)
					{
						if (error_code == 0)
						{
							for (size_t record = 0; record < table.Size(); ++record)
							{
								studies_to_acquire.push_back(*table.At(record, { "id" })[0]);
							}
						}
						condition.notify_one();
					});
					condition.wait(lock);
				}

				// Acquires the images for all available studies, while keeping worklist constraints in mind.
				QModelIndexList selected_worklist(m_ui_->view_worklists->selectionModel()->selectedIndexes());
				QStandardItem* worklist_item(m_models_.worklists->itemFromIndex(selected_worklist[0]));
				std::string worklist_id = !worklist_item->data().isNull() ? worklist_item->data().toList()[0].toString().toStdString() : std::string();
				for (const std::string& study_id : studies_to_acquire)
				{
					std::unique_lock<std::mutex> lock(completed_flag);
					m_source_.GetImageRecords(worklist_id, study_id, [&images, &condition](DataTable& table, int error_code)
					{
						if (error_code == 0)
						{
							for (size_t record = 0; record < table.Size(); ++record)
							{
								images.push_back(*table.At(record, { "id" })[0]);
							}
						}
						condition.notify_one();
					});
					condition.wait(lock);
				}
			}
		}
		return images;
	}

	void WorklistWindow::LoadSettings_(void)
	{
		try
		{
			std::unordered_map<std::string, std::string> values(INI::ParseINI("worklist_config.ini"));

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
			INI::WriteINI("worklist_config.ini", values);
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

		INI::WriteINI("worklist_config.ini", values);
	}

	void WorklistWindow::StopThumbnailLoading_(void)
	{
		m_continue_loading_ = false;
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
		m_models_.SetWorklistItems(DataTable());
		m_models_.SetPatientsItems(DataTable());
		m_models_.SetStudyItems(DataTable());
		m_models_.SetImageItems(DataTable(), this, m_continue_loading_);

		// Resets the view to the Filelist standard.
		WorklistSourceInterface::SourceType type = WorklistSourceInterface::SourceType::FILELIST;

		// Adjusts the GUI to the actual new source, if it was initialized succesfully.
		if (m_source_.IsInitialized())
		{
			type = m_source_.GetSourceType();

			if (type == WorklistSourceInterface::SourceType::FILELIST)
			{
				m_source_.GetImageRecords(std::string(), std::string(), [this, models=&m_models_, continue_loading=&m_continue_loading_](DataTable& table, int error)
				{
					if (error == 0)
					{
						models->SetImageItems(table, this, *continue_loading);
					}
				});
			}
			else if (type == WorklistSourceInterface::SourceType::FULL_WORKLIST)
			{
				std::vector<std::pair<std::set<std::string>, QAbstractItemView*>> headers(
				{
					{ std::set<std::string>(), nullptr },
					{ m_source_.GetPatientHeaders(DataTable::FIELD_SELECTION::VISIBLE), m_ui_->view_patients },
					{ m_source_.GetStudyHeaders(DataTable::FIELD_SELECTION::VISIBLE), m_ui_->view_studies },
					{ std::set<std::string>(), nullptr }
				});

				m_models_.UpdateHeaders(headers);
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

	void WorklistWindow::UpdateWorklist_(const QStandardItem* worklist_item, const std::vector<std::string>& image_list, bool remove)
	{
		QVariantList worklist_variants(worklist_item->data().toList());
		std::string worklist_id(worklist_variants[0].toString().toStdString());
		std::set<std::string> worklist_images;
		for (const std::string& image_id : Misc::Split(worklist_variants[1].toString().toStdString()))
		{
			worklist_images.insert(image_id);
		}

		std::string error_message;
		if (remove)
		{
			error_message = "Unable to delete images.";
			for (const std::string& image_id : image_list)
			{
				worklist_images.erase(image_id);
			}
		}
		else
		{
			error_message = "Unable to add images.";
			for (const std::string& image_id : image_list)
			{
				worklist_images.insert(image_id);
			}
		}
		m_source_.UpdateWorklistRecord(worklist_id, worklist_item->text().toStdString(), worklist_images, [this, error_message](const bool succesful)
		{
			if (!succesful)
			{
				// TODO: Create an emit to create the message box in the main thread
				QMessageBox::question(this, "Error", QString(error_message.data()), QMessageBox::Ok);
			}
			this->RequestWorklistRefresh();
		});
	}

	void WorklistWindow::SetModels_(void)
	{
		m_ui_->view_images->setModel(m_models_.images);
		m_ui_->view_patients->setModel(m_models_.patients);
		m_ui_->view_studies->setModel(m_models_.studies);
		m_ui_->view_worklists->setModel(m_models_.worklists);
	}

	bool WorklistWindow::eventFilter(QObject* obj, QEvent* event)
	{
		if (event->type() == QEvent::Drop)
		{
			static_cast<QDropEvent*>(event)->setDropAction(Qt::DropAction::IgnoreAction);
			OnImageDrop_((QDropEvent*)event);
		}
		return false;
	}

	//================================= Slots =================================//
	
	void WorklistWindow::SetSlots_(void)
	{
		connect(m_models_.worklists,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(OnWorklistClear_(QModelIndex, int, int)));

		connect(m_models_.patients,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(OnPatientsClear_(QModelIndex, int, int)));

		connect(m_models_.studies,
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

		connect(m_ui_->button_open_image,
			&QPushButton::clicked,
			this,
			&WorklistWindow::OnImageSelect_);

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

		QCoreApplication::instance()->installEventFilter(this);
	}

	void WorklistWindow::UpdateImageIcons(void)
	{
		m_models_.images->layoutChanged();
	}

	void WorklistWindow::UpdateStatusBar(const QString& message)
	{
		m_status_bar_access_.lock();
		m_ui_->status_bar->showMessage(message);
		m_status_bar_access_.unlock();
	}

	void WorklistWindow::OnImageDrop_(QDropEvent* drop_event)
	{
		// Identifies the source of the drop event.
		QObject* source(drop_event->source());
		WorklistModels::ModelEnum owner = WorklistModels::WORKLISTS;

		if (source == m_ui_->view_images)
		{
			owner = WorklistModels::IMAGES;
		}
		else if (source == m_ui_->view_studies)
		{
			owner = WorklistModels::STUDIES;
		}
		else if (source == m_ui_->view_patients)
		{
			owner = WorklistModels::PATIENTS;
		}

		// Acquires the additional image ID's.
		if (owner != WorklistModels::WORKLISTS)
		{
			QByteArray encoded = drop_event->mimeData()->data("application/x-qabstractitemmodeldatalist");
			QDataStream stream(&encoded, QIODevice::ReadOnly);
			int row, column;
			stream >> row >> column;

			QStandardItemModel* model(m_models_.GetModel(owner));
			QStandardItem* item = model->itemFromIndex(model->index(row, column));
			QStandardItem* worklist_item = m_models_.worklists->itemFromIndex(m_ui_->view_worklists->indexAt(drop_event->pos()));

			// Checks if the worklist item can be selected, and whether or not it's the statically created "All" entry.
			if (worklist_item && worklist_item->row() > 0)
			{
				std::string item_id(item->data().toString().toStdString());
				UpdateWorklist_(worklist_item, GetImagesForItem_(item_id, owner), false);
			}
		}
	}

	void WorklistWindow::OnImageDelete_(const std::string& id, const WorklistModels::ModelEnum& model)
	{

	}

	void WorklistWindow::OnWorklistClear_(QModelIndex index, int, int)
	{
		m_models_.patients->removeRows(0, m_models_.patients->rowCount());
		m_ui_->view_patients->update();
	}

	void WorklistWindow::OnPatientsClear_(QModelIndex index, int, int)
	{
		m_models_.studies->removeRows(0, m_models_.studies->rowCount());
		m_ui_->view_studies->update();
	}

	void WorklistWindow::OnStudyClear_(QModelIndex index, int, int)
	{
		m_models_.images->removeRows(0, m_models_.images->rowCount());
		m_ui_->view_images->update();
	}

	void WorklistWindow::OnWorklistSelect_(QModelIndex index)
	{
		QStandardItem* item(m_models_.worklists->itemFromIndex(index));
		std::string worklist_id = !item->data().isNull() ? item->data().toList()[0].toString().toStdString() : std::string();

		QStandardItemModel* patient_model	= m_models_.patients;
		QTableView* patient_view			= m_ui_->view_patients;
		m_source_.GetPatientRecords(worklist_id, [this, patient_model, patient_view](DataTable table, int error)
		{
			if (error == 0)
			{
				m_models_.SetPatientsItems(table);
				patient_view->update();
			}
		});
	}

	void WorklistWindow::OnPatientSelect_(QModelIndex index)
	{
		QStandardItem* item(m_models_.patients->itemFromIndex(index));
		std::string patient_id(item->data().toString().toStdString());

		QStandardItemModel* study_model = m_models_.studies;
		QTableView* study_view = m_ui_->view_studies;
		m_source_.GetStudyRecords(patient_id, [this, study_model, study_view](DataTable table, int error)
		{
			if (error == 0)
			{
				m_models_.SetStudyItems(table);
				study_view->update();
			}
		});
	}

	void WorklistWindow::OnStudySelect_(QModelIndex index)
	{
		QModelIndexList selected_worklist(m_ui_->view_worklists->selectionModel()->selectedIndexes());

		QStandardItem* study_item(m_models_.studies->itemFromIndex(index));
		QStandardItem* worklist_item(m_models_.worklists->itemFromIndex(selected_worklist[0]));

		std::string study_id(study_item->data().toString().toStdString());
		std::string worklist_id = !worklist_item->data().isNull() ? worklist_item->data().toList()[0].toString().toStdString() : std::string();

		m_source_.GetImageRecords(worklist_id, study_id, [this, models=&m_models_, continue_loading=&m_continue_loading_, image_view=m_ui_->view_images](DataTable table, int error)
		{
			if (error == 0)
			{
				models->SetImageItems(table, this, *continue_loading);
				image_view->update();
			}
		});
	}

	void WorklistWindow::OnImageSelect_(const bool checked)
	{
		if (m_workstation_)
		{
			QModelIndexList selected(m_ui_->view_images->selectionModel()->selectedIndexes());

			for (QModelIndex& index : selected)
			{
				QStandardItem* image(m_models_.images->itemFromIndex(index));
				std::string image_index(image->data().toString().toStdString());

				m_ui_->status_bar->showMessage("Loading image: 0%");
				auto image_loading([this](const boost::filesystem::path& filepath)
				{
					if (filepath.has_filename())
					{
						this->RequestOpenImage(QString::fromStdString(filepath.string()));
					}
					else
					{
						this->UpdateStatusBar("Failed to load image.");
					}
				});

				auto acquisition_tracking([this, bar = m_ui_->status_bar](const uint8_t progress)
				{
					if (bar->currentMessage().endsWith("%"))
					{
						this->UpdateStatusBar("Loading image: " + QString(std::to_string(progress).data()) + "%");
					}
				});

				m_source_.GetImageFile(image_index, image_loading, acquisition_tracking);
			}
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
			SetDataSource(dialog->selectedFiles()[0].toStdString(), std::unordered_map<std::string, std::string>());
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
			SetDataSource(dialog->selectedFiles()[0].toStdString(), std::unordered_map<std::string, std::string>());
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
			params.insert({ "token", results.token.toStdString() });
			params.insert({ "ignore_certificate", std::to_string(results.ignore_certificate) });

			SetDataSource(std::string(results.location.toStdString()), params);
		}
	}

	void WorklistWindow::OnCreateWorklist_(void)
	{
		bool succesful;
		QString worklist = QInputDialog::getText(this, tr("Create New Worklist"), tr("Worklist title:"), QLineEdit::Normal, "", &succesful);
		if (succesful && !worklist.isEmpty())
		{
			m_source_.AddWorklistRecord(std::string(worklist.toStdString()), [this](const bool succesful)
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
		m_source_.GetWorklistRecords([models=&m_models_](DataTable& table, int error)
		{
			if (error == 0)
			{
				models->SetWorklistItems(table);
			}
		});
	}
}
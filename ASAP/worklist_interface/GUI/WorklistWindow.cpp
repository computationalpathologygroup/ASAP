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
#include <QtConcurrent/qtconcurrentrun.h>
#include "core/stringconversion.h"

#ifdef BUILD_GRANDCHALLENGE_INTERFACE
#include "ExternalSourceDialog.h"
#endif 

#include "IconCreator.h"

namespace ASAP
{
	WorklistWindow::WorklistWindow(QWidget* parent) :
		CompositeChild(parent),
		m_ui(new Ui::WorklistWindowLayout),
		m_storage_directory(boost::filesystem::path("/temp/"), TemporaryDirectoryTracker::getStandardConfiguration()),
		m_source(m_storage_directory)
	{
		m_ui->setupUi(this);
#ifdef BUILD_GRANDCHALLENGE_INTERFACE
		m_ui->action_open_external->setVisible(true);
#endif
		m_settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
		setSlots();
		setModels();
		readSettings();
		updatePreviousSources();
		updateSourceViews();
	}

	WorklistWindow::~WorklistWindow(void)
	{
		stopThumbnailLoading();
		writeSettings();
	}

	void WorklistWindow::attachWorkstation(ASAP_Window& workstation, const int tab_id)
	{
		m_workstation			= &workstation;
		m_workstation_tab_id	= tab_id;
	}

	void WorklistWindow::setDataSource(const std::string source_path, const std::unordered_map<std::string, std::string> parameters)
	{
		setDataSource(SourceProxy::serializeSource(source_path, parameters));
	}

	void WorklistWindow::setDataSource(const std::string& source)
	{
		if (!m_source.isInitialized() || source != m_source.getCurrentSource())
		{
			std::pair<std::string, std::unordered_map<std::string, std::string>> deserialized_source(SourceProxy::deserializeSource(source));

			// Halts any leftover thumbnail loading.
			stopThumbnailLoading();

			try
			{
				// Attempts to load the data source and then confirms it has the required fields for the UI to function.
				m_ui->status_bar->showMessage("Loading source: " + QString(deserialized_source.first.data()));
				m_source.loadSource(source);

				if (!m_source.isInitialized())
				{
					throw std::runtime_error("Unable to load source: " + deserialized_source.first);
				}
				if (!checkSchema())
				{
					m_source.close();
					throw std::runtime_error("Loaded source has schema errors or lacks Patient and Study data.");
				}

				updatePreviousSources();
				updateSourceViews();
				if (m_source.getSourceType() == WorklistSourceInterface::SourceType::FILELIST) {
					m_workstation->openFile(QString::fromStdString(SourceProxy::deserializeSource(m_source.getCurrentSource()).first), "default");
					requiresTabSwitch(m_workstation_tab_id);
				}
				m_ui->status_bar->showMessage("Succesfully loaded source: " + QString(deserialized_source.first.data()));
			}
			catch (const std::exception& e)
			{
				m_ui->status_bar->showMessage("Unable to load source: " + QString(deserialized_source.first.data()));
				QMessageBox::question(this, "Error", e.what(), QMessageBox::Ok);
			}
		}
	}

	// Todo: Decouple and create abstraction of schema's
	bool WorklistWindow::checkSchema(void)
	{
		if (m_source.isInitialized())
		{
			// No schema check is required for a filelist source.
			if (m_source.getSourceType() == WorklistSourceInterface::FULL_WORKLIST)
			{
				std::set<std::string> worklist_headers(m_source.getWorklistHeaders());
				std::set<std::string> patient_headers(m_source.getPatientHeaders());
				std::set<std::string> study_headers(m_source.getStudyHeaders());
				std::set<std::string> image_headers(m_source.getImageHeaders());

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

	std::vector<std::string> WorklistWindow::getImagesForItem(const std::string& id, const WorklistModels::ModelEnum model)
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
					m_source.getStudyRecords(id, [&studies_to_acquire, &condition](DataTable& table, int error_code)
					{
						if (error_code == 0)
						{
							for (size_t record = 0; record < table.size(); ++record)
							{
								studies_to_acquire.push_back(*table.at(record, { "id" })[0]);
							}
						}
						condition.notify_one();
					});
					condition.wait(lock);
				}

				// Acquires the images for all available studies, while keeping worklist constraints in mind.
				QModelIndexList selected_worklist(m_ui->view_worklists->selectionModel()->selectedIndexes());
				QStandardItem* worklist_item(m_models.worklists->itemFromIndex(selected_worklist[0]));
				std::string worklist_id = !worklist_item->data().isNull() ? worklist_item->data().toList()[0].toString().toStdString() : std::string();
				for (const std::string& study_id : studies_to_acquire)
				{
					std::unique_lock<std::mutex> lock(completed_flag);
					m_source.getImageRecords(worklist_id, study_id, [&images, &condition](DataTable& table, int error_code)
					{
						if (error_code == 0)
						{
							for (size_t record = 0; record < table.size(); ++record)
							{
								images.push_back(*table.at(record, { "id" })[0]);
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

	void WorklistWindow::readSettings(void)
	{
		std::vector<std::string> sources;
		m_settings->beginGroup("WorklistInterface");
		int sourcesSize = m_settings->beginReadArray("sources");
		for (int i = 0; i < sourcesSize; ++i) {
			m_settings->setArrayIndex(i);
			sources.push_back(m_settings->value("source").toString().toStdString());
		}
		m_settings->endArray();
		m_settings->endGroup();
		if (!sources.empty()) {
			m_source.setSourceInformation(sources[0], sources);
		}
		else {
			m_source.setSourceInformation("", std::vector<std::string>());
		}
	}
	
	void WorklistWindow::writeSettings(void)
	{
		m_settings->beginGroup("WorklistInterface");
		m_settings->beginWriteArray("sources");
		std::deque<std::string> sources = m_source.getPreviousSources();
		for (unsigned int i = 0; i < sources.size(); ++i)
		{
			m_settings->setArrayIndex(i);
			m_settings->setValue("source", QString::fromStdString(sources[i]));
		}
		m_settings->endArray();
		m_settings->endGroup();
	}

	void WorklistWindow::stopThumbnailLoading(void)
	{
		if (m_thumbnail_loader) {
			m_thumbnail_loader->cancel();
			m_thumbnail_loader->waitForFinished();
		}
		QApplication::processEvents();
	}

	void WorklistWindow::updatePreviousSources(void)
	{
		for (const std::unique_ptr<QAction>& action : m_history_actions)
		{
			m_ui->menu_history->removeAction(action.get());
		}
		m_history_actions.clear();

		auto previous_source(m_source.getPreviousSources());
		for (const std::string& prev_source : previous_source)
		{
			std::pair<std::string, std::unordered_map<std::string, std::string>> source_parameters(SourceProxy::deserializeSource(prev_source));

			m_history_actions.push_back(std::unique_ptr<QAction>(new QAction(QString(source_parameters.first.data()), this)));
			m_ui->menu_history->addAction(m_history_actions.back().get());

			connect(m_history_actions.back().get(),
				&QAction::triggered,
				this,
				[action = m_history_actions.back().get(), this, source_parameters](bool checked)
				{
					this->setDataSource(source_parameters.first, source_parameters.second);
				});
		}
	}

	void WorklistWindow::updateSourceViews(void)
	{
		// Clears all previous source information.
		m_models.setWorklistItems(DataTable());
		m_models.setPatientsItems(DataTable());
		m_models.setStudyItems(DataTable());
		m_models.setImageItems(DataTable(), this);

		// Resets the view to the Filelist standard.
		WorklistSourceInterface::SourceType type = WorklistSourceInterface::SourceType::FILELIST;

		// Adjusts the GUI to the actual new source, if it was initialized succesfully.
		if (m_source.isInitialized())
		{
			type = m_source.getSourceType();

			if (type == WorklistSourceInterface::SourceType::DIRECTORY)
			{
				m_source.getImageRecords(std::string(), std::string(), [this, models=&m_models](DataTable& table, int error)
				{
					if (error == 0)
					{
						m_thumbnail_loader = models->setImageItems(table, this);
					}
				});
			}
			else if (type == WorklistSourceInterface::SourceType::FULL_WORKLIST)
			{
				std::vector<std::pair<std::set<std::string>, QAbstractItemView*>> headers(
				{
					{ std::set<std::string>(), nullptr },
					{ m_source.getPatientHeaders(DataTable::FIELD_SELECTION::VISIBLE), m_ui->view_patients },
					{ m_source.getStudyHeaders(DataTable::FIELD_SELECTION::VISIBLE), m_ui->view_studies },
					{ std::set<std::string>(), nullptr }
				});

				m_models.updateHeaders(headers);
				this->requestWorklistRefresh();
			}
		}

		// Default is regarded as the FILELIST sourcetype
		switch (type)
		{
			case WorklistSourceInterface::SourceType::FULL_WORKLIST:
				m_ui->label_worklists->setVisible(true);
				m_ui->label_patients->setVisible(true);
				m_ui->label_studies->setVisible(true);
				m_ui->view_worklists->setVisible(true);
				m_ui->view_patients->setVisible(true);
				m_ui->view_studies->setVisible(true);
				m_ui->button_create_worklist->setVisible(true);
				break;
			default:
				m_ui->label_worklists->setVisible(false);
				m_ui->label_patients->setVisible(false);
				m_ui->label_studies->setVisible(false);
				m_ui->view_worklists->setVisible(false);
				m_ui->view_patients->setVisible(false);
				m_ui->view_studies->setVisible(false);
				m_ui->button_create_worklist->setVisible(false);
				break;
		}
	}

	void WorklistWindow::updateWorklist(const QStandardItem* worklist_item, const std::vector<std::string>& image_list, bool remove)
	{
		if (worklist_item && !worklist_item->data().isNull())
		{
			QVariantList worklist_variants(worklist_item->data().toList());
			std::string worklist_id(worklist_variants[0].toString().toStdString());
			std::set<std::string> worklist_images;
			std::vector<std::string> imageIds;
			core::split(worklist_variants[1].toString().toStdString(), imageIds, ",");
			for (const std::string& image_id : imageIds)
			{
				worklist_images.insert(image_id);
			}

			QString error_message;
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
			m_source.updateWorklistRecord(worklist_id, worklist_item->text().toStdString(), worklist_images, [this, error_message](const bool succesful)
			{
				if (!succesful)
				{
					this->showMessageBox(error_message);
				}
				else
				{

				}
				this->requestWorklistRefresh();
			});
		}
	}

	void WorklistWindow::setModels(void)
	{
		m_ui->view_images->setModel(m_models.images);
		m_ui->view_patients->setModel(m_models.patients);
		m_ui->view_studies->setModel(m_models.studies);
		m_ui->view_worklists->setModel(m_models.worklists);
	}

	void WorklistWindow::keyPressEvent(QKeyEvent* event)
	{
		if (event->key() == Qt::Key::Key_Delete)
		{
			QWidget* origin(this->focusWidget());
			if (origin &&
				(origin == m_ui->view_worklists ||
				 origin == m_ui->view_patients ||
				 origin == m_ui->view_studies ||
				 origin == m_ui->view_images))
			{
				QAbstractItemView* view((QAbstractItemView*)origin);
				QStandardItemModel* model((QStandardItemModel*)view->model());
				WorklistModels::ModelEnum model_enum(m_models.getModelEnum(model));
				QStandardItem* item(model->itemFromIndex(view->selectionModel()->selectedIndexes()[0]));

				if (model_enum == WorklistModels::WORKLISTS && item->row() > 0)
				{
					m_source.deleteWorklistRecord(item->data().toList()[0].toString().toStdString(), [this](bool succesful)
					{
						if (!succesful)
						{
							this->showMessageBox("Unable to delete Worklist.");
						}
						else
						{
							this->requestWorklistRefresh();
						}
					});
				}
				else
				{
					QStandardItem* worklist_item(m_models.worklists->itemFromIndex(m_ui->view_worklists->selectionModel()->selectedIndexes()[0]));
					updateWorklist(worklist_item, getImagesForItem(item->data().toString().toStdString(), model_enum), true);
				}
			}
		}
	}

	bool WorklistWindow::eventFilter(QObject* obj, QEvent* event)
	{
		if (event->type() == QEvent::Drop)
		{
			static_cast<QDropEvent*>(event)->setDropAction(Qt::DropAction::IgnoreAction);
			onImageDrop((QDropEvent*)event);
		}
		return false;
	}

	//================================= Slots =================================//
	
	void WorklistWindow::setSlots(void)
	{
		connect(m_models.worklists,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(onWorklistClear(QModelIndex, int, int)));

		connect(m_models.patients,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(onPatientsClear(QModelIndex, int, int)));

		connect(m_models.studies,
			SIGNAL(rowsRemoved(QModelIndex, int, int)),
			this,
			SLOT(onStudyClear(QModelIndex, int, int)));

		connect(m_ui->view_worklists,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(onWorklistSelect(QModelIndex)));

		connect(m_models.worklists,
			&QStandardItemModel::itemChanged,
			this,
			&WorklistWindow::onWorklistNameChange);

		connect(m_ui->view_patients,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(onPatientSelect(QModelIndex)));

		connect(m_ui->view_studies,
			SIGNAL(clicked(QModelIndex)),
			this,
			SLOT(onStudySelect(QModelIndex)));

		connect(m_ui->view_images,
			SIGNAL(doubleClicked(QModelIndex)),
			this,
			SLOT(onImageDoubleClicked(QModelIndex)));

		connect(m_ui->button_open_image,
			&QPushButton::clicked,
			this,
			&WorklistWindow::onImageSelect);

		connect(m_ui->button_create_worklist,
			&QPushButton::clicked,
			this,
			&WorklistWindow::onCreateWorklist);

		connect(m_ui->action_open_file,
			&QAction::triggered,
			this,
			&WorklistWindow::onSelectFileSource);

		connect(m_ui->action_open_folder,
			&QAction::triggered,
			this,
			&WorklistWindow::onSelectFolderSource);
#ifdef BUILD_GRANDCHALLENGE_INTERFACE
		connect(m_ui->action_open_external,
			&QAction::triggered,
			this,
			&WorklistWindow::onSelectExternalSource);
#endif
		connect(this,
			&WorklistWindow::requestStatusBarUpdate,
			this,
			&WorklistWindow::updateStatusBar);
			
		connect(this,
			&WorklistWindow::requestOpenImage,
			this,
			&WorklistWindow::onOpenImage);

		connect(this,
			&WorklistWindow::requestWorklistRefresh,
			this,
			&WorklistWindow::onWorklistRefresh);

		connect(this,
			&WorklistWindow::showMessageBox,
			this,
			&WorklistWindow::onShowMessageBox);

		QCoreApplication::instance()->installEventFilter(this);
	}

	void WorklistWindow::updateImageIcons(int itemRow, const QIcon& newIcon)
	{
		if (itemRow >= 0) {
			m_models.images->item(itemRow)->setIcon(newIcon);
		}
		else {
			m_models.images->layoutChanged();
		}
	}

	void WorklistWindow::updateStatusBar(const QString& message)
	{
		m_status_bar_access.lock();
		m_ui->status_bar->showMessage(message);
		m_status_bar_access.unlock();
	}

	void WorklistWindow::onShowMessageBox(const QString message)
	{
		QMessageBox::question(this, "Error", message, QMessageBox::Ok);
	}

	void WorklistWindow::onImageDrop(QDropEvent* drop_event)
	{
		// Identifies the source of the drop event.
		QObject* source(drop_event->source());
		WorklistModels::ModelEnum owner = m_models.getModelEnum((QStandardItemModel*)static_cast<QAbstractItemView*>(source)->model());

		// Acquires the additional image ID's.
		if (owner != WorklistModels::WORKLISTS)
		{
			QByteArray encoded = drop_event->mimeData()->data("application/x-qabstractitemmodeldatalist");
			QDataStream stream(&encoded, QIODevice::ReadOnly);
			int row, column;
			stream >> row >> column;

			QStandardItemModel* model(m_models.getModel(owner));
			QStandardItem* item = model->itemFromIndex(model->index(row, column));
			QStandardItem* worklist_item = m_models.worklists->itemFromIndex(m_ui->view_worklists->indexAt(drop_event->pos()));

			// Checks if the worklist item can be selected, and whether or not it's the statically created "All" entry.
			if (worklist_item && worklist_item->row() > 0)
			{
				std::string item_id(item->data().toString().toStdString());
				updateWorklist(worklist_item, getImagesForItem(item_id, owner), false);
			}
		}
	}

	void WorklistWindow::onWorklistClear(QModelIndex index, int, int)
	{
		m_models.patients->removeRows(0, m_models.patients->rowCount());
		m_ui->view_patients->update();
	}

	void WorklistWindow::onPatientsClear(QModelIndex index, int, int)
	{
		m_models.studies->removeRows(0, m_models.studies->rowCount());
		m_ui->view_studies->update();
	}

	void WorklistWindow::onStudyClear(QModelIndex index, int, int)
	{
		m_models.images->removeRows(0, m_models.images->rowCount());
		m_ui->view_images->update();
	}

	void WorklistWindow::onWorklistSelect(QModelIndex index)
	{
		QStandardItem* item(m_models.worklists->itemFromIndex(index));
		std::string worklist_id = !item->data().isNull() ? item->data().toList()[0].toString().toStdString() : std::string();

		QStandardItemModel* patient_model	= m_models.patients;
		QTableView* patient_view			= m_ui->view_patients;
		m_source.getPatientRecords(worklist_id, [this, patient_model, patient_view](DataTable table, int error)
		{
			if (error == 0)
			{
				m_models.setPatientsItems(table);
				patient_view->update();
			}
		});
	}

	void WorklistWindow::onPatientSelect(QModelIndex index)
	{
		QStandardItem* item(m_models.patients->itemFromIndex(index));
		std::string patient_id(item->data().toString().toStdString());

		QStandardItemModel* study_model = m_models.studies;
		QTableView* study_view = m_ui->view_studies;
		m_source.getStudyRecords(patient_id, [this, study_model, study_view](DataTable table, int error)
		{
			if (error == 0)
			{
				m_models.setStudyItems(table);
				study_view->update();
			}
		});
	}

	void WorklistWindow::onStudySelect(QModelIndex index)
	{
		QModelIndexList selected_worklist(m_ui->view_worklists->selectionModel()->selectedIndexes());

		QStandardItem* study_item(m_models.studies->itemFromIndex(index));
		QStandardItem* worklist_item(m_models.worklists->itemFromIndex(selected_worklist[0]));

		std::string study_id(study_item->data().toString().toStdString());
		std::string worklist_id = !worklist_item->data().isNull() ? worklist_item->data().toList()[0].toString().toStdString() : std::string();

		m_source.getImageRecords(worklist_id, study_id, [this, models=&m_models, image_view=m_ui->view_images](DataTable table, int error)
		{
			if (error == 0)
			{
				stopThumbnailLoading();
				m_thumbnail_loader = models->setImageItems(table, this);
				image_view->update();
			}
		});
	}

	void WorklistWindow::onImageDoubleClicked(QModelIndex index)
	{
		if (m_workstation)
		{
			getImageFromIndex(index);
		}
	}

	void WorklistWindow::onImageSelect(const bool checked)
	{
		if (m_workstation)
		{
			QModelIndexList selected(m_ui->view_images->selectionModel()->selectedIndexes());

			for (QModelIndex& index : selected)
			{
				getImageFromIndex(index);
			}
		}
	}

	void WorklistWindow::getImageFromIndex(const QModelIndex& index)
	{
		QStandardItem* image(m_models.images->itemFromIndex(index));
		std::string image_index(image->data().toString().toStdString());

		m_ui->status_bar->showMessage("Loading image: 0%");
		auto image_loading([this](const boost::filesystem::path& filepath)
			{
				if (filepath.has_filename())
				{
					this->requestOpenImage(QString::fromStdString(filepath.string()));
				}
				else
				{
					this->updateStatusBar("Failed to load image.");
				}
			});

		auto acquisition_tracking([this, bar = m_ui->status_bar](const uint8_t progress)
		{
			if (bar->currentMessage().endsWith("%"))
			{
				this->updateStatusBar("Loading image: " + QString(std::to_string(progress).data()) + "%");
			}
		});

		m_source.getImageFile(image_index, image_loading, acquisition_tracking);
	}

	void WorklistWindow::onSelectFileSource(bool checked)
	{
		QList<QString> filename_and_factory = m_workstation->getFileNameAndFactory();
		if (!filename_and_factory.isEmpty() && !filename_and_factory[0].isEmpty())
		{
			setDataSource(filename_and_factory[0].toStdString(), std::unordered_map<std::string, std::string>());
		}
	}
	
	void WorklistWindow::onSelectFolderSource(bool checked)
	{
		QFileDialog* dialog = new QFileDialog(this);
		dialog->setFileMode(QFileDialog::Directory);
		dialog->exec();
		QStringList names = dialog->selectedFiles();

		if (names.size() > 0)
		{
			setDataSource(dialog->selectedFiles()[0].toStdString(), std::unordered_map<std::string, std::string>());
		}
	}

	void WorklistWindow::onSelectExternalSource(bool checked)
	{
#ifdef BUILD_GRANDCHALLENGE_INTERFACE
		ExternalSourceDialog* dialog = new ExternalSourceDialog(this);
		dialog->exec();

		if (dialog->hasValidCredentials())
		{
			ExternalSourceDialog::SourceDialogResults results(dialog->getLoginDetails());

			std::unordered_map<std::string, std::string> params;
			params.insert({ "token", results.token.toStdString() });
			params.insert({ "ignore_certificate", std::to_string(results.ignore_certificate) });

			setDataSource(std::string(results.location.toStdString()), params);
		}
#endif
	}

	void WorklistWindow::onOpenImage(QString path)
	{
		this->updateStatusBar("Loaded file: " + path);
		m_workstation->openFile(path);
		requiresTabSwitch(m_workstation_tab_id);
	}

	void ASAP::WorklistWindow::onIconDoubleClicked(const QModelIndex& index)
	{
		getImageFromIndex(index);
	}

	void WorklistWindow::onCreateWorklist(void)
	{
		bool succesful;
		QString worklist = QInputDialog::getText(this, tr("Create New Worklist"), tr("Worklist title:"), QLineEdit::Normal, "", &succesful);
		if (succesful && !worklist.isEmpty())
		{
			m_source.addWorklistRecord(std::string(worklist.toStdString()), [this](const bool succesful)
			{
				if (succesful)
				{
					this->requestWorklistRefresh();
				}
			});
		}
	}

	void WorklistWindow::onWorklistNameChange(QStandardItem* item)
	{
		updateWorklist(item, { }, false);
	}

	void WorklistWindow::onWorklistRefresh(void)
	{
		m_source.getWorklistRecords([models=&m_models](DataTable& table, int error)
		{
			if (error == 0)
			{
				models->setWorklistItems(table);
			}
		});
	}
}
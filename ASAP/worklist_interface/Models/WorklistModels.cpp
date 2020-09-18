#include "WorklistModels.h"

#include <stdexcept>
#include <cctype>

#include "../GUI/WorklistWindow.h"
#include "../Misc/StringConversions.h"

namespace ASAP
{
	WorklistModels::WorklistModels(void) : images(new QStandardItemModel(0, 0)), patients(new QStandardItemModel(0, 0)), studies(new QStandardItemModel(0, 0)), worklists(new QStandardItemModel(0, 0))
	{
	}

	QStandardItemModel* WorklistModels::GetModel(const ModelEnum model)
	{
		switch (model)
		{
			case WORKLISTS: return worklists;
			case PATIENTS:	return patients;
			case STUDIES:	return studies;
			case IMAGES:	return images;
		}
		return nullptr;
	}

	WorklistModels::ModelEnum WorklistModels::GetModelEnum(QStandardItemModel* model)
	{
		if (model == worklists)
		{
			return WORKLISTS;
		}
		else if (model == patients)
		{
			return PATIENTS;
		}
		else if (model == studies)
		{
			return STUDIES;
		}
		else if (model == images)
		{
			return IMAGES;
		}
		throw std::runtime_error("Model pointer doesn't match with any available model.");
	}

	void WorklistModels::SetWorklistItems(const DataTable& items)
	{
		if (worklists->rowCount() == 0)
		{
			QStandardItem* model_item(new QStandardItem("All"));
			worklists->setItem(worklists->rowCount(), 0, model_item);
		}

		worklists->removeRows(1, worklists->rowCount() - 1);
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, std::vector<std::string>{ "id", "title", "images" }));
			QStandardItem* model_item(new QStandardItem(QString(record[1]->data())));
			
			model_item->setData(QVariant(QList<QString>({ QString(record[0]->data()), QString(record[2]->data()) })));
			worklists->setItem(worklists->rowCount(), 0, model_item);
		}
	}

	void WorklistModels::SetPatientsItems(const DataTable& items)
	{
		patients->removeRows(0, patients->rowCount());
		patients->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, DataTable::FIELD_SELECTION::VISIBLE));
			for (size_t field = 0; field < record.size(); ++field)
			{
				QStandardItem* model_item = new QStandardItem(QString(record[field]->data()));
				model_item->setData(QVariant(items.At(item, { "id" })[0]->data()));
				patients->setItem(item, field, model_item);
			}
		}
	}

	void WorklistModels::SetStudyItems(const DataTable& items)
	{
		studies->removeRows(0, studies->rowCount());
		studies->setRowCount(items.Size());
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, DataTable::FIELD_SELECTION::VISIBLE));
			for (size_t field = 0; field < record.size(); ++field)
			{
				QStandardItem* model_item = new QStandardItem(QString(record[field]->data()));
				model_item->setData(QVariant(items.At(item, { "id" })[0]->data()));
				studies->setItem(item, field, model_item);
			}
		}
	}

	std::unique_ptr<QFutureWatcher<void> > WorklistModels::SetImageItems(const DataTable& items, WorklistWindow* window)
	{
		images->removeRows(0, images->rowCount());

		// Creates placeholder items
		images->setRowCount(items.Size());
		QImage image(200, 200, QImage::Format::Format_BGR30);
		image.fill(Qt::white);
		QIcon placeholder = QIcon(QPixmap::fromImage(image));
		QList < std::pair<int, std::string> > index_locations;
		for (size_t item = 0; item < items.Size(); ++item)
		{
			std::vector<const std::string*> record(items.At(item, { "id", "title" , "location"}));
			
			QStandardItem* standard_item(new QStandardItem(placeholder, QString(record[1]->data())));
			standard_item->setData(QVariant(QString(record[0]->data())));
			images->setItem(item, 0, standard_item);
			index_locations.push_back(std::pair<int, std::string>(static_cast<int>(item), *record[2]));

		}
		auto icon_connection = QObject::connect(&m_creator,
			&IconCreator::RequiresItemRefresh,
			window,
			&WorklistWindow::updateImageIcons);
		std::function<bool(const std::pair<int, std::string> & index_location)> create_icons = [creator = &m_creator, total_size = items.Size(), window = window](const std::pair<int, std::string>& index_location) -> bool	{
			bool valid = creator->InsertIcon(index_location);
			return valid;
		};
		QFuture<bool> future = QtConcurrent::mapped(index_locations, create_icons);
		std::unique_ptr<QFutureWatcher<void> > future_watcher(new QFutureWatcher<void>);
		QObject::connect(&(*future_watcher), &QFutureWatcher<void>::progressValueChanged, [=](int pv) {window->updateStatusBar("Loading thumbnail " + QString::fromStdString(std::to_string(pv)) + " of " + QString::fromStdString(std::to_string(items.Size()))); });
		QObject::connect(&(*future_watcher), &QFutureWatcher<void>::finished, [=]() {window->updateStatusBar("Finished loading thumbnails."); QObject::disconnect(icon_connection); });
		future_watcher->setFuture(future);
		return future_watcher;
	}

	void WorklistModels::UpdateHeaders(std::vector<std::pair<std::set<std::string>, QAbstractItemView*>>& header_view_couple)
	{
		std::vector<QStandardItemModel*> models({ worklists, patients, studies, images });
		for (size_t m = 0; m < models.size(); ++m)
		{
			if (header_view_couple[m].second)
			{
				SetHeaders_(header_view_couple[m].first, models[m], header_view_couple[m].second);
			}
		}
	}

	void WorklistModels::SetHeaders_(const std::set<std::string> headers, QStandardItemModel* model, QAbstractItemView* view)
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
}
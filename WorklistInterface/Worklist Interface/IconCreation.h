#pragma once
#include <string>
#include <vector>

#include <QStandardItemModel>
#include <qlistview.h>

#include <Data/DataTable.h>

namespace ASAP::Worklist::GUI
{
	void CreateIcons(const DataTable& image_items, QStandardItemModel* image_model, QListView* view, const size_t size);
	QIcon CreateIcon(const std::string& filepath, const size_t size);
}
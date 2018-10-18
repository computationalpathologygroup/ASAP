#pragma once
#include <string>
#include <vector>

#include <QStandardItemModel>
#include <qlistview.h>
#include <qstatusbar.h>

#include <Data/DataTable.h>

namespace ASAP::Worklist::GUI
{
	void InsertIcons(const DataTable& image_items, QStandardItemModel* image_model, QStatusBar* status_bar, const size_t size);
	QIcon CreateIcon(const std::string& filepath, const size_t size);
}
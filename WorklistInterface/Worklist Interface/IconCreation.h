#pragma once
#include <string>
#include <vector>

#include <QStandardItemModel>

namespace ASAP::Worklist::GUI
{
	void CreateIcons(const std::vector<std::pair<QStandardItem*, std::string>>& files, const size_t size);
	QIcon CreateIcon(const std::string& filepath, const size_t size);
}
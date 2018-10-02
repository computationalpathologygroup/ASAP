#include "CompositeWindow.h"

namespace ASAP::Worklist::GUI
{
	CompositeWindow::CompositeWindow(QWidget* parent) : QMainWindow(parent), m_ui_(new Ui::CompositeWindowLayout)
	{
		m_ui_->setupUi(this);
	}

	void CompositeWindow::AddTab(QMainWindow* window, const std::string tab_name)
	{
		m_ui_->tabWidget->addTab(window, QString(tab_name.data()));
	}
}
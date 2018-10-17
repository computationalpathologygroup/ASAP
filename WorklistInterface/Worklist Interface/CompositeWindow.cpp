#include "CompositeWindow.h"

#include "CompositeChild.h"

namespace ASAP::Worklist::GUI
{
	CompositeWindow::CompositeWindow(QWidget* parent) : QMainWindow(parent), m_ui_(new Ui::CompositeWindowLayout)
	{
		m_ui_->setupUi(this);
		m_ui_->menu_bar->deleteLater();

		connect(m_ui_->tabWidget,
				SIGNAL(currentChanged(int)),
				this,
				SLOT(OnTabChange_(int)));
	}

	void CompositeWindow::AddTab(QMainWindow* window, const std::string tab_name)
	{
		m_children_.push_back(window);
		m_mapped_children_.insert({tab_name, window});

		m_ui_->tabWidget->addTab(window, QString(tab_name.data()));
	}

	void CompositeWindow::OnTabChange_(int index)
	{
		CompositeChild* child_window = dynamic_cast<CompositeChild*>(m_children_[index]);
		if (child_window)
		{
			m_ui_->menu_bar = child_window->GetMenuElement();
		}
	}
}
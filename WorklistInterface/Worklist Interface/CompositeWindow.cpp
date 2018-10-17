#include "CompositeWindow.h"

#include "CompositeChild.h"

namespace ASAP::Worklist::GUI
{
	CompositeWindow::CompositeWindow(QWidget* parent) : QMainWindow(parent), m_ui_(new Ui::CompositeWindowLayout), m_current_child_(-1)
	{
		m_ui_->setupUi(this);
		m_ui_->menu_bar->deleteLater();

		SetSlots_();
	}

	void CompositeWindow::AddTab(QMainWindow* window, const std::string tab_name)
	{
		m_children_.push_back(window);
		m_mapped_children_.insert({tab_name, window});

		m_ui_->tabWidget->addTab(window, QString(tab_name.data()));
	}

	void CompositeWindow::SetSlots_(void)
	{
		connect(m_ui_->tabWidget,
			SIGNAL(currentChanged(int)),
			this,
			SLOT(OnTabChange_(int)));
	}

	void CompositeWindow::OnTabChange_(int index)
	{
		if (m_current_child_ > -1)
		{
			CompositeChild* previous_window = dynamic_cast<CompositeChild*>(m_children_[m_current_child_]);
			if (previous_window)
			{
				previous_window->setMenuBar(m_ui_->menu_bar);
			}
		}
		
		if (index > -1)
		{
			CompositeChild* current_window = dynamic_cast<CompositeChild*>(m_children_[index]);
			if (current_window)
			{
				this->setMenuBar(current_window->GetMenuElement());
			}
		}

		m_current_child_ = index;
	}
}
#include "CompositeWindow.h"

#include <qshortcut.h>

namespace ASAP::GUI
{
	CompositeWindow::CompositeWindow(QWidget* parent) : QMainWindow(parent), m_ui_(new Ui::CompositeWindowLayout), m_current_child_(-1)
	{
		m_ui_->setupUi(this);
		m_ui_->menu_bar->deleteLater();

		SetSlots_();
	}

	int CompositeWindow::AddTab(QMainWindow* window, const std::string tab_name)
	{
		m_children_.push_back(window);
		m_mapped_children_.insert({tab_name, m_children_.size() - 1});
		int id = m_ui_->tabWidget->addTab(window, QString(tab_name.data()));

		// Connects a TabRequest signal if the added tab has implemented the CompositeChild interface.
		CompositeChild* child = dynamic_cast<CompositeChild*>(window);
		if (child)
		{
			connect(child,
					&CompositeChild::RequiresTabSwitch,
					this,
					&CompositeWindow::OnTabRequest_);
		}

		return id;
	}

	int CompositeWindow::AddTab(CompositeChild* window, const std::string tab_name, std::vector<ShortcutAction>& shortcuts)
	{
		for (ShortcutAction& shortcut : shortcuts)
		{
			RegisterKeySequence_(shortcut);
		}
	
		return AddTab(window, tab_name);
	}

	void CompositeWindow::RegisterKeySequence_(const ShortcutAction& shortcut)
	{
		QShortcut* new_shortcut(new QShortcut(shortcut.sequence, this));
		connect(new_shortcut, &QShortcut::activated, shortcut.action);
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
		// Stores the menu bar back into the originating child.
		if (m_current_child_ > -1 && index > -1)
		{
			m_children_[m_current_child_]->setMenuBar(this->menuBar());
		}
		
		// Acquires the menu bar from the current child.
		if (index > -1)
		{
			this->setMenuBar(m_children_[index]->menuBar());
		}

		m_current_child_ = index;
	}

	void CompositeWindow::OnTabRequest_(int tab_id)
	{
		m_ui_->tabWidget->setCurrentIndex(tab_id);
	}
}
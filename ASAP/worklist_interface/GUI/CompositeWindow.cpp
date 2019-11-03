#include "CompositeWindow.h"
#include <QDebug>
#include <qshortcut.h>

namespace ASAP
{
	CompositeWindow::CompositeWindow(QWidget* parent) : QMainWindow(parent), m_ui_(new Ui::CompositeWindowLayout), m_current_child_(-1)
	{
		m_ui_->setupUi(this);

		SetSlots_();
	}

	int CompositeWindow::AddTab(QMainWindow* window, const std::string tab_name)
	{
		m_children_.push_back(window);
		if (!this->menuBar()->children().isEmpty()) {
			this->menuBar()->addSeparator();
		}
		QMenuBar* cur_menu_bar = window->menuBar();
		for (auto child : cur_menu_bar->children()) {
			if (auto child_action = qobject_cast<QAction*>(child)) {
				menuBar()->addAction(child_action);
			}
			else if (auto child_menu = qobject_cast<QMenu*>(child)) {
				if (child_menu->objectName() != "menuFile") {
					menuBar()->addMenu(child_menu);
				}
			}
		}
		window->menuBar()->hide();
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
		// Ensures the object still exists.
		if (this)
		{
			m_current_child_ = index;
		}
	}

	void CompositeWindow::OnTabRequest_(int tab_id)
	{
		m_ui_->tabWidget->setCurrentIndex(tab_id);
	}
}
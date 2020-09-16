#include "CompositeWindow.h"
#include <QDebug>
#include <qshortcut.h>

namespace ASAP
{
	CompositeWindow::CompositeWindow(QWidget* parent) : QMainWindow(parent), m_ui_(new Ui::CompositeWindowLayout), m_current_child_(-1)
	{
		m_ui_->setupUi(this);
		m_settings = new QSettings(QSettings::IniFormat, QSettings::UserScope, "DIAG", "ASAP", this);
		setSlots_();
		this->readSettings();
	}

	CompositeWindow::~CompositeWindow()
	{
		writeSettings();
	}

	int CompositeWindow::addTab(QMainWindow* window, const std::string tab_name)
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
					&CompositeChild::requiresTabSwitch,
					this,
					&CompositeWindow::onTabRequest);
		}
		return id;
	}

	int CompositeWindow::addTab(CompositeChild* window, const std::string tab_name, std::vector<ShortcutAction>& shortcuts)
	{
		for (ShortcutAction& shortcut : shortcuts)
		{
			registerKeySequence_(shortcut);
		}
	
		return addTab(window, tab_name);
	}

	void CompositeWindow::registerKeySequence_(const ShortcutAction& shortcut)
	{
		QShortcut* new_shortcut(new QShortcut(shortcut.sequence, this));
		connect(new_shortcut, &QShortcut::activated, shortcut.action);
	}

	void CompositeWindow::setSlots_(void)
	{
		connect(m_ui_->tabWidget,
			SIGNAL(currentChanged(int)),
			this,
			SLOT(OnTabChange_(int)));
	}

	void CompositeWindow::readSettings()
	{
		m_settings->beginGroup("ASAP_Composite");
		resize(m_settings->value("size", QSize(1037, 786)).toSize());
		if (m_settings->value("maximized", false).toBool()) {
			this->setWindowState(Qt::WindowMaximized);
		}
		m_settings->endGroup();
	}

	void CompositeWindow::writeSettings()
	{
		m_settings->beginGroup("ASAP_Composite");
		m_settings->setValue("size", size());
		m_settings->setValue("maximized", isMaximized());
		m_settings->endGroup();
	}

	void CompositeWindow::onTabChange(int index)
	{
		// Ensures the object still exists.
		if (this)
		{
			m_current_child_ = index;
		}
	}

	void CompositeWindow::onTabRequest(int tab_id)
	{
		m_ui_->tabWidget->setCurrentIndex(tab_id);
	}
}
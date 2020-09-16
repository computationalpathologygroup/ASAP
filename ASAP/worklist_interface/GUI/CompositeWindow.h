#ifndef __ASAP_GUI_COMPOSITEWINDOW__
#define __ASAP_GUI_COMPOSITEWINDOW__

#include <functional>
#include <memory>
#include <unordered_map>
#include <QtWidgets/QMainWindow>
#include <QSettings>

#include "CompositeChild.h"
#include "ui_CompositeWindowLayout.h"

namespace ASAP
{
	/// <summary>
	/// Holds a shortcut bound to an action.
	/// </summary>
	struct ShortcutAction
	{
		std::function<void(void)>	action;
		QKeySequence				sequence;
	};

	/// <summary>
	/// Hosts QMainwindows in a unified window with a TabWidget to switch between
	/// each child. If a MenuBar is available, it's placed ontop of the TabWidget.
	///
	/// For signal based context switching, the CompositeChild can be implemented
	/// by a hosted window.
	/// </summary>
	class CompositeWindow : public QMainWindow
	{
		Q_OBJECT

		public:
			explicit CompositeWindow(QWidget* parent = 0);

			~CompositeWindow();

			int addTab(QMainWindow* window, const std::string tab_name);
			int addTab(CompositeChild* window, const std::string tab_name, std::vector<ShortcutAction>& shortcuts);

		private:
			int											m_current_child_;
			std::vector<QMainWindow*>					m_children_;
			std::unordered_map<std::string, size_t>		m_mapped_children_;
			std::unique_ptr<Ui::CompositeWindowLayout>	m_ui_;
			QSettings*									m_settings;

			void registerKeySequence_(const ShortcutAction& shortcut);
			void setSlots_(void);
			void readSettings();
			void writeSettings();
			
		private slots:
			void onTabChange(int index);
			void onTabRequest(int tab_id);
	};
}
#endif // __ASAP_GUI_COMPOSITEWINDOW__
#ifndef __ASAP_GUI_COMPOSITECHILD__
#define __ASAP_GUI_COMPOSITECHILD__
#include <QMainWindow>

namespace ASAP
{
	/// <summary>
	/// Provides an interface that interacts with the CompositeWindow class,
	/// allowing the implementation to request a tab or context switch 
	/// through a signal.
	/// </summary>
	class CompositeChild : public QMainWindow
	{
		Q_OBJECT
		public:
			CompositeChild(QWidget *parent);

		signals:
			void requiresTabSwitch(int tab_id);
	};
}
#endif // __ASAP_GUI_COMPOSITECHILD__
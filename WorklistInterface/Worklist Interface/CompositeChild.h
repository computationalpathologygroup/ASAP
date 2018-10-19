#pragma once
#include <QMainWindow>

namespace ASAP::Worklist::GUI
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
			void RequiresTabSwitch(int tab_id);
	};
}
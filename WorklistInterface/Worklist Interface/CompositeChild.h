#pragma once
#include <QMainWindow>

/// <summary>
/// Provides an interface that interacts with the CompositeWindow class,
/// allowing the implementation to request a tab or context switch 
/// through a signal.
/// </summary>
namespace ASAP::Worklist::GUI
{
	class CompositeChild : public QMainWindow
	{
		Q_OBJECT
		public:
			CompositeChild(QWidget *parent);

		signals:
			void RequiresTabSwitch(int tab_id);
	};
}
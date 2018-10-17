#pragma once
#include <QObject>
#include <QtWidgets/QMenubar>

namespace ASAP::Worklist::GUI
{
	class CompositeChild : public QObject
	{
		Q_OBJECT
		public:
			virtual QMenuBar* GetMenuElement(void) = 0;

		public slots:
			virtual void IsCompositeChild(void) = 0;

		signals:
			void RequiresContextSwitch(const std::string context_id);
	};
}
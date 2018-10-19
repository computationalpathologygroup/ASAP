#pragma once
#include <QDialog>

namespace ASAP::Worklist::GUI
{
	class ExternalSourceDialog : public QDialog
	{
		Q_OBJECT
		public:
			ExternalSourceDialog(void);

		signals:
			void AcquiredLoginDetails(const QString& location, const QString& username, const QString& password);
	};
}
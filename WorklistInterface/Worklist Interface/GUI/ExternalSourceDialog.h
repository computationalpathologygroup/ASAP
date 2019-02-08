#ifndef __ASAP_GUI_EXTERNALSOURCEDIALOG__ 
#define __ASAP_GUI_EXTERNALSOURCEDIALOG__

#include <QDialog>

namespace ASAP::GUI
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
#endif // __ASAP_GUI_EXTERNALSOURCEDIALOG__
#ifndef __ASAP_GUI_EXTERNALSOURCEDIALOG__ 
#define __ASAP_GUI_EXTERNALSOURCEDIALOG__

#include <QDialog>

class QLabel;
class QLineEdit;
class QCheckBox;
class QDialogButtonBox;
class QGridLayout;

namespace ASAP
{
	class ExternalSourceDialog : public QDialog
	{
		Q_OBJECT
		public:
			struct SourceDialogResults
			{
				QString location;
				QString token;
				bool	ignore_certificate;
			};

			explicit ExternalSourceDialog(QWidget* parent = nullptr);

			SourceDialogResults getLoginDetails(void);
			bool hasValidCredentials(void);


		private:
			QLabel* m_label_location;
			QLabel*	m_label_token;

			QLineEdit* m_input_location;
			QLineEdit* m_input_token;
			QCheckBox* m_input_ignore_certificate;

			QDialogButtonBox*	m_buttons;
			QGridLayout* m_grid_layout;

			bool m_valid;

			void setupGUI(void);

		private slots:
			void saveCredentials(void);
	};
}
#endif // __ASAP_GUI_EXTERNALSOURCEDIALOG__
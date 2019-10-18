#ifndef __ASAP_GUI_EXTERNALSOURCEDIALOG__ 
#define __ASAP_GUI_EXTERNALSOURCEDIALOG__

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QGridLayout>
#include <QCheckBox>

namespace ASAP::GUI
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

			SourceDialogResults GetLoginDetails(void);
			bool HasValidCredentials(void);


		private:
			QLabel*				m_label_location_;
			QLabel*				m_label_token_;

			QLineEdit*			m_input_location_;
			QLineEdit*			m_input_token_;
			QCheckBox*			m_input_ignore_certificate_;

			QDialogButtonBox*	m_buttons_;
			QGridLayout*		m_grid_layout_;

			bool				m_valid_;

			void SetupGUI_(void);

		private slots:
			void SaveCredentials_(void);
	};
}
#endif // __ASAP_GUI_EXTERNALSOURCEDIALOG__
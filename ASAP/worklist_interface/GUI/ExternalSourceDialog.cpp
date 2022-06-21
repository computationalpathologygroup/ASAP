#include "ExternalSourceDialog.h"

#include <QLabel>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QLineEdit>
#include <QGridLayout>
#include <QCheckBox>

namespace ASAP
{
	ExternalSourceDialog::ExternalSourceDialog(QWidget* parent) : QDialog(parent), m_valid(false)
	{
		setWindowTitle(tr("Open external source"));
		setModal(true);

		setupGUI();
	}

	ExternalSourceDialog::SourceDialogResults ExternalSourceDialog::getLoginDetails(void)
	{
		return { m_input_location->text(), m_input_token->text(), m_input_ignore_certificate->checkState() == Qt::Checked };
	}

	bool ExternalSourceDialog::hasValidCredentials(void)
	{
		return m_valid;
	}

	void ExternalSourceDialog::setupGUI(void)
	{
		// Initializes and configures the location label.
		m_label_location = new QLabel(this);
		m_label_location->setText(tr("Location"));

		// Initializes and configures the token label.
		m_label_token = new QLabel(this);
		m_label_token->setText(tr("Token"));

		// Initializes and configures the location input.
		m_input_location = new QLineEdit(this);

		// Initializes and configures the token input.
		m_input_token = new QLineEdit(this);

		// Initializes and configures the certificate checkbox.
		m_input_ignore_certificate = new QCheckBox(this);
		m_input_ignore_certificate->setText(tr("Accept invalid certificate"));

		// Initializes and configures the dialog buttons.
		m_buttons = new QDialogButtonBox(this);
		m_buttons->addButton(QDialogButtonBox::Ok);
		m_buttons->addButton(QDialogButtonBox::Cancel);

		// Links the labels to the inputs.
		m_label_location->setBuddy(m_input_location);
		m_label_token->setBuddy(m_input_token);

		// Initializes the layout.
		m_grid_layout = new QGridLayout(this);
		m_grid_layout->addWidget(m_label_location, 0, 0);
		m_grid_layout->addWidget(m_input_location, 0, 1);
		m_grid_layout->addWidget(m_label_token, 1, 0);
		m_grid_layout->addWidget(m_input_token, 1, 1);
		m_grid_layout->addWidget(m_input_ignore_certificate, 2, 0, 1, 2);
		m_grid_layout->addWidget(m_buttons, 3, 0, 1, 2);

		// Connects signals to slots
		connect(m_buttons->button(QDialogButtonBox::Cancel),
			SIGNAL(clicked()),
			this,
			SLOT(close())
		);

		connect(m_buttons->button(QDialogButtonBox::Ok),
			SIGNAL(clicked()),
			this,
			SLOT(saveCredentials())
		);
	}

	void ExternalSourceDialog::saveCredentials(void)
	{
		if (!m_input_location->text().isEmpty())
		{
			if (!m_input_token->text().isEmpty())
			{
				m_valid = true;
				close();
			}
			else
			{
				//TODO: Add tooltip
			}
		}
		else
		{
			//TODO: Add tooltip
		}
	}
}
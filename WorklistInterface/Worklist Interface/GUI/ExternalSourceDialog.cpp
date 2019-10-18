#include "ExternalSourceDialog.h"

namespace ASAP::GUI
{
	ExternalSourceDialog::ExternalSourceDialog(QWidget* parent) : QDialog(parent)
	{
		setWindowTitle(tr("Open External Source"));
		setModal(true);

		SetupGUI_();
	}

	ExternalSourceDialog::SourceDialogResults ExternalSourceDialog::GetLoginDetails(void)
	{
		return { m_input_location_->text(), m_input_token_->text(), m_input_ignore_certificate_->checkState() == Qt::Checked };
	}

	bool ExternalSourceDialog::HasValidCredentials(void)
	{
		return m_valid_;
	}

	void ExternalSourceDialog::SetupGUI_(void)
	{
		// Initializes and configures the location label.
		m_label_location_ = new QLabel(this);
		m_label_location_->setText(tr("Location"));

		// Initializes and configures the token label.
		m_label_token_ = new QLabel(this);
		m_label_token_->setText(tr("Token"));

		// Initializes and configures the location input.
		m_input_location_ = new QLineEdit(this);

		// Initializes and configures the token input.
		m_input_token_ = new QLineEdit(this);

		// Initializes and configures the certificate checkbox.
		m_input_ignore_certificate_ = new QCheckBox(this);
		m_input_ignore_certificate_->setText(tr("Accept invalid certificate"));

		// Initializes and configures the dialog buttons.
		m_buttons_ = new QDialogButtonBox(this);
		m_buttons_->addButton(QDialogButtonBox::Ok);
		m_buttons_->addButton(QDialogButtonBox::Cancel);

		// Links the labels to the inputs.
		m_label_location_->setBuddy(m_input_location_);
		m_label_token_->setBuddy(m_input_token_);

		// Initializes the layout.
		m_grid_layout_ = new QGridLayout(this);
		m_grid_layout_->addWidget(m_label_location_, 0, 0);
		m_grid_layout_->addWidget(m_input_location_, 0, 1);
		m_grid_layout_->addWidget(m_label_token_, 1, 0);
		m_grid_layout_->addWidget(m_input_token_, 1, 1);
		m_grid_layout_->addWidget(m_input_ignore_certificate_, 2, 0, 1, 2);
		m_grid_layout_->addWidget(m_buttons_, 3, 0, 1, 2);

		// Connects signals to slots
		connect(m_buttons_->button(QDialogButtonBox::Cancel),
			SIGNAL(clicked()),
			this,
			SLOT(close())
		);

		connect(m_buttons_->button(QDialogButtonBox::Ok),
			SIGNAL(clicked()),
			this,
			SLOT(SaveCredentials_())
		);
	}

	void ExternalSourceDialog::SaveCredentials_(void)
	{
		if (!m_input_location_->text().isEmpty())
		{
			if (!m_input_token_->text().isEmpty())
			{
				m_valid_ = true;
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
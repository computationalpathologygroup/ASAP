#include "DocumentWindow.h"

#include <sstream>

#include <qmessagebox.h>

namespace ASAP
{
	DocumentWindow::DocumentWindow(QWidget* parent) : QMainWindow(parent), m_active_document_(nullptr)
	{
		SetupUI_();
	}

	void DocumentWindow::AddDocument(Document& document)
	{
		std::string filename(document.GetFilepath().filename().string());

		// Inserts document into internal list.
		auto result = m_documents_.insert({ document.GetFilepath().filename().string() , &document });

		if (result.second)
		{
			// Inserts document into tab bar;
			m_document_bar_->insertTab(m_document_bar_->count(), QString::fromStdString(filename));
		}
		else
		{
			std::stringstream message;
			message << "Document: " << filename << " has already been added to this window.";

			QMessageBox message_box;
			message_box.setText(QString::fromStdString(message.str()));
			message_box.exec();
		}
	}

	void DocumentWindow::SetupSlots_(void)
	{

	}

	void DocumentWindow::SetupUI_(void)
	{
		m_document_bar_ = new QTabBar(this);

		m_view_ = new PathologyViewer(this);
		m_view_->setObjectName(QStringLiteral("pathologyView"));
	}
}
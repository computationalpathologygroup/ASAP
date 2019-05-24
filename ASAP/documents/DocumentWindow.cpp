#include "DocumentWindow.h"

#include <sstream>
#include <qformlayout.h>
#include <qmessagebox.h>
#include <qlabel.h>

#include "../PathologyViewer.h"

namespace ASAP
{
	DocumentWindow::DocumentWindow(QWidget* parent) : QWidget(parent), m_active_document_(nullptr)
	{
		//resizeDocks

	//	this->setResizeAnchor(parent);


		SetupUI_();
		SetupSlots_();
	}

	void DocumentWindow::AddDocument(Document& document)
	{
		std::string filename(document.GetFilepath().filename().string());

		// Inserts document into internal list.
		auto result = m_documents_.insert({ filename, &document });

		if (result.second)
		{
			// Inserts document into tab bar;
			m_document_bar_->insertTab(m_document_bar_->count(), QString::fromStdString(filename));
			m_view_->initialize(*result.first->second);
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
		connect(m_document_bar_,
			&QTabBar::currentChanged,
			this,
			&DocumentWindow::OnDocumentSelect_);

		connect(m_document_bar_,
			&QTabBar::tabCloseRequested,
			this,
			&DocumentWindow::OnTabClose_);
	}

	void DocumentWindow::SetupUI_(void)
	{
		m_coordinates_label_	= new QLabel("Coordinates", this);
		m_document_bar_			= new QTabBar(this);		
		m_view_					= new PathologyViewer(this);

		m_document_bar_->setTabsClosable(true);

		m_view_->setObjectName(QStringLiteral("pathologyView"));

		QFormLayout* layout(new QFormLayout());
		layout->addRow(m_document_bar_);		
		layout->addRow(m_view_);
		layout->addRow(m_coordinates_label_);
		this->setLayout(layout);

		this->setContentsMargins(0, 0, 0, 0);
	}

	void DocumentWindow::OnDocumentSelect_(int index)
	{
		auto document_it = m_documents_.find(m_document_bar_->tabText(index).toStdString());
		if (document_it != m_documents_.end())
		{
			m_view_->initialize(*document_it->second);
		}
	}

	void DocumentWindow::OnTabClose_(int index)
	{
		m_view_->close();
		m_documents_.erase(m_document_bar_->tabText(index).toStdString());
		m_document_bar_->removeTab(index);
	}
}
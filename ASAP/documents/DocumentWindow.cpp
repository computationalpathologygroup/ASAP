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
		SetupUI_();
		SetupSlots_();
	}

	DocumentWindow::~DocumentWindow(void)
	{
		Clear();
	}

	void DocumentWindow::AddDocumentInstance(DocumentInstance& instance)
	{
		// Inserts document into internal list.
		auto result = m_documents_.insert({ instance.name, instance });
		if (result.second)
		{
			// Inserts document into tab bar;
			m_document_bar_->insertTab(m_document_bar_->count(), QString::fromStdString(instance.name));
		}
		else
		{
			std::stringstream message;
			message << "Unable to add: " << instance.name << " to this viewer.";

			QMessageBox message_box;
			message_box.setText(QString::fromStdString(message.str()));
			message_box.exec();
		}
	}

	void DocumentWindow::Clear(void)
	{
		for (size_t tab = 0; tab < m_document_bar_->count(); ++tab)
		{
			OnTabClose_(tab);
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
		m_document_bar_			= new QTabBar(this);		
		m_view_					= new PathologyViewer(this);

		m_document_bar_->setTabsClosable(true);
		m_document_bar_->setDrawBase(true);
		m_document_bar_->setVisible(true);
		m_document_bar_->setAcceptDrops(true);
		m_document_bar_->setMovable(true);

		m_view_->setObjectName(QStringLiteral("pathologyView"));

		QBoxLayout* layout(new QBoxLayout(QBoxLayout::Direction::TopToBottom));
		layout->addWidget(m_document_bar_);
		layout->addWidget(m_view_);
		this->setLayout(layout);
 		this->setContentsMargins(0, 0, 0, 0);
	}

	void DocumentWindow::OnDocumentSelect_(int index)
	{
		auto document_it = m_documents_.find(m_document_bar_->tabText(index).toStdString());
		if (document_it != m_documents_.end())
		{
			m_active_document_ = &document_it->second;
			m_view_->initialize(*m_active_document_);
		}
	}

	void DocumentWindow::OnTabClose_(int index)
	{
		std::string tab_name(m_document_bar_->tabText(index).toStdString());
		m_documents_.erase(tab_name);
		m_document_bar_->removeTab(index);

		if (m_active_document_->name == tab_name)
		{
			m_view_->close();
			m_active_document_ = nullptr;
		}
	}
}
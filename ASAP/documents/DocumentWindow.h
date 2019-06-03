#ifndef __ASAP_DOCUMENTS_DOCUMENTWINDOW__
#define __ASAP_DOCUMENTS_DOCUMENTWINDOW__

#include "asaplib_export.h"
#include <string>
#include <unordered_map>
#include <qlabel.h>
#include <qtabbar.h>
#include <QtWidgets/qwidget.h>


#include "Document.h"

class PathologyViewer;

namespace ASAP
{
	class DocumentWindow : public QWidget
	{
		Q_OBJECT

		public:
			PathologyViewer*		m_view_;

			explicit DocumentWindow(QWidget* parent = 0);

			void AddDocument(Document& document);

		private:
			Document*									m_active_document_;
			std::unordered_map<std::string, Document*>	m_documents_;
			QTabBar*									m_document_bar_;

			void SetupUI_(void);
			void SetupSlots_(void);

		private slots:
			void OnDocumentSelect_(int index);
			void OnTabClose_(int index);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTWINDOW__
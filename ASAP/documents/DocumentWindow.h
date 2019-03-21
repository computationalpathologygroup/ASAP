#ifndef __ASAP_DOCUMENTS_DOCUMENTWINDOW__
#define __ASAP_DOCUMENTS_DOCUMENTWINDOW__

#include "asaplib_export.h"
#include <string>
#include <unordered_map>
#include <QMainWindow>
#include <qtabbar.h>

#include "../PathologyViewer.h"

namespace ASAP
{
	class DocumentWindow : public QMainWindow
	{
		Q_OBJECT

		public:
			PathologyViewer*		m_view_;

			DocumentWindow(QWidget* parent);

			void AddDocument(Document& document);

		private:
			Document*									m_active_document_;
			std::unordered_map<std::string, Document*>	m_documents_;
			QTabBar*									m_document_bar_;

			void SetupUI_(void);
			void SetupSlots_(void);

		private slots:
			void OnDocumentSelect_(void);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTWINDOW__
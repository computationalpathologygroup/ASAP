#ifndef __ASAP_DOCUMENTS_DOCUMENTWINDOW__
#define __ASAP_DOCUMENTS_DOCUMENTWINDOW__

#include "asaplib_export.h"
#include <string>
#include <unordered_map>
#include <qlabel.h>
#include <qtabbar.h>
#include <QtWidgets/qwidget.h>

#include "Document.h"
#include "DocumentInstance.h"
#include "../WSITileGraphicsItemCache.h"

class PathologyViewer;

namespace ASAP
{
	class DocumentWindow : public QWidget
	{
		Q_OBJECT

		public:
			PathologyViewer*		m_view_;

			explicit DocumentWindow(WSITileGraphicsItemCache& cache, QWidget* parent = 0);
			~DocumentWindow(void);

			void AddDocumentInstance(DocumentInstance& instance);
			void Clear(void);

		signals:
			void acquiredDocumentInstance(const DocumentInstance& instance);
			void changedDocumentInstanceDisplay(const DocumentInstance& instance);
			void closedDocumentInstance(const DocumentInstance& instance);
			void receivedFocus(DocumentWindow* window);
			

		private:
			DocumentInstance*									m_active_document_;
			std::unordered_map<std::string, DocumentInstance>	m_documents_;
			QTabBar*											m_document_bar_;

			void SetupUI_(WSITileGraphicsItemCache& cache);
			void SetupSlots_(void);

		private slots:
			void OnDocumentSelect_(int index);
			void OnTabClose_(int index);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTWINDOW__
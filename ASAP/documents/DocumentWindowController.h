#ifndef __ASAP_DOCUMENTS_DOCUMENTWINDOWCONTROLLER__
#define __ASAP_DOCUMENTS_DOCUMENTWINDOWCONTROLLER__

#include <vector>
#include <memory>
#include <mutex>

#include <QObject>
#include <QMouseEvent>

#include "documents/DocumentInstance.h"
#include "Documents/DocumentWindow.h"
#include "PathologyViewer.h"
#include "WSITileGraphicsItemCache.h"

#include "asaplib_export.h"

namespace ASAP
{
	class ASAPLIB_EXPORT DocumentWindowController : public QObject
	{
		Q_OBJECT
		public:
			DocumentWindowController(void);
			~DocumentWindowController(void);

			void CleanAllWindows(void);

			uint64_t GetCacheSize(void) const;
			/// <summary>
			/// Changes the size of the internal cache. This operation isn't thread safe.
			/// </summary>
			/// <param name="size">The new size in bytes.</param>
			void SetCacheSize(uint64_t size);

			DocumentWindow* GetActiveWindow(void) const;
			DocumentWindow* SpawnWindow(QWidget* parent = nullptr);

		signals:
			void viewerFocusChanged(DocumentWindow* window);

		private:
			DocumentWindow*					m_active_;
			std::vector<DocumentWindow*>	m_windows_;
			WSITileGraphicsItemCache		m_cache_;
			std::mutex						m_active_change_mutex_;

		private slots:
			void CheckMouseOrigin_(QMouseEvent* event);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTWINDOWCONTROLLER__
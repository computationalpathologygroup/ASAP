#ifndef __ASAP_DOCUMENTS_DOCUMENTWINDOWCONTROLLER__
#define __ASAP_DOCUMENTS_DOCUMENTWINDOWCONTROLLER__

#include <QObject>
#include <vector>
#include <memory>

#include "documents/DocumentInstance.h"
#include "Documents/DocumentWindow.h"
#include "PathologyViewer.h"

#include "asaplib_export.h"

class MultiResolutionImage;
class RenderThread;
class PrefetchThread;
class ToolPluginInterface;
class MiniMap;
class WSITileGraphicsItemCache;
class TileManager;
class ScaleBar;
class QSettings;
class PathologyViewer;
class DocumentWindow;

namespace ASAP
{
	class ASAPLIB_EXPORT DocumentWindowController : QObject
	{
		Q_OBJECT
		public:
			DocumentWindowController(void);

			DocumentWindow* GetActiveWindow(void) const;
			DocumentWindow* SpawnWindow(QWidget* parent = nullptr);

		signals:
			void FocusChanged(DocumentWindow* window);

		private:
			DocumentWindow* m_active_;
			std::vector<DocumentWindow*> m_viewers_;

			void SetupSlots_(void);

		private slots:
			void OnFocusChanged_(const PathologyViewer* view);
	};
}
#endif // __ASAP_DOCUMENTS_DOCUMENTWINDOWCONTROLLER__
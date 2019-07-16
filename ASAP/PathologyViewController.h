#ifndef __PATHOLOGY_VIEW_CONTROLLER__
#define __PATHOLOGY_VIEW_CONTROLLER__

#include <mutex>
#include <unordered_set>

#include <QObject>

#include "PathologyViewer.h"
#include "WSITileGraphicsItemCache.h"
#include "documents/DocumentInstance.h"
#include "Documents/DocumentWindow.h"
#include "interfaces/interfaces.h"

#include "asaplib_export.h"

namespace ASAP
{
	class ASAPLIB_EXPORT PathologyViewController : public QObject
	{
		public:
			PathologyViewController(void);
			~PathologyViewController(void);

			void SetMasterViewer(PathologyViewer* viewer);
			PathologyViewer* GetMasterViewer(void);

			void AddSlaveViewer(PathologyViewer* viewer);
			void ClearSlaveViewers(void);
			const std::unordered_set<PathologyViewer*>& GetSlaveViewers(void) const;
			void RemoveSlaveViewer(PathologyViewer* viewer);

			bool IsPanning(void) const;
			float GetPanSensitivity(void) const;
			float GetZoomSensitivity(void) const;		
			void SetPanSensitivity(float sensitivity);
			void SetZoomSensitivity(float sensitivity);

			void AddTool(std::shared_ptr<ToolPluginInterface> tool);
			std::shared_ptr<ToolPluginInterface> GetActiveTool(void) const;
			bool HasTool(const std::string& name) const;
			void SetActiveTool(const std::string& name);
			
			void Pan(const QPoint position);
			void TogglePan(bool pan, const QPoint& startPos = QPoint());
			void Zoom(const float steps);
			void Zoom(const float steps, const QPointF& center_view, const QPointF& center_scene);

		public slots:

		private:
			std::mutex m_modification_mutex_;

			PathologyViewer*						m_master_;
			std::unordered_set<PathologyViewer*>	m_slaves_;

			std::shared_ptr<ToolPluginInterface>						m_active_tool_;
			std::map<std::string, std::shared_ptr<ToolPluginInterface>> m_tools_;
			

			bool		m_is_panning_;
			float		m_pan_sensitivity_;
			float		m_zoom_sensitivity_;
			float		m_zoom_steps_;
			QPointF		m_zoom_view_center_;
			QPointF		m_zoom_scene_center_;

			void ConnectObserved_(PathologyViewer* viewer);
			void ConnectObserver_(PathologyViewer* viewer);
			void DisconnectObserved_(PathologyViewer* viewer);
			void DisconnectObserver_(PathologyViewer* viewer);

			void UpdatePan_(const QPoint position);
			void ZoomObservers_(const qreal x);
			void ZoomFinished_(void);

		private slots:
			void OnMasterMouseEvent_(QMouseEvent* event, QPointF location);
			virtual void OnKeyPressEvent(QKeyEvent* event);
			virtual void OnMouseMoveEvent(QMouseEvent* event);
			virtual void OnMousePressEvent(QMouseEvent* event);
			virtual void OnMouseReleaseEvent(QMouseEvent* event);
			virtual void OnMouseDoubleClickEvent(QMouseEvent* event);
			virtual void OnWheelEvent(QWheelEvent* event);
	};
}
#endif // __PATHOLOGY_VIEW_CONTROLLER__
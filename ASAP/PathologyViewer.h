#ifndef PATHOLOGYVIEWER_H
#define PATHOLOGYVIEWER_H
#include "asaplib_export.h"
#include <QGraphicsView>
#include <vector>
#include <memory>

#include "documents/DocumentInstance.h"

class MultiResolutionImage;
class RenderThread;
class PrefetchThread;
class ToolPluginInterface;
class MiniMap;
class WSITileGraphicsItemCache;
class TileManager;
class ScaleBar;
class QSettings;

class ASAPLIB_EXPORT PathologyViewer : public QGraphicsView
{
    Q_OBJECT
    
	public:
		PathologyViewer(WSITileGraphicsItemCache& cache, QWidget* parent = 0);
		~PathologyViewer(void);

		void initialize(ASAP::DocumentInstance& instance);
		void close(void);

		/*
		QPointF zoom_view_center;
		QPointF zoom_scene_center;

		
		float getZoomSensitivity(void) const;
		float getPanSensitivity(void) const;
		void  setZoomSensitivity(float zoomSensitivity);
		void  setPanSensitivity(float panSensitivity);
		*/

		float getForegroundOpacity() const;
		void setForegroundOpacity(const float& opacity);
		void setForegroundWindowAndLevel(const float& window, const float& level);
		void setForegroundLUT(const std::string& LUTname);
		void setForegroundChannel(unsigned int channel);

		void togglePan(bool pan, const QPoint& startPos = QPoint());
		/*bool isPanning();
		void zoom(float numSteps, QPointF center_view, QPointF center_scene);
   
		bool hasTool(const std::string& toolName) const;
		void addTool(std::shared_ptr<ToolPluginInterface> tool);
		void setActiveTool(const std::string& toolName);
		std::shared_ptr<ToolPluginInterface> getActiveTool(void);*/


		float getSceneScale(void) const;




		void setAutoUpdate(bool autoUpdate);

		void modifyZoom(const qreal x, float num_steps, const QPointF& view_center, const QPointF& scene_center);
		void modifyZoom(const QPointF& zoom_view, const QPointF& zoom_scene, const qreal zoom_factor);
		void modifyPan(const QPoint& pan_to_point);
		void setViewState(const PathologyViewState& state);
		void refreshView(void);

		bool eventFilter(QObject *object, QEvent *event);


	signals:
		void mouseMoveOccured(QMouseEvent* event);
		void mousePressOccured(QMouseEvent* event);
		void mouseReleaseOccured(QMouseEvent* event);
		void mouseDoubleClickOccured(QMouseEvent* event);
		void wheelOccured(QWheelEvent* event);

	//	void mouseEventOccured(QMouseEvent* event, const QPointF scene_location);
		void fieldOfViewChanged(const QRectF& FOV, const unsigned int level);
		void updateBBox(const QRectF& FOV);
		void backgroundChannelChanged(int channelNr);
		void receivedFocus(const PathologyViewer* viewer);
		void documentInstanceChanged(ASAP::DocumentInstance* instance);

	public slots :
		void moveTo(const QPointF& pos);
		//void changeActiveTool();
		void onFieldOfViewChanged(const QRectF& FOV, const unsigned int level);
		void onForegroundImageChanged(std::weak_ptr<MultiResolutionImage> for_img, float scale);

	private slots:
		void showContextMenu(const QPoint& pos);
		//void scalingTime(qreal x);
	//	void zoomFinished();

	private :
		float _sceneScale;

		// Interface to the multi-resolution image. Please note that PathologyViewer
		// should never modify _img (it is modified in the RenderThread due to calls
		// to readRegion), otherwise race conditions could occur.
		ASAP::DocumentInstance* m_instance_;
		//std::weak_ptr<MultiResolutionImage> _for_img;

		// Internals
		WSITileGraphicsItemCache&	_cache;
		TileManager*				_manager;
		PrefetchThread*				_prefetchthread;

		// UI
		MiniMap*	_map;
		ScaleBar*	_scaleBar;
		QSettings*	_settings;

	

		// Members to track panning and zooming
	/*	float _zoomSensitivity;
		float _numScheduledScalings;
		float _panSensitivity;
		bool _pan;*/

		// Members related to rendering
		RenderThread* _renderthread;

		void setPan_(const QPoint& pan_to_point);
		void setZoom_(const QPointF& zoom_view, const QPointF& zoom_scene, const qreal zoom_factor);

		virtual void resizeEvent(QResizeEvent *event);

		
		// Functions for panning
		virtual void mouseMoveEvent(QMouseEvent *event);
		virtual void mousePressEvent(QMouseEvent *event);
		virtual void mouseReleaseEvent(QMouseEvent *event);
		virtual void mouseDoubleClickEvent(QMouseEvent *event);
		//virtual void keyPressEvent(QKeyEvent *event);
		

		// Functions for zooming and resizing
		void wheelEvent(QWheelEvent *event);    
		
		// Functions for overviewmap
		void initializeGUIComponents(unsigned int level);
		void initializeImage(unsigned int tileSize, unsigned int lastLevel);
};
#endif // PATHOLOGYVIEWER_H
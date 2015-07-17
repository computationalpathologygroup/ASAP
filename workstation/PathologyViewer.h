#ifndef PATHOLOGYVIEWER_H
#define PATHOLOGYVIEWER_H
#include "config/pathology_config.h"
#include <QGraphicsView>
#include <vector>
#include <memory>

class MultiResolutionImage;
class RenderThread;
class PrefetchThread;
class ToolPluginInterface;
class MiniMap;
class WSITileGraphicsItemCache;

class EXPORT_PATHOLOGYWORKSTATION PathologyViewer : public QGraphicsView
{
    Q_OBJECT
    
public:

  QPointF _zoomToScenePos;
  QPointF _zoomToViewPos;

    PathologyViewer(QWidget *parent = 0);
    ~PathologyViewer();

    void initialize(MultiResolutionImage *img);
    void close();

    float getZoomSensitivity() const;
    void  setZoomSensitivity(float zoomSensitivity);
    void  setPanSensitivity(float panSensitivity);
    float getPanSensitivity() const;

    float getForegroundOpacity() const;
    void  setForegroundOpacity(const float& opacity);

    void togglePan(bool pan, const QPoint& startPos = QPoint());
    void pan(const QPoint& panTo);
    bool isPanning();
    void zoom(float numSteps);
   
    bool hasTool(const std::string& toolName) const;
    void addTool(ToolPluginInterface* tool);
    void setActiveTool(const std::string& toolName);
    float getSceneScale() { return _sceneScale; }

    unsigned long long getCacheSize();
    void setCacheSize(unsigned long long& maxCacheSize);
    void setAutoUpdate(bool autoUpdate);

signals :
    void fieldOfViewChanged(const QRectF& FOV, MultiResolutionImage* img, const unsigned int level, int channel);
    void updateBBox(const QRectF& FOV);
    void channelChanged(int channelNr);

public slots :
    void moveTo(const QPointF& pos);
    void changeActiveTool();
    void onFieldOfViewChanged(const QRectF& FOV, MultiResolutionImage* img, const unsigned int level, int channel);
    void onForegroundImageChanged(MultiResolutionImage* for_img);

private :
    
    // Functions for panning
    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void mouseDoubleClickEvent(QMouseEvent *event);
    virtual void keyPressEvent(QKeyEvent *event);
    virtual void resizeEvent(QResizeEvent *event);

    // Functions for zooming and resizing
    void wheelEvent(QWheelEvent *event);    

    // Functions for overviewmap
    void initializeMiniMap(unsigned int level);
    void initializeImage(QGraphicsScene *scn, unsigned int tileSize, unsigned int lastLevel);
    float _sceneScale;

    // Interface to the multi-resolution image. Please note that PathologyViewer
    // should never modify _img (it is modified in the RenderThread due to calls
    // to readRegion), otherwise race conditions could occur.
    MultiResolutionImage *_img;

    // Minimap
    MiniMap *_map; 

    // Tools
    ToolPluginInterface* _activeTool;

    // Members to track panning and zooming
    float _zoomSensitivity;
    float _numScheduledScalings;
    float _panSensitivity;
    bool _pan;
    QPoint _prevPan;

    RenderThread* _renderthread;
    PrefetchThread* _prefetchthread;

    unsigned long long _cacheSize;
    WSITileGraphicsItemCache* _cache;

    std::map<std::string, ToolPluginInterface*> _tools;

private slots :
    void showContextMenu(const QPoint& pos);
    void scalingTime(qreal x);
    void zoomFinished();
};
  

#endif
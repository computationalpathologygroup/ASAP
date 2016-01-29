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
class TileManager;
class ScaleBar;
class QSettings;

class EXPORT_PATHOLOGYWORKSTATION PathologyViewer : public QGraphicsView
{
    Q_OBJECT
    
public:

    QPointF _zoomToScenePos;
    QPointF _zoomToViewPos;

    PathologyViewer(QWidget *parent = 0);
    ~PathologyViewer();

    void initialize(std::shared_ptr<MultiResolutionImage> img);
    void close();

    float getZoomSensitivity() const;
    void  setZoomSensitivity(float zoomSensitivity);
    void  setPanSensitivity(float panSensitivity);
    float getPanSensitivity() const;

    float getForegroundOpacity() const;
    void  setForegroundOpacity(const float& opacity);

    void setForegroundWindowAndLevel(const float& window, const float& level);
    void setForegroundLUT(const std::string& LUTname);
    void setForegroundChannel(unsigned int channel);

    void togglePan(bool pan, const QPoint& startPos = QPoint());
    void pan(const QPoint& panTo);
    bool isPanning();
    void zoom(float numSteps);
   
    bool hasTool(const std::string& toolName) const;
    void addTool(std::shared_ptr<ToolPluginInterface> tool);
    void setActiveTool(const std::string& toolName);
    float getSceneScale() { return _sceneScale; }

    unsigned long long getCacheSize();
    void setCacheSize(unsigned long long& maxCacheSize);
    void setAutoUpdate(bool autoUpdate);

signals :
    void fieldOfViewChanged(const QRectF& FOV, const unsigned int level);
    void updateBBox(const QRectF& FOV);
    void backgroundChannelChanged(int channelNr);

public slots :
    void moveTo(const QPointF& pos);
    void changeActiveTool();
    void onFieldOfViewChanged(const QRectF& FOV, const unsigned int level);
    void onForegroundImageChanged(std::weak_ptr<MultiResolutionImage> for_img, float scale);

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
    void initializeGUIComponents(unsigned int level);
    void initializeImage(QGraphicsScene *scn, unsigned int tileSize, unsigned int lastLevel);
    float _sceneScale;

    // Interface to the multi-resolution image. Please note that PathologyViewer
    // should never modify _img (it is modified in the RenderThread due to calls
    // to readRegion), otherwise race conditions could occur.
    std::shared_ptr<MultiResolutionImage> _img;
    std::weak_ptr<MultiResolutionImage> _for_img;

    // Minimap
    MiniMap* _map;

    // ScaleBar
    ScaleBar* _scaleBar;

    // QSettings
    QSettings* _settings;

    // Tools
    std::shared_ptr<ToolPluginInterface> _activeTool;

    // Members to track panning and zooming
    float _zoomSensitivity;
    float _numScheduledScalings;
    float _panSensitivity;
    bool _pan;
    QPoint _prevPan;

    // Members related to rendering
    RenderThread* _renderthread;
    int _backgroundChannel;
    int _foregroundChannel;
    float _opacity;
    float _window;
    float _level;
    std::string _LUTname;
    float _foregroundImageScale;

    PrefetchThread* _prefetchthread;

    TileManager* _manager;
    unsigned long long _cacheSize;
    WSITileGraphicsItemCache* _cache;

    std::map<std::string, std::shared_ptr<ToolPluginInterface> > _tools;

private slots :
    void showContextMenu(const QPoint& pos);
    void scalingTime(qreal x);
    void zoomFinished();
};
  

#endif
#ifndef PATHOLOGY_VIEW_STATE_H
#define PATHOLOGY_VIEW_STATE_H
#include "asaplib_export.h"

#include <qpoint.h>

struct ASAPLIB_EXPORT PathologyViewState
{
	QPoint	pan_position;
	QPointF zoom_view_center;
	QPointF zoom_scene_center;
	qreal	zoom_factor;
};
#endif // PATHOLOGY_VIEW_STATE_H
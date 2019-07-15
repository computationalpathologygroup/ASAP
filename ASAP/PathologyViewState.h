#ifndef PATHOLOGY_VIEW_STATE_H
#define PATHOLOGY_VIEW_STATE_H
#include "asaplib_export.h"

#include <memory>

#include <qpoint.h>

#include "multiresolutionimageinterface/MultiResolutionImage.h"

struct ASAPLIB_EXPORT PathologyViewState
{
	std::weak_ptr<MultiResolutionImage> foreground_image;

	uint32_t	background_channel;
	uint32_t	foreground_channel;
	float		foreground_opacity;
	float		foreground_window;
	float		foreground_level;
	std::string foreground_lut_name;
	float		foreground_scale;

	QPoint	pan_position;
	QPointF zoom_view_center;
	QPointF zoom_scene_center;
	qreal	zoom_factor;

	PathologyViewState(void) : zoom_factor(1.0f)
	{
	}
};
#endif // PATHOLOGY_VIEW_STATE_H
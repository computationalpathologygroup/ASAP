#include "PathologyViewController.h"

#include <QTimeLine>

namespace ASAP
{
	PathologyViewController::PathologyViewController(void) : QObject(), m_master_(nullptr)
	{
	}

	PathologyViewController::~PathologyViewController(void)
	{
		m_modification_mutex_.lock();
		m_modification_mutex_.unlock();
	}

	void PathologyViewController::SetMasterViewer(PathologyViewer* viewer)
	{
		m_modification_mutex_.lock();
		if (m_master_ != viewer)
		{
			if (m_master_)
			{
				DisconnectObserved_(m_master_);
				DisconnectObserver_(m_master_);
				m_is_panning_ = false;
			}

			m_master_ = viewer;
			ConnectObserved_(m_master_);
			ConnectObserver_(m_master_);
		}
		m_modification_mutex_.unlock();
	}

	PathologyViewer* PathologyViewController::GetMasterViewer(void)
	{
		return m_master_;
	}

	void PathologyViewController::AddSlaveViewer(PathologyViewer* viewer)
	{
		m_modification_mutex_.lock();
		auto result = m_slaves_.insert(viewer);
		if (result.second)
		{
			ConnectObserver_(viewer);
		}
		m_modification_mutex_.unlock();
	}

	void PathologyViewController::ClearSlaveViewers(void)
	{
		m_modification_mutex_.lock();
		for (PathologyViewer* viewer : m_slaves_)
		{
			DisconnectObserver_(viewer);
		}
		m_slaves_.clear();
		m_modification_mutex_.unlock();
	}

	const std::unordered_set<PathologyViewer*>& PathologyViewController::GetSlaveViewers(void) const
	{
		return m_slaves_;
	}

	void PathologyViewController::RemoveSlaveViewer(PathologyViewer* viewer)
	{
		m_modification_mutex_.lock();
		DisconnectObserver_(viewer);
		m_slaves_.erase(viewer);
		m_modification_mutex_.unlock();
	}

	bool PathologyViewController::IsPanning(void) const
	{
		return m_is_panning_;
	}

	float PathologyViewController::GetPanSensitivity(void) const
	{
		return m_pan_sensitivity_;
	}

	float PathologyViewController::GetZoomSensitivity(void) const
	{
		return m_zoom_sensitivity_;
	}

	void PathologyViewController::SetPanSensitivity(float sensitivity)
	{
		if (sensitivity > 1)
		{
			m_pan_sensitivity_ = 1;
		}
		else if (sensitivity < 0.01)
		{
			m_pan_sensitivity_ = 0.01;
		}
		else
		{
			m_pan_sensitivity_ = sensitivity;
		}
	}
	void PathologyViewController::SetZoomSensitivity(float sensitivity)
	{
		if (sensitivity > 1)
		{
			m_zoom_sensitivity_ = 1;
		}
		else if (sensitivity < 0.01)
		{
			m_zoom_sensitivity_ = 0.01;
		}
		else
		{
			m_zoom_sensitivity_ = sensitivity;
		}
	}

	void PathologyViewController::AddTool(std::shared_ptr<ToolPluginInterface> tool)
	{
		m_modification_mutex_.lock();
		if (tool)
		{
			m_tools_.insert({ tool->name(), tool });
			tool->setController(*this);
			tool->setViewer(m_master_);
		}
		m_modification_mutex_.unlock();
	}

	std::shared_ptr<ToolPluginInterface> PathologyViewController::GetActiveTool(void) const
	{
		return m_active_tool_;
	}

	bool PathologyViewController::HasTool(const std::string& name) const
	{
		return m_tools_.find(name) != m_tools_.end();
	}

	void PathologyViewController::SetActiveTool(const std::string& name)
	{
		m_modification_mutex_.lock();
		auto tool = m_tools_.find(name);
		if (tool != m_tools_.end())
		{
			if (m_active_tool_)
			{
				m_active_tool_->setActive(false);
			}
			m_active_tool_ = tool->second;
			m_active_tool_->setActive(true);
		}
		m_modification_mutex_.unlock();
	}

	void PathologyViewController::Pan(const QPoint position)
	{
		UpdatePan_(position);
	}

	void PathologyViewController::TogglePan(bool pan, const QPoint& startPos)
	{
		m_modification_mutex_.lock();
		bool changed = false;
		if (pan && !m_is_panning_)
		{
			m_is_panning_	= true;
			changed			= true;
		}
		else if (m_is_panning_)
		{
			m_is_panning_	= false;
			changed			= true;
		}

		if (changed && m_master_)
		{
			m_master_->togglePan(m_is_panning_, startPos);
			for (PathologyViewer* viewer : m_slaves_)
			{
				if (viewer)
				{
					viewer->togglePan(m_is_panning_, startPos);
				}
			}
		}
		m_modification_mutex_.unlock();
	}

	void PathologyViewController::Zoom(const float steps)
	{
		Zoom(steps, m_zoom_view_center_, m_zoom_scene_center_);
	}

	void PathologyViewController::Zoom(const float steps, const QPointF& center_view, const QPointF& center_scene)
	{
		m_modification_mutex_.lock();
		if (m_master_)
		{
			m_zoom_view_center_		= center_view;
			m_zoom_scene_center_	= center_scene;
			m_zoom_steps_			+= steps;

			if (m_zoom_steps_ * steps < 0)
			{
				m_zoom_steps_ = steps;
			}

			QTimeLine* anim = new QTimeLine(300, this);
			anim->setUpdateInterval(5);

			connect(anim,
				&QTimeLine::valueChanged,
				this,
				&PathologyViewController::ZoomObservers_);

			connect(anim,
				&QTimeLine::finished,
				this,
				&PathologyViewController::ZoomFinished_);

			anim->start();
		}
		m_modification_mutex_.unlock();
	}

	//############ Public Slots ############//


	//############ Privates Methods ############//


	void PathologyViewController::ConnectObserved_(PathologyViewer* viewer)
	{
		QObject::connect(viewer,
			&PathologyViewer::mouseDoubleClickOccured,
			this,
			&PathologyViewController::OnMouseDoubleClickEvent);

		QObject::connect(viewer,
			&PathologyViewer::mouseMoveOccured,
			this,
			&PathologyViewController::OnMouseMoveEvent);

		QObject::connect(viewer,
			&PathologyViewer::mousePressOccured,
			this,
			&PathologyViewController::OnMousePressEvent);

		QObject::connect(viewer,
			&PathologyViewer::mouseReleaseOccured,
			this,
			&PathologyViewController::OnMouseReleaseEvent);

		QObject::connect(viewer,
			&PathologyViewer::wheelOccured,
			this,
			&PathologyViewController::OnWheelEvent);
	}

	void PathologyViewController::ConnectObserver_(PathologyViewer* viewer)
	{
	}

	void PathologyViewController::DisconnectObserved_(PathologyViewer* viewer)
	{
		QObject::disconnect(viewer,
			&PathologyViewer::mouseDoubleClickOccured,
			this,
			&PathologyViewController::OnMouseDoubleClickEvent);

		QObject::disconnect(viewer,
			&PathologyViewer::mouseMoveOccured,
			this,
			&PathologyViewController::OnMouseMoveEvent);

		QObject::disconnect(viewer,
			&PathologyViewer::mousePressOccured,
			this,
			&PathologyViewController::OnMousePressEvent);

		QObject::disconnect(viewer,
			&PathologyViewer::mouseReleaseOccured,
			this,
			&PathologyViewController::OnMouseReleaseEvent);

		QObject::disconnect(viewer,
			&PathologyViewer::wheelOccured,
			this,
			&PathologyViewController::OnWheelEvent);
	}

	void PathologyViewController::DisconnectObserver_(PathologyViewer* viewer)
	{

	}

	void PathologyViewController::UpdatePan_(const QPoint position)
	{
		m_modification_mutex_.lock();
		if (m_master_)
		{
			m_master_->modifyPan(position);
			for (PathologyViewer* viewer : m_slaves_)
			{
				if (viewer)
				{
					viewer->modifyPan(position);
				}
			}
		}
		m_modification_mutex_.unlock();
	}

	void PathologyViewController::ZoomObservers_(const qreal x)
	{
		m_modification_mutex_.lock();
		if (m_master_)
		{
			m_master_->modifyZoom(x, m_zoom_steps_, m_zoom_view_center_, m_zoom_scene_center_);
		}
		for (PathologyViewer* viewer : m_slaves_)
		{
			if (viewer)
			{
				viewer->modifyZoom(x, m_zoom_steps_, m_zoom_view_center_, m_zoom_scene_center_);
			}
		}
		m_modification_mutex_.unlock();
	}

	void PathologyViewController::ZoomFinished_(void)
	{
		if (m_zoom_steps_ > 0)
		{
			m_zoom_steps_--;
		}
		else
		{
			m_zoom_steps_++;
		}
		sender()->~QObject();
	}

	//############ Private Slots ############//

	void PathologyViewController::OnMasterMouseEvent_(QMouseEvent* event, QPointF location)
	{
		switch (event->type())
		{
			case QEvent::MouseButtonDblClick:	OnMouseDoubleClickEvent(event);		break;
			case QEvent::MouseButtonPress:		OnMousePressEvent(event);			break;
			case QEvent::MouseButtonRelease:	OnMouseReleaseEvent(event);			break;
			case QEvent::MouseMove:				OnMouseMoveEvent(event);			break;
			case QEvent::Wheel:					OnWheelEvent((QWheelEvent*)event);	break;
		}
	}

	void PathologyViewController::OnKeyPressEvent(QKeyEvent* event)
	{
		event->ignore();
		if (m_active_tool_)
		{
			m_active_tool_->keyPressEvent(event);
		}
	}

	void PathologyViewController::OnMouseDoubleClickEvent(QMouseEvent* event)
	{
		event->ignore();
		if (m_active_tool_)
		{
			m_active_tool_->mouseDoubleClickEvent(event);
		}
	}

	void PathologyViewController::OnMouseMoveEvent(QMouseEvent* event)
	{
		QPointF img_loc = m_master_->mapToScene(event->pos()) / m_master_->getSceneScale();
		// TODO: Clean this up or implement a more elegant selection method.
		//qobject_cast<QMainWindow*>(this->parentWidget()->parentWidget()->parentWidget())->statusBar()->showMessage(QString("Current position in image coordinates: (") + QString::number(imgLoc.x()) + QString(", ") + QString::number(imgLoc.y()) + QString(")"), 1000);

		if (m_is_panning_)
		{
			UpdatePan_(event->pos());
			event->accept();
			return;
		}
		if (m_active_tool_)
		{
			m_active_tool_->mouseMoveEvent(event);
			if (event->isAccepted())
			{
				return;
			}
		}
		event->ignore();
	}

	void PathologyViewController::OnMousePressEvent(QMouseEvent* event)
	{
		if (event->button() == Qt::MiddleButton)
		{
			TogglePan(true, event->pos());
			event->accept();
			return;
		}
		else if (m_active_tool_ && event->button() == Qt::LeftButton)
		{
			m_active_tool_->mousePressEvent(event);
			if (event->isAccepted())
			{
				return;
			}
		}
		event->ignore();
	}

	void PathologyViewController::OnMouseReleaseEvent(QMouseEvent* event)
	{
		if (event->button() == Qt::MiddleButton)
		{
			TogglePan(false);
			event->accept();
			return;
		}
		else  if (m_active_tool_ && event->button() == Qt::LeftButton)
		{
			m_active_tool_->mouseReleaseEvent(event);
			if (event->isAccepted())
			{
				return;
			}
		}
		event->ignore();
	}

	void PathologyViewController::OnWheelEvent(QWheelEvent* event)
	{
		int numDegrees	= event->delta() / 8;
		int numSteps	= numDegrees / 15;  // see QWheelEvent documentation
		Zoom(numSteps, event->pos(), m_master_->mapToScene(event->pos()));
	}
}
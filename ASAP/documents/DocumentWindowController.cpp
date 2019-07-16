#include "DocumentWindowController.h"

namespace ASAP
{
	DocumentWindowController::DocumentWindowController(void)
	{
	}

	DocumentWindowController::~DocumentWindowController(void)
	{
		m_active_change_mutex_.lock();
		m_active_change_mutex_.unlock();
	}

	uint64_t DocumentWindowController::GetCacheSize(void) const
	{
		return m_cache_.getMaxCacheSize();
	}

	void DocumentWindowController::SetCacheSize(uint64_t size)
	{
		m_cache_.setMaxCacheSize(size);
	}

	ASAP::DocumentWindow* DocumentWindowController::GetActiveWindow(void) const
	{
		return m_active_;
	}

	ASAP::DocumentWindow* DocumentWindowController::SpawnWindow(QWidget* parent)
	{
		m_viewers_.push_back(new DocumentWindow(m_cache_, parent));
		ASAP::DocumentWindow* viewer = m_viewers_.back();

		QObject::connect(viewer->m_view_,
			&PathologyViewer::mouseMoveOccured,
			this,
			&DocumentWindowController::CheckMouseMoveOrigin_);

		return viewer;
	}

	void DocumentWindowController::SetupSlots_(void)
	{

	}

	void DocumentWindowController::CheckMouseMoveOrigin_(QMouseEvent* event)
	{
		m_active_change_mutex_.lock();
		if (!m_active_ || sender() != m_active_->m_view_)
		{
			for (DocumentWindow* window : m_viewers_)
			{
				if (window->m_view_ == sender())
				{
					m_active_ = window;
					emit viewerFocusChanged(window);
					break;
				}
			}
		}
		m_active_change_mutex_.unlock();
	}
}
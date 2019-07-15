#include "DocumentWindowController.h"

namespace ASAP
{
	DocumentWindowController::DocumentWindowController(void)
	{
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
			&PathologyViewer::receivedFocus,
			this,
			&DocumentWindowController::OnFocusChanged_);
		return viewer;
	}

	void DocumentWindowController::SetupSlots_(void)
	{

	}

	void DocumentWindowController::OnFocusChanged_(const PathologyViewer* view)
	{
		// Finishes the operations of the currently active view.

		// Switches the focused view to active view.
		for (DocumentWindow* w : m_viewers_)
		{
			if (w->m_view_ == view)
			{
				m_active_ = w;
				break;
			}
		}
	}
}
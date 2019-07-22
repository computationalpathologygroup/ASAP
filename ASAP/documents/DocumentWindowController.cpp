#include "DocumentWindowController.h"

namespace ASAP
{
	DocumentWindowController::DocumentWindowController(void)
	{
	}

	DocumentWindowController::~DocumentWindowController(void)
	{
		CleanAllWindows();
		m_active_change_mutex_.lock();
		m_active_change_mutex_.unlock();
	}

	void DocumentWindowController::CleanAllWindows(void)
	{
		m_active_change_mutex_.lock();
		for (DocumentWindow* window : m_windows_)
		{
			window->Clear();
		}
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
		m_windows_.push_back(new DocumentWindow(m_cache_, parent));
		ASAP::DocumentWindow* window = m_windows_.back();

		QObject::connect(window->viewer,
			&PathologyViewer::mouseReleaseOccured,
			this,
			&DocumentWindowController::CheckMouseOrigin_);

		return window;
	}

	void DocumentWindowController::SetupSlots_(void)
	{

	}

	void DocumentWindowController::CheckMouseOrigin_(QMouseEvent* event)
	{
		m_active_change_mutex_.lock();
		if (!m_active_ || sender() != m_active_->viewer)
		{
			for (DocumentWindow* window : m_windows_)
			{
				if (window->viewer == sender())
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
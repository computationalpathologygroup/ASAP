#include "interfaces.h"

bool WorkstationExtensionPluginInterface::initialize(ASAP::PathologyViewController& controller)
{
	_controller = &controller;
	connect(_controller,
		&ASAP::PathologyViewController::MasterChangeStarted,
		this,
		&WorkstationExtensionPluginInterface::onViewerChangeStart);

	connect(_controller,
		&ASAP::PathologyViewController::MasterChangeFinished,
		this,
		&WorkstationExtensionPluginInterface::onViewerChangeFinished);

	return true;
}

void WorkstationExtensionPluginInterface::onViewerChangeStart(void)
{
	PathologyViewer* master_view(_controller->GetMasterViewer());
	if (master_view)
	{
		disconnect(master_view,
			&PathologyViewer::documentInstanceChanged,
			this,
			&WorkstationExtensionPluginInterface::onDocumentChange);

		ASAP::DocumentWindow* window((ASAP::DocumentWindow*)master_view->parentWidget());
		disconnect(window,
			&ASAP::DocumentWindow::DocumentInstanceCloseStarted,
			this,
			&WorkstationExtensionPluginInterface::onDocumentInstanceClose);
	}
}

void WorkstationExtensionPluginInterface::onViewerChangeFinished(void)
{
	PathologyViewer* master_view(_controller->GetMasterViewer());
	if (master_view)
	{
		connect(master_view,
			&PathologyViewer::documentInstanceChanged,
			this,
			&WorkstationExtensionPluginInterface::onDocumentChange);

		ASAP::DocumentWindow* window((ASAP::DocumentWindow*)master_view->parentWidget());
		connect(window,
			&ASAP::DocumentWindow::DocumentInstanceCloseStarted,
			this,
			&WorkstationExtensionPluginInterface::onDocumentInstanceClose);
	}
}
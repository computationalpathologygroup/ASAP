#pragma once

#include <memory>
#include <QtWidgets/QMainWindow>

#include "ui_CompositeWindowLayout.h"

namespace ASAP::Worklist::GUI
{
	class CompositeWindow : public QMainWindow
	{
		Q_OBJECT

		public:
			explicit CompositeWindow(QWidget* parent = 0);

			void AddTab(QMainWindow* window, const std::string tab_name);

		private:
			std::unique_ptr<Ui::CompositeWindowLayout> m_ui_;

		private slots:
			void OnTabChange_(int index);
	};
}
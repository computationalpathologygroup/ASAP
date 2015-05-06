#include <qtoolbutton.h>
#include <qtoolbar.h>

class WorkstationToolbar : public QToolBar {

public:
	WorkstationToolbar();
	std::vector<QToolButton*> getButtons();
	void setAllButtonsUp();
	void activateButtons();
	void deactivateButtons();

private:
	std::vector<QToolButton*> _buttons;
	void initialize();
};
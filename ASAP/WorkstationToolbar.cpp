
#include <QToolBar>
#include <vector>
#include <WorkstationToolbar.h>
#include <qdebug.h>

using std::vector;

static const int NUMBER_OF_BUTTONS = 4;
static const int BUTTON_NAME_LENGTH = 15;
static const int INACTIVE_BUTTON_NAME_LENGTH = 24;

WorkstationToolbar::WorkstationToolbar()
{
	this->initialize();
};

void WorkstationToolbar::initialize() 
{
	const char icons[NUMBER_OF_BUTTONS][INACTIVE_BUTTON_NAME_LENGTH] =
		{":/inactive_dot_____.xpm", ":/inactive_poly____.xpm", ":/inactive_freehand.xpm", ":/inactive_spline__.xpm"};

	for (int i=0; i<NUMBER_OF_BUTTONS; i++)
	{
		QToolButton* qToolButton = new QToolButton();
		_buttons.push_back(qToolButton);
		QPixmap qPixmap(icons[i]);
		qToolButton->setText(icons[i]);
		qToolButton->setIcon(QIcon(qPixmap));
		qToolButton->setCheckable(false);
		this->addWidget(qToolButton);
	}
};

std::vector<QToolButton*> WorkstationToolbar::getButtons()
{
	return _buttons;
};

void WorkstationToolbar::setAllButtonsUp()
{
	for (int i=0; i<_buttons.size(); i++)
	{
		_buttons[i]->setChecked(false);
		_buttons[i]->update();
	}
};

void WorkstationToolbar::activateButtons()
{
	const char icons[NUMBER_OF_BUTTONS][BUTTON_NAME_LENGTH] = 
		{":/dot_____.xpm", ":/poly____.xpm", ":/freehand.xpm", ":/spline__.xpm"};

	for (int i=0; i<NUMBER_OF_BUTTONS; i++)
	{
		QToolButton* qToolButton = _buttons.at(i);
		QPixmap qPixmap(icons[i]);
		qToolButton->setText(icons[i]);
		qToolButton->setIcon(QIcon(qPixmap));
		qToolButton->setCheckable(true);
	}
}

void WorkstationToolbar::deactivateButtons() 
{
	const char icons[NUMBER_OF_BUTTONS][INACTIVE_BUTTON_NAME_LENGTH] =
		{":/inactive_dot_____.xpm", ":/inactive_poly____.xpm", ":/inactive_freehand.xpm", ":/inactive_spline__.xpm"};

	for (int i=0; i<NUMBER_OF_BUTTONS; i++)
	{
		QToolButton* qToolButton = _buttons.at(i);
		QPixmap qPixmap(icons[i]);
		qToolButton->setText(icons[i]);
		qToolButton->setIcon(QIcon(qPixmap));
		qToolButton->setCheckable(false);
	}
}
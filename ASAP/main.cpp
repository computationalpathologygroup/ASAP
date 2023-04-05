#include "ASAP_Window.h"
#include <QApplication>
#include <QStyle>

int main(int argc, char *argv[])
{    
    QApplication a(argc, argv);
    a.setStyle("windows");
	  ASAP_Window w;
    w.show();

    return a.exec();
}

#include "ASAP_Window.h"
#include <QApplication>

int main(int argc, char *argv[])
{    
    QApplication a(argc, argv);
	ASAP_Window w;
    w.show();

    return a.exec();
}

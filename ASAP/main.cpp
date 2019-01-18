#include "ASAP.h"
#include <QApplication>

int main(int argc, char *argv[])
{    
    QApplication a(argc, argv);
    ASAP w;
    w.show();

    return a.exec();
}

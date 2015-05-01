#include "pathologyworkstation.h"
#include <QApplication>

int main(int argc, char *argv[])
{    
    QApplication a(argc, argv);
    PathologyWorkstation w;
    w.show();

    return a.exec();
}

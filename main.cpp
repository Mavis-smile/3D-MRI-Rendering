#include "mainwindow.h"
#include "LoginDialog.h"
#include <QtGlobal>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    LoginDialog login;
    if (login.exec() != QDialog::Accepted || !login.isAuthenticated()) {
        return 0;
    }

    MainWindow w;
    w.show();
    return a.exec();
}

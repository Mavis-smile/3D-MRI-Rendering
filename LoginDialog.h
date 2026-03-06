#ifndef LOGINDIALOG_H
#define LOGINDIALOG_H

#include <QDialog>

namespace Ui {
class LoginDialog;
}

class LoginDialog : public QDialog
{
    Q_OBJECT

public:
    explicit LoginDialog(QWidget *parent = nullptr);
    ~LoginDialog();

    bool isAuthenticated() const { return m_authenticated; }

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void attemptLogin();

private:
    Ui::LoginDialog *ui;
    bool m_authenticated = false;
};

#endif // LOGINDIALOG_H

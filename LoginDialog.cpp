#include "LoginDialog.h"
#include "ui_LoginDialog.h"
#include "UserManager.h"
#include <QCloseEvent>
#include <QMessageBox>
#include <QApplication>

LoginDialog::LoginDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::LoginDialog)
{
    ui->setupUi(this);
    connect(ui->loginButton, &QPushButton::clicked, this, &LoginDialog::attemptLogin);
}

LoginDialog::~LoginDialog()
{
    delete ui;
}

void LoginDialog::closeEvent(QCloseEvent *event)
{
    QDialog::closeEvent(event);
}

void LoginDialog::attemptLogin()
{
    m_authenticated = UserManager::instance().authenticate(
        ui->usernameEdit->text(), ui->passwordEdit->text());

    if (m_authenticated) {
        accept(); // Close dialog with Accepted result
    } else {
        QMessageBox::warning(this, "Error", "Invalid credentials");
        ui->passwordEdit->clear();
    }
}

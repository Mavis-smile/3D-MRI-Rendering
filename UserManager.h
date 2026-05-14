#ifndef USERMANAGER_H
#define USERMANAGER_H

#include <QString>
#include <QStringList>
#include <QMap>

class UserManager
{
public:
    static UserManager& instance();

    bool authenticate(const QString& username, const QString& password) const;
    bool addUser(const QString& username, const QString& password);
    bool updateUser(const QString& oldUsername, const QString& newUsername, const QString& newPassword);
    bool deleteUser(const QString& username);

    QStringList usernames() const;
    int userCount() const;

private:
    UserManager();
    UserManager(const UserManager&) = delete;
    UserManager& operator=(const UserManager&) = delete;

    void load();
    void save() const;
    static QString hashPassword(const QString& password);

    QMap<QString, QString> m_users; // username -> SHA-256 hex hash
};

#endif // USERMANAGER_H

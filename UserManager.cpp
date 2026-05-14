#include "UserManager.h"
#include <QSettings>
#include <QCryptographicHash>

static const char* ORG_NAME  = "MRIVisualizer";
static const char* APP_NAME  = "3DMRIRendering";
static const char* DEFAULT_USER = "nasikandar";

// ── helpers ──────────────────────────────────────────────────────────────────

QString UserManager::hashPassword(const QString& password)
{
    return QString(QCryptographicHash::hash(
        password.toUtf8(), QCryptographicHash::Sha256).toHex());
}

// ── ctor ─────────────────────────────────────────────────────────────────────

UserManager::UserManager()
{
    load();
    if (m_users.isEmpty()) {
        // Seed the default account on first run
        m_users[QString::fromLatin1(DEFAULT_USER)] =
            hashPassword(QString::fromLatin1(DEFAULT_USER));
        save();
    }
}

// ── singleton ─────────────────────────────────────────────────────────────────

UserManager& UserManager::instance()
{
    static UserManager inst;
    return inst;
}

// ── public API ────────────────────────────────────────────────────────────────

bool UserManager::authenticate(const QString& username, const QString& password) const
{
    auto it = m_users.constFind(username);
    if (it == m_users.constEnd()) return false;
    return it.value() == hashPassword(password);
}

bool UserManager::addUser(const QString& username, const QString& password)
{
    if (username.trimmed().isEmpty() || password.isEmpty()) return false;
    if (m_users.contains(username)) return false;
    m_users[username] = hashPassword(password);
    save();
    return true;
}

bool UserManager::updateUser(const QString& oldUsername,
                              const QString& newUsername,
                              const QString& newPassword)
{
    if (!m_users.contains(oldUsername)) return false;
    if (oldUsername != newUsername && m_users.contains(newUsername)) return false;
    m_users.remove(oldUsername);
    m_users[newUsername] = hashPassword(newPassword);
    save();
    return true;
}

bool UserManager::deleteUser(const QString& username)
{
    if (m_users.size() <= 1) return false;
    int removed = m_users.remove(username);
    if (removed > 0) save();
    return removed > 0;
}

QStringList UserManager::usernames() const
{
    return m_users.keys();
}

int UserManager::userCount() const
{
    return m_users.size();
}

// ── persistence ───────────────────────────────────────────────────────────────

void UserManager::load()
{
    QSettings settings(ORG_NAME, APP_NAME);
    int size = settings.beginReadArray("users");
    for (int i = 0; i < size; ++i) {
        settings.setArrayIndex(i);
        QString user = settings.value("username").toString();
        QString hash = settings.value("password").toString();
        if (!user.isEmpty())
            m_users[user] = hash;
    }
    settings.endArray();
}

void UserManager::save() const
{
    QSettings settings(ORG_NAME, APP_NAME);
    settings.remove("users");
    settings.beginWriteArray("users");
    int i = 0;
    for (auto it = m_users.constBegin(); it != m_users.constEnd(); ++it, ++i) {
        settings.setArrayIndex(i);
        settings.setValue("username", it.key());
        settings.setValue("password", it.value());
    }
    settings.endArray();
    settings.sync();
}

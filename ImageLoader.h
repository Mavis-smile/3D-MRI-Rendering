#ifndef IMAGELOADER_H
#define IMAGELOADER_H


#include <QObject>
#include <QImage>
#include <QVector3D>

class ImageLoader : public QObject {
    Q_OBJECT
public:
    explicit ImageLoader(QObject* parent = nullptr);  // Constructor
    QVector<QVector<QVector3D>> loadVolume(const QStringList& filePaths);

private:
    QImage preprocessSlice(const QImage& slice);
};

#endif // IMAGELOADER_H

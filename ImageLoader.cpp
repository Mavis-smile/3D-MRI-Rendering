#include "ImageLoader.h"
#include <QFileInfo>

ImageLoader::ImageLoader(QObject* parent)
    : QObject(parent)
{
    // Constructor implementation
}

QVector<QVector<QVector3D>> ImageLoader::loadVolume(const QStringList& filePaths) {
    QVector<QVector<QVector3D>> volumeData;

    foreach(const QString& path, filePaths) {
        QImage img(path);
        if(img.isNull()) {
            qWarning() << "Failed to load:" << path;
            continue;
        }

        QImage processed = preprocessSlice(img);
        QVector<QVector<QVector3D>> sliceData;

        for(int y = 0; y < processed.height(); y++) {
            QVector<QVector3D> row;
            for(int x = 0; x < processed.width(); x++) {
                QRgb pixel = processed.pixel(x, y);
                row.append(QVector3D(qRed(pixel), qGreen(pixel), qBlue(pixel)));
            }
            sliceData.append(row);
        }

        volumeData.append(sliceData);
    }

    return volumeData;
}
QImage ImageLoader::preprocessSlice(const QImage& slice) {
    // Convert to grayscale
    QImage gray = slice.convertToFormat(QImage::Format_Grayscale8);

    // Example: Apply threshold (adjust 128 to your needs)
    gray = gray.convertToFormat(QImage::Format_Mono);

    return gray;
}

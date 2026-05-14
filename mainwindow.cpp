#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "GLWidget.h"
#include <QFile>
#include <QSaveFile>
#include <QThread>
#include <QListView>
#include <QTreeView>
#include <QAbstractItemView>
#include <QProgressDialog>
#include <QElapsedTimer>
#include <QSharedPointer>
#include <QThreadPool>
#include <QScrollArea>
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include <QPushButton>
#include <QMouseEvent>
#include <QFileInfo>
#include <QDataStream>
#include <QTextStream>
#include <QRegularExpression>
#include <QComboBox>
#include <QCoreApplication>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <cstring>
#include <cmath>
#include <limits>

#include "MeshSimplifier.h"

namespace {
struct LoadedSliceResult {
    int index = -1;
    bool valid = false;
    QImage original;
    QImage processed;
    cv::Mat segmentation16;
    bool hasNative16 = false;
};

struct ImportFinalizeResult {
    QVector<QImage> originalImages;
    QVector<QImage> loadedImages;
    QVector<cv::Mat> segmentationSlices16;
    int failedCount = 0;
    int native16Count = 0;
    bool hasSegmentation16Data = false;
};

struct StlExportResult {
    bool success = false;
    QString errorMessage;
    quint32 writtenTriangles = 0;
    qint64 bytesOnDisk = 0;
    qint64 expectedBytes = 0;
    int skippedInvalidTriangles = 0;
    int skippedDegenerateTriangles = 0;
};

enum class ExportPreset {
    Ultra,
    Medium,
    Low
};

struct ExportOptions {
    bool simplifyBeforeExport = false;
    ExportPreset preset = ExportPreset::Ultra;
    int targetFaceCount = 0;
    double aggressiveness = 1.0;
};

quint32 floatBits(float value) {
    quint32 bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct StlVertexKey {
    quint32 x;
    quint32 y;
    quint32 z;

    bool operator==(const StlVertexKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct StlVertexKeyHash {
    std::size_t operator()(const StlVertexKey& key) const {
        const std::size_t h1 = std::hash<quint32>{}(key.x);
        const std::size_t h2 = std::hash<quint32>{}(key.y);
        const std::size_t h3 = std::hash<quint32>{}(key.z);
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

void compactIndexedMesh(QVector<QVector3D>* vertices, QVector<unsigned int>* indices) {
    if (!vertices || !indices || vertices->isEmpty() || indices->isEmpty()) {
        return;
    }

    QVector<int> remap(vertices->size(), -1);
    QVector<QVector3D> compactVertices;
    compactVertices.reserve(vertices->size());

    QVector<unsigned int> compactIndices;
    compactIndices.reserve(indices->size());

    for (int i = 0; i < indices->size(); ++i) {
        const unsigned int idx = indices->at(i);
        if (idx >= static_cast<unsigned int>(vertices->size())) {
            continue;
        }
        int mapped = remap[static_cast<int>(idx)];
        if (mapped < 0) {
            mapped = compactVertices.size();
            remap[static_cast<int>(idx)] = mapped;
            compactVertices.append(vertices->at(static_cast<int>(idx)));
        }
        compactIndices.append(static_cast<unsigned int>(mapped));
    }

    vertices->swap(compactVertices);
    indices->swap(compactIndices);
}

qint64 estimateBinaryStlBytes(int triangleCount)
{
    if (triangleCount <= 0) {
        return 84;
    }
    return 84LL + (static_cast<qint64>(triangleCount) * 50LL);
}

QString formatMegabytes(qint64 bytes)
{
    return QString::number(static_cast<double>(bytes) / (1024.0 * 1024.0), 'f', 2);
}

QString presetName(ExportPreset preset)
{
    switch (preset) {
    case ExportPreset::Ultra: return QStringLiteral("Ultra");
    case ExportPreset::Medium: return QStringLiteral("Medium");
    case ExportPreset::Low: return QStringLiteral("Low");
    }
    return QStringLiteral("Ultra");
}

double presetKeepRatio(ExportPreset preset)
{
    switch (preset) {
    case ExportPreset::Ultra: return 1.0;
    case ExportPreset::Medium: return 0.65;
    case ExportPreset::Low: return 0.35;
    }
    return 1.0;
}

ExportPreset recommendedPreset(int triangleCount, qint64 estimatedBytes)
{
    if (triangleCount >= 3000000 || estimatedBytes >= 500LL * 1024LL * 1024LL) {
        return ExportPreset::Low;
    }
    if (triangleCount >= 300000) {
        return ExportPreset::Medium;
    }
    if (triangleCount >= 100000) {
        return ExportPreset::Low;
    }
    return ExportPreset::Ultra;
}

void logExportEstimate(int triangleCount, qint64 estimatedBytes)
{
    qDebug().noquote() << "-------";
    qDebug().noquote() << QString("Triangles: %1").arg(triangleCount);
    qDebug().noquote() << QString("Estimated Binary STL Size: %1 MB").arg(formatMegabytes(estimatedBytes));
    qDebug().noquote() << "-------------------------------";
    if (estimatedBytes > 200LL * 1024LL * 1024LL) {
        qWarning() << "Estimated binary STL size exceeds 200 MB; simplification is recommended.";
    }
}

bool promptExportOptions(QWidget* parent, int triangleCount, qint64 estimatedBytes, ExportOptions* options)
{
    if (!options) {
        return false;
    }

    QDialog dialog(parent);
    dialog.setWindowTitle("Export STL Options");

    auto* layout = new QVBoxLayout(&dialog);
    auto* form = new QFormLayout();

    auto* simplifyCheck = new QCheckBox("Simplify before export (keep 35% triangles)", &dialog);
    simplifyCheck->setChecked(estimatedBytes > 200LL * 1024LL * 1024LL);

    auto* estimateLabel = new QLabel(&dialog);
    estimateLabel->setWordWrap(true);
    
    // Lambda to update the estimate label based on checkbox state
    auto updateEstimate = [simplifyCheck, estimateLabel, triangleCount]() {
        if (simplifyCheck->isChecked()) {
            const int estimatedTriangles = qMax(1, int(std::lround(double(triangleCount) * 0.35)));
            const qint64 estimatedSize = 84LL + (static_cast<qint64>(estimatedTriangles) * 50LL);
            estimateLabel->setText(QString("Estimated after simplification:\n~%1 triangles (~%2 MB)")
                                      .arg(estimatedTriangles)
                                      .arg(formatMegabytes(estimatedSize)));
        } else {
            estimateLabel->setText("");
        }
    };

    QObject::connect(simplifyCheck, &QCheckBox::toggled, updateEstimate);

    form->addRow(simplifyCheck);
    form->addRow(estimateLabel);
    layout->addLayout(form);

    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, &dialog);
    layout->addWidget(buttons);
    QObject::connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);

    // Update estimate on initial load
    updateEstimate();

    if (dialog.exec() != QDialog::Accepted) {
        return false;
    }

    options->simplifyBeforeExport = simplifyCheck->isChecked();
    options->preset = ExportPreset::Low;  // Always use Low preset (35% triangles)
    options->targetFaceCount = qMax(1, int(std::lround(double(triangleCount) * 0.35)));
    options->aggressiveness = 2.5;  // Conservative aggressiveness; internal anatomy preservation via smart triangle classification handles target achievement
    return true;
}

StlExportResult exportMeshToBinaryStlFile(
    const QString& fileName,
    const QVector<QVector3D>& vertices,
    const QVector<unsigned int>& indices,
    const std::function<void(int, const QString&)>& progressCallback,
    const std::function<bool()>& cancelCallback = std::function<bool()>()
) {
    StlExportResult result;

    auto reportProgress = [&](int progress, const QString& message) {
        if (progressCallback) {
            progressCallback(qBound(0, progress, 100), message);
        }
    };

    reportProgress(1, QString("[1%] Opening STL output file..."));

    QSaveFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        result.errorMessage = QString("Failed to open file for writing: %1").arg(file.errorString());
        return result;
    }

    auto writeAll = [&](const char* data, qint64 bytes, const char* label) -> bool {
        const qint64 written = file.write(data, bytes);
        if (written != bytes) {
            result.errorMessage = QString("Write failed at %1: expected %2 bytes, wrote %3 (%4)")
                .arg(label)
                .arg(bytes)
                .arg(written)
                .arg(file.errorString());
            return false;
        }
        return true;
    };

    QByteArray header(80, '\0');
    const QByteArray title = QByteArray("Generated by 3DMRIRendering (Binary STL)");
    std::memcpy(header.data(), title.constData(), static_cast<size_t>(qMin(header.size(), title.size())));
    if (!writeAll(header.constData(), header.size(), "header")) {
        return result;
    }

    const quint32 triangleCountPlaceholder = 0;
    if (!writeAll(reinterpret_cast<const char*>(&triangleCountPlaceholder), sizeof(triangleCountPlaceholder), "triangle-count-placeholder")) {
        return result;
    }

    const int triangleCount = indices.size() / 3;
    const int reportStride = qMax(1, triangleCount / 200);
    int processedTriangles = 0;

    reportProgress(5, QString("[5%] Writing STL triangles (0/%1)...").arg(triangleCount));

    for (int i = 0; i + 2 < indices.size(); i += 3) {
        if (cancelCallback && cancelCallback()) {
            result.errorMessage = "STL export canceled.";
            return result;
        }

        const unsigned int i0 = indices[i];
        const unsigned int i1 = indices[i + 1];
        const unsigned int i2 = indices[i + 2];

        ++processedTriangles;

        if (i0 >= static_cast<unsigned int>(vertices.size()) ||
            i1 >= static_cast<unsigned int>(vertices.size()) ||
            i2 >= static_cast<unsigned int>(vertices.size())) {
            ++result.skippedInvalidTriangles;
            continue;
        }

        const QVector3D& v0 = vertices[static_cast<int>(i0)];
        const QVector3D& v1 = vertices[static_cast<int>(i1)];
        const QVector3D& v2 = vertices[static_cast<int>(i2)];

        QVector3D normal = QVector3D::crossProduct(v1 - v0, v2 - v0);
        if (normal.lengthSquared() <= 1e-20f) {
            ++result.skippedDegenerateTriangles;
            continue;
        }
        normal.normalize();

        const float triData[12] = {
            normal.x(), normal.y(), normal.z(),
            v0.x(), v0.y(), v0.z(),
            v1.x(), v1.y(), v1.z(),
            v2.x(), v2.y(), v2.z()
        };
        const quint16 attributeByteCount = 0;

        if (!writeAll(reinterpret_cast<const char*>(triData), sizeof(triData), "triangle-data") ||
            !writeAll(reinterpret_cast<const char*>(&attributeByteCount), sizeof(attributeByteCount), "triangle-attribute")) {
            return result;
        }

        ++result.writtenTriangles;

        if (processedTriangles % reportStride == 0 || processedTriangles == triangleCount) {
            const int progress = 5 + int((90.0 * double(processedTriangles)) / double(qMax(1, triangleCount)));
            reportProgress(progress, QString("[%1%] Writing STL triangles %2/%3...").arg(progress).arg(processedTriangles).arg(triangleCount));
        }
    }

    if (cancelCallback && cancelCallback()) {
        result.errorMessage = "STL export canceled.";
        return result;
    }

    if (!file.seek(80)) {
        result.errorMessage = QString("Failed to seek back to STL triangle count: %1").arg(file.errorString());
        return result;
    }

    const quint32 triangleCountLE = result.writtenTriangles;
    if (!writeAll(reinterpret_cast<const char*>(&triangleCountLE), sizeof(triangleCountLE), "triangle-count")) {
        return result;
    }

    if (!file.commit()) {
        result.errorMessage = QString("Failed to finalize STL file: %1").arg(file.errorString());
        return result;
    }

    result.bytesOnDisk = QFileInfo(fileName).size();
    result.expectedBytes = 84LL + (static_cast<qint64>(result.writtenTriangles) * 50LL);
    if (result.bytesOnDisk != result.expectedBytes || result.bytesOnDisk <= 84) {
        result.errorMessage = QString("STL file size mismatch. Expected %1 bytes, got %2 bytes.")
            .arg(result.expectedBytes)
            .arg(result.bytesOnDisk);
        return result;
    }

    result.success = true;
    reportProgress(100, QString("[100%] STL export complete (%1 triangles).").arg(result.writtenTriangles));
    return result;
}

void keepLargestTriangleComponent(QVector<QVector3D>* vertices, QVector<unsigned int>* indices) {
    if (!vertices || !indices || indices->size() < 6) {
        return;
    }

    const int triCount = indices->size() / 3;
    QVector<QVector<int>> vertexToTriangles(vertices->size());
    for (int t = 0; t < triCount; ++t) {
        const unsigned int i0 = indices->at(3 * t + 0);
        const unsigned int i1 = indices->at(3 * t + 1);
        const unsigned int i2 = indices->at(3 * t + 2);
        if (i0 >= static_cast<unsigned int>(vertices->size()) ||
            i1 >= static_cast<unsigned int>(vertices->size()) ||
            i2 >= static_cast<unsigned int>(vertices->size())) {
            continue;
        }
        vertexToTriangles[static_cast<int>(i0)].append(t);
        vertexToTriangles[static_cast<int>(i1)].append(t);
        vertexToTriangles[static_cast<int>(i2)].append(t);
    }

    QVector<char> visited(triCount, 0);
    QVector<int> bestComponent;

    for (int start = 0; start < triCount; ++start) {
        if (visited[start]) {
            continue;
        }

        QVector<int> stack;
        QVector<int> component;
        stack.append(start);
        visited[start] = 1;

        while (!stack.isEmpty()) {
            const int t = stack.back();
            stack.pop_back();
            component.append(t);

            const unsigned int tri[3] = {
                indices->at(3 * t + 0),
                indices->at(3 * t + 1),
                indices->at(3 * t + 2)
            };

            for (int k = 0; k < 3; ++k) {
                const unsigned int v = tri[k];
                if (v >= static_cast<unsigned int>(vertexToTriangles.size())) {
                    continue;
                }
                const QVector<int>& neigh = vertexToTriangles[static_cast<int>(v)];
                for (int nt : neigh) {
                    if (!visited[nt]) {
                        visited[nt] = 1;
                        stack.append(nt);
                    }
                }
            }
        }

        if (component.size() > bestComponent.size()) {
            bestComponent = component;
        }
    }

    if (bestComponent.isEmpty() || bestComponent.size() >= (triCount * 9 / 10)) {
        return;
    }

    QVector<char> keep(triCount, 0);
    for (int t : bestComponent) {
        keep[t] = 1;
    }

    QVector<unsigned int> filtered;
    filtered.reserve(bestComponent.size() * 3);
    for (int t = 0; t < triCount; ++t) {
        if (!keep[t]) {
            continue;
        }
        filtered.append(indices->at(3 * t + 0));
        filtered.append(indices->at(3 * t + 1));
        filtered.append(indices->at(3 * t + 2));
    }

    const int removedTris = triCount - bestComponent.size();
    if (removedTris > 0) {
        qDebug() << "Mesh cleanup: removed" << removedTris
                 << "triangle(s) from detached component(s).";
    }

    indices->swap(filtered);
    compactIndexedMesh(vertices, indices);
}

void removeBorderPlaneSlab(QVector<QVector3D>* vertices, QVector<unsigned int>* indices) {
    if (!vertices || !indices || vertices->isEmpty() || indices->size() < 300) {
        return;
    }

    QVector3D minV = vertices->first();
    QVector3D maxV = vertices->first();
    for (const QVector3D& v : *vertices) {
        minV.setX(qMin(minV.x(), v.x()));
        minV.setY(qMin(minV.y(), v.y()));
        minV.setZ(qMin(minV.z(), v.z()));
        maxV.setX(qMax(maxV.x(), v.x()));
        maxV.setY(qMax(maxV.y(), v.y()));
        maxV.setZ(qMax(maxV.z(), v.z()));
    }

    const float sx = qMax(1e-6f, maxV.x() - minV.x());
    const float sy = qMax(1e-6f, maxV.y() - minV.y());
    const float sz = qMax(1e-6f, maxV.z() - minV.z());
    const float bandX = qMax(1e-4f, sx * 0.006f);
    const float bandY = qMax(1e-4f, sy * 0.006f);
    const float bandZ = qMax(1e-4f, sz * 0.006f);

    auto triNearSide = [&](const QVector3D& a, const QVector3D& b, const QVector3D& c,
                           int axis, bool nearMin) -> bool {
        if (axis == 0) {
            const float ref = nearMin ? minV.x() : maxV.x();
            return nearMin
                ? (a.x() <= ref + bandX && b.x() <= ref + bandX && c.x() <= ref + bandX)
                : (a.x() >= ref - bandX && b.x() >= ref - bandX && c.x() >= ref - bandX);
        }
        if (axis == 1) {
            const float ref = nearMin ? minV.y() : maxV.y();
            return nearMin
                ? (a.y() <= ref + bandY && b.y() <= ref + bandY && c.y() <= ref + bandY)
                : (a.y() >= ref - bandY && b.y() >= ref - bandY && c.y() >= ref - bandY);
        }
        const float ref = nearMin ? minV.z() : maxV.z();
        return nearMin
            ? (a.z() <= ref + bandZ && b.z() <= ref + bandZ && c.z() <= ref + bandZ)
            : (a.z() >= ref - bandZ && b.z() >= ref - bandZ && c.z() >= ref - bandZ);
    };

    const int triCount = indices->size() / 3;
    int bestAxis = -1;
    bool bestNearMin = true;
    int bestCount = 0;

    for (int axis = 0; axis < 3; ++axis) {
        for (int side = 0; side < 2; ++side) {
            const bool nearMin = (side == 0);
            int count = 0;
            for (int t = 0; t < triCount; ++t) {
                const unsigned int i0 = indices->at(3 * t + 0);
                const unsigned int i1 = indices->at(3 * t + 1);
                const unsigned int i2 = indices->at(3 * t + 2);
                if (i0 >= static_cast<unsigned int>(vertices->size()) ||
                    i1 >= static_cast<unsigned int>(vertices->size()) ||
                    i2 >= static_cast<unsigned int>(vertices->size())) {
                    continue;
                }
                const QVector3D& a = vertices->at(static_cast<int>(i0));
                const QVector3D& b = vertices->at(static_cast<int>(i1));
                const QVector3D& c = vertices->at(static_cast<int>(i2));
                if (triNearSide(a, b, c, axis, nearMin)) {
                    ++count;
                }
            }
            if (count > bestCount) {
                bestCount = count;
                bestAxis = axis;
                bestNearMin = nearMin;
            }
        }
    }

    if (bestAxis < 0 || bestCount < qMax(120, triCount / 50)) {
        return;
    }

    QVector<unsigned int> filtered;
    filtered.reserve(indices->size());
    int removed = 0;
    for (int t = 0; t < triCount; ++t) {
        const unsigned int i0 = indices->at(3 * t + 0);
        const unsigned int i1 = indices->at(3 * t + 1);
        const unsigned int i2 = indices->at(3 * t + 2);
        if (i0 >= static_cast<unsigned int>(vertices->size()) ||
            i1 >= static_cast<unsigned int>(vertices->size()) ||
            i2 >= static_cast<unsigned int>(vertices->size())) {
            continue;
        }
        const QVector3D& a = vertices->at(static_cast<int>(i0));
        const QVector3D& b = vertices->at(static_cast<int>(i1));
        const QVector3D& c = vertices->at(static_cast<int>(i2));
        if (triNearSide(a, b, c, bestAxis, bestNearMin)) {
            ++removed;
            continue;
        }

        filtered.append(i0);
        filtered.append(i1);
        filtered.append(i2);
    }

    if (removed <= 0) {
        return;
    }

    qDebug() << "Mesh cleanup: removed" << removed
             << "border-slab triangle(s) at axis" << bestAxis
             << (bestNearMin ? "min" : "max");

    indices->swap(filtered);
    compactIndexedMesh(vertices, indices);
}

unsigned int appendOrReuseVertex(
    const QVector3D& v,
    QVector<QVector3D>* vertices,
    std::unordered_map<StlVertexKey, unsigned int, StlVertexKeyHash>* map
) {
    const StlVertexKey key{floatBits(v.x()), floatBits(v.y()), floatBits(v.z())};
    const auto it = map->find(key);
    if (it != map->end()) {
        return it->second;
    }

    const unsigned int idx = static_cast<unsigned int>(vertices->size());
    vertices->append(v);
    map->emplace(key, idx);
    return idx;
}

int extractLastNumber(const QString& name, bool* ok = nullptr) {
    // Only accept trailing numeric index (e.g. rec0123), not arbitrary digits in prefix.
    static const QRegularExpression trailingNumberRegex("(\\d+)$");
    const QRegularExpressionMatch match = trailingNumberRegex.match(name);
    const bool hasTrailingNumber = match.hasMatch();

    if (ok) {
        *ok = hasTrailingNumber;
    }
    if (!hasTrailingNumber) {
        return -1;
    }
    return match.captured(1).toInt();
}

bool numericFileLess(const QFileInfo& a, const QFileInfo& b) {
    bool okA = false;
    bool okB = false;
    const int numA = extractLastNumber(a.completeBaseName(), &okA);
    const int numB = extractLastNumber(b.completeBaseName(), &okB);

    if (okA && okB && numA != numB) {
        return numA < numB;
    }
    if (okA != okB) {
        return okA;
    }
    return QString::localeAwareCompare(a.fileName(), b.fileName()) < 0;
}

bool looksLikeReconstructedSliceName(const QString& baseName) {
    const QString lower = baseName.toLower();
    return lower.contains("rec") || lower.contains("slice") || lower.contains("section");
}

bool looksLikeProjectionSeries(const QFileInfoList& files, const QDir& selectedDir, bool hasRecCandidate) {
    if (files.size() < 120) {
        return false;
    }

    int numericCount = 0;
    int reconstructedNameCount = 0;
    int tifCount = 0;

    for (const QFileInfo& fi : files) {
        if (fi.suffix().compare("tif", Qt::CaseInsensitive) == 0 || fi.suffix().compare("tiff", Qt::CaseInsensitive) == 0) {
            ++tifCount;
        }

        bool hasTrailingNumber = false;
        extractLastNumber(fi.completeBaseName(), &hasTrailingNumber);
        if (hasTrailingNumber) {
            ++numericCount;
        }

        if (looksLikeReconstructedSliceName(fi.completeBaseName())) {
            ++reconstructedNameCount;
        }

    }

    const bool mostlyNumeric = (numericCount >= int(files.size() * 0.80));
    const bool mostlyTiff = (tifCount >= int(files.size() * 0.80));
    const bool lacksRecTokens = (reconstructedNameCount <= int(files.size() * 0.10));
    const bool denseAcquisitionLike = (files.size() >= 300);
    const QString dirName = QFileInfo(selectedDir.absolutePath()).fileName();
    const bool dataLikeFolder = dirName.contains("data", Qt::CaseInsensitive)
        || dirName.contains("proj", Qt::CaseInsensitive)
        || dirName.contains("raw", Qt::CaseInsensitive);

    return mostlyNumeric && mostlyTiff && lacksRecTokens && denseAcquisitionLike && (dataLikeFolder || hasRecCandidate);
}

QString findLikelyReconstructionDir(const QDir& selectedDir) {
    const QStringList imageFilters = {"*.bmp", "*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.dcm", "*.dicom"};

    auto pickBestRecDir = [&](const QDir& baseDir) -> QString {
        const QFileInfoList childDirs = baseDir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
        QString bestPath;
        int bestScore = -1;

        for (const QFileInfo& child : childDirs) {
            const QString name = child.fileName();
            const bool recLikeName = name.contains("_rec", Qt::CaseInsensitive)
                || name.contains("recon", Qt::CaseInsensitive)
                || name.contains("reconstruction", Qt::CaseInsensitive);
            if (!recLikeName) {
                continue;
            }

            QDir candidate(child.absoluteFilePath());
            const QFileInfoList imgs = candidate.entryInfoList(imageFilters, QDir::Files, QDir::NoSort);
            if (imgs.isEmpty()) {
                continue;
            }

            int recTokenCount = 0;
            for (const QFileInfo& fi : imgs) {
                if (looksLikeReconstructedSliceName(fi.completeBaseName())) {
                    ++recTokenCount;
                }
            }

            const int score = imgs.size() + (recTokenCount * 3);
            if (score > bestScore) {
                bestScore = score;
                bestPath = candidate.absolutePath();
            }
        }

        return bestPath;
    };

    const QString localCandidate = pickBestRecDir(selectedDir);
    if (!localCandidate.isEmpty()) {
        return localCandidate;
    }

    QDir parentDir = selectedDir;
    if (parentDir.cdUp()) {
        return pickBestRecDir(parentDir);
    }

    return QString();
}

void centerDialogOnWidget(QDialog* dialog, QWidget* parentWidget) {
    if (!dialog) {
        return;
    }

    QRect targetRect;
    if (parentWidget) {
        const QRect localRect = parentWidget->rect();
        targetRect = QRect(parentWidget->mapToGlobal(localRect.topLeft()), localRect.size());
    } else if (QWidget* active = QApplication::activeWindow()) {
        const QRect localRect = active->rect();
        targetRect = QRect(active->mapToGlobal(localRect.topLeft()), localRect.size());
    }

    if (!targetRect.isValid() || targetRect.isEmpty()) {
        return;
    }

    dialog->adjustSize();
    QSize size = dialog->size();
    if (!size.isValid() || size.isEmpty()) {
        size = dialog->sizeHint();
    }
    if (!size.isValid() || size.isEmpty()) {
        size = dialog->minimumSizeHint();
    }

    dialog->move(targetRect.center() - QPoint(size.width() / 2, size.height() / 2));
}

bool loadBinaryStlMesh(
    QFile& file,
    QVector<QVector3D>* outVertices,
    QVector<unsigned int>* outIndices,
    QString* outError
) {
    if (!outVertices || !outIndices) {
        if (outError) {
            *outError = "Internal error: null output buffer.";
        }
        return false;
    }

    if (file.size() < 84) {
        if (outError) {
            *outError = "File too small to be a valid binary STL.";
        }
        return false;
    }

    if (!file.seek(80)) {
        if (outError) {
            *outError = "Failed to seek STL triangle count.";
        }
        return false;
    }

    quint32 triangleCount = 0;
    if (file.read(reinterpret_cast<char*>(&triangleCount), sizeof(triangleCount)) != sizeof(triangleCount)) {
        if (outError) {
            *outError = "Failed to read STL triangle count.";
        }
        return false;
    }

    const quint64 expectedSize = 84ULL + (50ULL * static_cast<quint64>(triangleCount));
    if (expectedSize != static_cast<quint64>(file.size())) {
        if (outError) {
            *outError = QString("Binary STL size mismatch (expected %1, got %2)")
                .arg(expectedSize)
                .arg(file.size());
        }
        return false;
    }

    if (!file.seek(84)) {
        if (outError) {
            *outError = "Failed to seek STL triangle payload.";
        }
        return false;
    }

    QDataStream in(&file);
    in.setByteOrder(QDataStream::LittleEndian);
    in.setFloatingPointPrecision(QDataStream::SinglePrecision);

    outVertices->clear();
    outIndices->clear();
    outIndices->reserve(static_cast<int>(qMin<quint64>(static_cast<quint64>(triangleCount) * 3ULL,
                                                       static_cast<quint64>(std::numeric_limits<int>::max()))));

    std::unordered_map<StlVertexKey, unsigned int, StlVertexKeyHash> vertexMap;
    const quint64 reserveVertices = qMin<quint64>(static_cast<quint64>(triangleCount) * 2ULL,
                                                  static_cast<quint64>(std::numeric_limits<int>::max()));
    outVertices->reserve(static_cast<int>(reserveVertices));
    vertexMap.reserve(static_cast<std::size_t>(reserveVertices));

    for (quint32 t = 0; t < triangleCount; ++t) {
        float nx = 0.0f, ny = 0.0f, nz = 0.0f;
        float x0 = 0.0f, y0 = 0.0f, z0 = 0.0f;
        float x1 = 0.0f, y1 = 0.0f, z1 = 0.0f;
        float x2 = 0.0f, y2 = 0.0f, z2 = 0.0f;
        quint16 attr = 0;

        in >> nx >> ny >> nz;
        in >> x0 >> y0 >> z0;
        in >> x1 >> y1 >> z1;
        in >> x2 >> y2 >> z2;
        in >> attr;

        if (in.status() != QDataStream::Ok) {
            if (outError) {
                *outError = QString("Failed while reading binary STL triangle %1.").arg(t);
            }
            return false;
        }

        Q_UNUSED(nx);
        Q_UNUSED(ny);
        Q_UNUSED(nz);
        Q_UNUSED(attr);

        const QVector3D v0(x0, y0, z0);
        const QVector3D v1(x1, y1, z1);
        const QVector3D v2(x2, y2, z2);

        // Drop degenerate triangles early to avoid wasted upload/render work.
        if (QVector3D::crossProduct(v1 - v0, v2 - v0).lengthSquared() <= 1e-20f) {
            continue;
        }

        const unsigned int i0 = appendOrReuseVertex(v0, outVertices, &vertexMap);
        const unsigned int i1 = appendOrReuseVertex(v1, outVertices, &vertexMap);
        const unsigned int i2 = appendOrReuseVertex(v2, outVertices, &vertexMap);

        if (i0 == i1 || i1 == i2 || i0 == i2) {
            continue;
        }

        outIndices->append(i0);
        outIndices->append(i1);
        outIndices->append(i2);
    }

    return !outVertices->isEmpty() && !outIndices->isEmpty();
}

bool loadAsciiStlMesh(
    QFile& file,
    QVector<QVector3D>* outVertices,
    QVector<unsigned int>* outIndices,
    QString* outError
) {
    if (!outVertices || !outIndices) {
        if (outError) {
            *outError = "Internal error: null output buffer.";
        }
        return false;
    }

    if (!file.seek(0)) {
        if (outError) {
            *outError = "Failed to seek ASCII STL start.";
        }
        return false;
    }

    QTextStream in(&file);
    QVector<QVector3D> triVerts;
    triVerts.reserve(3);

    outVertices->clear();
    outIndices->clear();
    outVertices->reserve(1024);
    outIndices->reserve(1024);
    std::unordered_map<StlVertexKey, unsigned int, StlVertexKeyHash> vertexMap;
    vertexMap.reserve(2048);

    while (!in.atEnd()) {
        const QString line = in.readLine().trimmed();
        if (line.isEmpty()) {
            continue;
        }

        if (!line.startsWith("vertex", Qt::CaseInsensitive)) {
            continue;
        }

        const QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() < 4) {
            continue;
        }

        bool okX = false;
        bool okY = false;
        bool okZ = false;
        const float x = parts[1].toFloat(&okX);
        const float y = parts[2].toFloat(&okY);
        const float z = parts[3].toFloat(&okZ);
        if (!okX || !okY || !okZ) {
            continue;
        }

        triVerts.append(QVector3D(x, y, z));
        if (triVerts.size() == 3) {
            if (QVector3D::crossProduct(triVerts[1] - triVerts[0], triVerts[2] - triVerts[0]).lengthSquared() > 1e-20f) {
                const unsigned int i0 = appendOrReuseVertex(triVerts[0], outVertices, &vertexMap);
                const unsigned int i1 = appendOrReuseVertex(triVerts[1], outVertices, &vertexMap);
                const unsigned int i2 = appendOrReuseVertex(triVerts[2], outVertices, &vertexMap);
                if (i0 != i1 && i1 != i2 && i0 != i2) {
                    outIndices->append(i0);
                    outIndices->append(i1);
                    outIndices->append(i2);
                }
            }
            triVerts.clear();
        }
    }

    if (outVertices->isEmpty() || outIndices->isEmpty()) {
        if (outError) {
            *outError = "No valid triangles found in ASCII STL.";
        }
        return false;
    }

    return true;
}

bool loadStlMesh(
    const QString& stlPath,
    QVector<QVector3D>* outVertices,
    QVector<unsigned int>* outIndices,
    QString* outError
) {
    QFile file(stlPath);
    if (!file.open(QIODevice::ReadOnly)) {
        if (outError) {
            *outError = QString("Failed to open STL file: %1").arg(file.errorString());
        }
        return false;
    }

    QString binaryError;
    if (loadBinaryStlMesh(file, outVertices, outIndices, &binaryError)) {
        return true;
    }

    QString asciiError;
    if (loadAsciiStlMesh(file, outVertices, outIndices, &asciiError)) {
        return true;
    }

    if (outError) {
        *outError = QString("Binary parse failed: %1 | ASCII parse failed: %2")
            .arg(binaryError)
            .arg(asciiError);
    }
    return false;
}

bool loadTxtMeshPair(
    const QString& verticesPath,
    const QString& facesPath,
    QVector<QVector3D>* outVertices,
    QVector<unsigned int>* outIndices,
    QString* outError
) {
    if (!outVertices || !outIndices) {
        if (outError) {
            *outError = "Internal error: null output buffer.";
        }
        return false;
    }

    QFile verticesFile(verticesPath);
    if (!verticesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (outError) {
            *outError = QString("Failed to open vertices file: %1").arg(verticesFile.errorString());
        }
        return false;
    }

    outVertices->clear();
    outIndices->clear();

    QTextStream vin(&verticesFile);
    while (!vin.atEnd()) {
        const QString line = vin.readLine().trimmed();
        if (line.isEmpty()) {
            continue;
        }

        const QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() < 3) {
            continue;
        }

        bool okX = false;
        bool okY = false;
        bool okZ = false;
        const float x = parts[0].toFloat(&okX);
        const float y = parts[1].toFloat(&okY);
        const float z = parts[2].toFloat(&okZ);
        if (okX && okY && okZ) {
            outVertices->append(QVector3D(x, y, z));
        }
    }
    verticesFile.close();

    if (outVertices->isEmpty()) {
        if (outError) {
            *outError = "No valid vertices found in vertices TXT.";
        }
        return false;
    }

    QFile facesFile(facesPath);
    if (!facesFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        if (outError) {
            *outError = QString("Failed to open faces file: %1").arg(facesFile.errorString());
        }
        return false;
    }

    QTextStream fin(&facesFile);
    bool formatDetermined = false;
    bool isMatlabFormat = false;
    while (!fin.atEnd()) {
        const QString line = fin.readLine().trimmed();
        if (line.isEmpty()) {
            continue;
        }

        const QStringList parts = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (parts.size() < 3) {
            continue;
        }

        bool okA = false;
        bool okB = false;
        bool okC = false;
        unsigned int a = parts[0].toUInt(&okA);
        unsigned int b = parts[1].toUInt(&okB);
        unsigned int c = parts[2].toUInt(&okC);
        if (!okA || !okB || !okC) {
            continue;
        }

        if (!formatDetermined) {
            isMatlabFormat = (a > 0 && b > 0 && c > 0);
            formatDetermined = true;
        }

        if (isMatlabFormat) {
            --a;
            --b;
            --c;
        }

        if (a < static_cast<unsigned int>(outVertices->size()) &&
            b < static_cast<unsigned int>(outVertices->size()) &&
            c < static_cast<unsigned int>(outVertices->size())) {
            outIndices->append(a);
            outIndices->append(b);
            outIndices->append(c);
        }
    }
    facesFile.close();

    if (outIndices->isEmpty()) {
        if (outError) {
            *outError = "No valid faces found in faces TXT.";
        }
        return false;
    }

    return true;
}

struct LoadedMeshResult {
    QVector<QVector3D> vertices;
    QVector<unsigned int> indices;
    QString error;
    QString sourceDescription;
    bool ok = false;
};

int inferForegroundPolarityHint(const QVector<cv::Mat>& slices16) {
    if (slices16.isEmpty()) {
        return 0;
    }

    int darkVotes = 0;
    int brightVotes = 0;
    const int zStep = qMax(1, slices16.size() / 24);

    for (int z = 0; z < slices16.size(); z += zStep) {
        const cv::Mat& src = slices16[z];
        if (src.empty()) {
            continue;
        }

        cv::Mat u16;
        if (src.type() == CV_16UC1) {
            u16 = src;
        } else {
            src.convertTo(u16, CV_16UC1);
        }

        if (u16.empty()) {
            continue;
        }

        const int rows = u16.rows;
        const int cols = u16.cols;
        const int cx0 = cols / 4;
        const int cx1 = cols - cx0;
        const int cy0 = rows / 4;
        const int cy1 = rows - cy0;
        const int bx = qMax(1, cols / 10);
        const int by = qMax(1, rows / 10);

        double centerSum = 0.0;
        qint64 centerCount = 0;
        double borderSum = 0.0;
        qint64 borderCount = 0;

        const int yStep = qMax(1, rows / 200);
        const int xStep = qMax(1, cols / 200);
        for (int y = 0; y < rows; y += yStep) {
            const quint16* row = u16.ptr<quint16>(y);
            for (int x = 0; x < cols; x += xStep) {
                const quint16 v = row[x];
                const bool inCenter = (x >= cx0 && x < cx1 && y >= cy0 && y < cy1);
                const bool inBorder = (x < bx || x >= cols - bx || y < by || y >= rows - by);
                if (inCenter) {
                    centerSum += double(v);
                    ++centerCount;
                }
                if (inBorder) {
                    borderSum += double(v);
                    ++borderCount;
                }
            }
        }

        if (centerCount == 0 || borderCount == 0) {
            continue;
        }

        const double centerMean = centerSum / double(centerCount);
        const double borderMean = borderSum / double(borderCount);

        if (centerMean < borderMean) {
            ++darkVotes;
        } else {
            ++brightVotes;
        }
    }

    if (darkVotes >= brightVotes + 2) {
        return 1;
    }
    if (brightVotes >= darkVotes + 2) {
        return 2;
    }
    return 0;
}

double meanAbsDiffSampled(const QImage& imgA, const QImage& imgB) {
    if (imgA.size() != imgB.size() || imgA.isNull() || imgB.isNull()) {
        return 255.0;
    }

    const QImage grayA = (imgA.format() == QImage::Format_Grayscale8) ? imgA : imgA.convertToFormat(QImage::Format_Grayscale8);
    const QImage grayB = (imgB.format() == QImage::Format_Grayscale8) ? imgB : imgB.convertToFormat(QImage::Format_Grayscale8);

    const int sampleStepX = qMax(1, grayA.width() / 256);
    const int sampleStepY = qMax(1, grayA.height() / 256);

    qint64 accum = 0;
    qint64 count = 0;
    for (int y = 0; y < grayA.height(); y += sampleStepY) {
        const uchar* lineA = grayA.constScanLine(y);
        const uchar* lineB = grayB.constScanLine(y);
        for (int x = 0; x < grayA.width(); x += sampleStepX) {
            accum += qAbs(int(lineA[x]) - int(lineB[x]));
            ++count;
        }
    }

    return count > 0 ? (double(accum) / double(count)) : 0.0;
}

QImage cvMatToDisplayableQImage(const cv::Mat& input) {
    if (input.empty()) {
        return QImage();
    }

    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input;
    } else if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
    } else {
        return QImage();
    }

    cv::Mat normalized8;
    if (gray.depth() == CV_8U) {
        normalized8 = gray;
    } else {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(gray, &minValue, &maxValue);

        if (maxValue <= minValue + std::numeric_limits<double>::epsilon()) {
            normalized8 = cv::Mat(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));
        } else {
            const double scale = 255.0 / (maxValue - minValue);
            const double shift = -minValue * scale;
            gray.convertTo(normalized8, CV_8UC1, scale, shift);
        }
    }

    return MainWindow::cvMatToQImage(normalized8);
}

cv::Mat qImageToSegmentationU16(const QImage& image) {
    if (image.isNull()) {
        return cv::Mat();
    }

    const QImage gray = (image.format() == QImage::Format_Grayscale8)
        ? image
        : image.convertToFormat(QImage::Format_Grayscale8);

    cv::Mat mat8(gray.height(), gray.width(), CV_8UC1,
                 const_cast<uchar*>(gray.constBits()), gray.bytesPerLine());
    cv::Mat mat16;
    mat8.convertTo(mat16, CV_16UC1, 257.0);
    return mat16;
}

cv::Rect detectSegmentationCropRect(const cv::Mat& src) {
    if (src.empty()) {
        return cv::Rect();
    }

    cv::Mat gray = src;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else if (src.channels() == 4) {
        cv::cvtColor(src, gray, cv::COLOR_BGRA2GRAY);
    }

    cv::Mat normalized8;
    if (gray.depth() == CV_16U) {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(gray, &minValue, &maxValue);
        if (maxValue > minValue) {
            gray.convertTo(normalized8, CV_8UC1, 255.0 / (maxValue - minValue), -minValue * 255.0 / (maxValue - minValue));
        } else {
            normalized8 = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(0));
        }
    } else if (gray.depth() == CV_8U) {
        normalized8 = gray;
    } else {
        gray.convertTo(normalized8, CV_8UC1);
    }

    cv::GaussianBlur(normalized8, normalized8, cv::Size(3, 3), 0.0, 0.0);

    cv::Mat binary;
    cv::threshold(normalized8, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return cv::Rect();
    }

    cv::Rect boundingRectUnion;
    bool hasValidContour = false;
    for (const auto& contour : contours) {
        const cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() < (src.cols * src.rows / 500)) {
            continue;
        }

        // Keep border-touching anatomy, but reject giant frame-like artifacts.
        const bool frameLike = (rect.width >= (src.cols * 9 / 10))
            || (rect.height >= (src.rows * 9 / 10));
        if (frameLike && rect.area() > (src.cols * src.rows * 3 / 5)) {
            continue;
        }

        if (!hasValidContour) {
            boundingRectUnion = rect;
            hasValidContour = true;
        } else {
            boundingRectUnion |= rect;
        }
    }

    if (!hasValidContour) {
        return cv::Rect();
    }

    const int marginX = qMax(4, boundingRectUnion.width / 30);
    const int marginY = qMax(4, boundingRectUnion.height / 30);
    const QRect cropRect(
        qMax(0, boundingRectUnion.x - marginX),
        qMax(0, boundingRectUnion.y - marginY),
        qMin(src.cols - qMax(0, boundingRectUnion.x - marginX), boundingRectUnion.width + 2 * marginX),
        qMin(src.rows - qMax(0, boundingRectUnion.y - marginY), boundingRectUnion.height + 2 * marginY)
    );

    if (cropRect.width() <= 0 || cropRect.height() <= 0) {
        return cv::Rect();
    }

    return cv::Rect(cropRect.x(), cropRect.y(), cropRect.width(), cropRect.height());
}

cv::Mat cropSegmentationSlice16(const cv::Mat& src, const cv::Rect& cropRect) {
    if (src.empty()) {
        return cv::Mat();
    }

    if (cropRect.width <= 0 || cropRect.height <= 0) {
        return src.clone();
    }

    const cv::Rect boundedRect = cropRect & cv::Rect(0, 0, src.cols, src.rows);
    if (boundedRect.width <= 0 || boundedRect.height <= 0) {
        return src.clone();
    }

    return src(boundedRect).clone();
}

int histogramPercentile(const std::vector<qint64>& histogram, qint64 total, double pct) {
    if (total <= 0) {
        return 0;
    }

    const qint64 target = qBound<qint64>(0, qint64(std::llround(double(total) * pct)), total - 1);
    qint64 accum = 0;
    for (int i = 0; i < int(histogram.size()); ++i) {
        accum += histogram[i];
        if (accum > target) {
            return i;
        }
    }
    return int(histogram.size()) - 1;
}

int computeOtsuThreshold16FromSlices(const QVector<cv::Mat>& slices16) {
    if (slices16.isEmpty()) {
        return 32768;
    }

    std::vector<qint64> histogram(65536, 0);
    qint64 totalPixels = 0;

    for (const cv::Mat& slice : slices16) {
        if (slice.empty()) {
            continue;
        }

        cv::Mat u16;
        if (slice.type() == CV_16UC1) {
            u16 = slice;
        } else {
            slice.convertTo(u16, CV_16UC1);
        }

        const int x0 = u16.cols / 10;
        const int y0 = u16.rows / 10;
        const int x1 = u16.cols - x0;
        const int y1 = u16.rows - y0;
        for (int y = y0; y < y1; ++y) {
            const quint16* row = reinterpret_cast<const quint16*>(u16.ptr(y));
            for (int x = x0; x < x1; ++x) {
                histogram[row[x]]++;
                ++totalPixels;
            }
        }
    }

    if (totalPixels <= 0) {
        return 32768;
    }

    const int p01 = histogramPercentile(histogram, totalPixels, 0.01);
    const int p99 = histogramPercentile(histogram, totalPixels, 0.99);
    if (p99 <= p01) {
        return 32768;
    }

    std::vector<qint64> clippedHistogram(65536, 0);
    qint64 clippedTotal = 0;
    for (int i = p01; i <= p99; ++i) {
        clippedHistogram[i] = histogram[i];
        clippedTotal += histogram[i];
    }
    if (clippedTotal <= 0) {
        return 32768;
    }

    double sumTotal = 0.0;
    for (int i = 0; i < 65536; ++i) {
        sumTotal += static_cast<double>(i) * static_cast<double>(clippedHistogram[i]);
    }

    qint64 sumBackground = 0;
    qint64 weightBackground = 0;
    double maxVariance = 0.0;
    int optimalThreshold = 32768;

    for (int t = 0; t < 65536; ++t) {
        weightBackground += clippedHistogram[t];
        if (weightBackground == 0) {
            continue;
        }

        const qint64 weightForeground = clippedTotal - weightBackground;
        if (weightForeground == 0) {
            break;
        }

        sumBackground += static_cast<qint64>(t) * clippedHistogram[t];

        const double muBackground = static_cast<double>(sumBackground) / static_cast<double>(weightBackground);
        const double muForeground = (sumTotal - static_cast<double>(sumBackground)) / static_cast<double>(weightForeground);
        const double variance = static_cast<double>(weightBackground) * static_cast<double>(weightForeground)
                                * (muBackground - muForeground) * (muBackground - muForeground);

        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }

    return optimalThreshold;
}

quint16 readDicomU16(const QByteArray& data, int offset, bool littleEndian) {
    const uchar* p = reinterpret_cast<const uchar*>(data.constData() + offset);
    if (littleEndian) {
        return quint16(quint16(p[0]) | (quint16(p[1]) << 8));
    }
    return quint16((quint16(p[0]) << 8) | quint16(p[1]));
}

quint32 readDicomU32(const QByteArray& data, int offset, bool littleEndian) {
    const uchar* p = reinterpret_cast<const uchar*>(data.constData() + offset);
    if (littleEndian) {
        return quint32(p[0]) | (quint32(p[1]) << 8) | (quint32(p[2]) << 16) | (quint32(p[3]) << 24);
    }
    return (quint32(p[0]) << 24) | (quint32(p[1]) << 16) | (quint32(p[2]) << 8) | quint32(p[3]);
}

bool dicomVrUsesLongLength(const QByteArray& vr) {
    return vr == "OB" || vr == "OD" || vr == "OF" || vr == "OL" || vr == "OW" || vr == "SQ" ||
           vr == "UC" || vr == "UR" || vr == "UT" || vr == "UN";
}

QString readDicomTextValue(const QByteArray& data, int offset, quint32 length) {
    QByteArray raw = data.mid(offset, int(length));
    const int zeroTerminator = raw.indexOf('\0');
    if (zeroTerminator >= 0) {
        raw.truncate(zeroTerminator);
    }
    return QString::fromLatin1(raw).trimmed();
}

double parseDicomDouble(const QString& text, double fallback) {
    const QStringList tokens = text.split('\\', Qt::SkipEmptyParts);
    const QString firstToken = tokens.isEmpty() ? text.trimmed() : tokens.first().trimmed();
    bool ok = false;
    const double value = firstToken.toDouble(&ok);
    return ok ? value : fallback;
}

bool decodeUncompressedDicomMonochromeRaw16(const QByteArray& bytes, cv::Mat* out16) {
    if (!out16 || bytes.size() < 8) {
        return false;
    }

    int pos = 0;
    if (bytes.size() >= 132 && bytes.mid(128, 4) == "DICM") {
        pos = 132;
    }

    bool parsingMetaHeader = (pos == 132);
    bool datasetLittleEndian = true;

    bool datasetExplicitVR = true;
    if (pos == 0 && bytes.size() >= 6) {
        const uchar vr0 = static_cast<uchar>(bytes[4]);
        const uchar vr1 = static_cast<uchar>(bytes[5]);
        datasetExplicitVR = (vr0 >= 'A' && vr0 <= 'Z' && vr1 >= 'A' && vr1 <= 'Z');
    }

    int rows = 0;
    int cols = 0;
    int bitsAllocated = 0;
    int samplesPerPixel = 1;
    int pixelRepresentation = 0;
    QString photometricInterpretation = "MONOCHROME2";
    int pixelDataOffset = -1;
    quint32 pixelDataLength = 0;

    while (pos + 8 <= bytes.size()) {
        if (parsingMetaHeader && readDicomU16(bytes, pos, true) != 0x0002) {
            parsingMetaHeader = false;
            continue;
        }

        const bool explicitVR = parsingMetaHeader ? true : datasetExplicitVR;
        const bool littleEndian = parsingMetaHeader ? true : datasetLittleEndian;

        if (pos + 8 > bytes.size()) {
            break;
        }

        const quint16 group = readDicomU16(bytes, pos, littleEndian);
        const quint16 element = readDicomU16(bytes, pos + 2, littleEndian);
        pos += 4;

        quint32 valueLength = 0;
        if (explicitVR) {
            if (pos + 2 > bytes.size()) {
                break;
            }
            const QByteArray vr = bytes.mid(pos, 2);
            pos += 2;

            if (dicomVrUsesLongLength(vr)) {
                if (pos + 6 > bytes.size()) {
                    break;
                }
                pos += 2;
                valueLength = readDicomU32(bytes, pos, littleEndian);
                pos += 4;
            } else {
                if (pos + 2 > bytes.size()) {
                    break;
                }
                valueLength = readDicomU16(bytes, pos, littleEndian);
                pos += 2;
            }
        } else {
            if (pos + 4 > bytes.size()) {
                break;
            }
            valueLength = readDicomU32(bytes, pos, littleEndian);
            pos += 4;
        }

        if (valueLength == 0xFFFFFFFFu) {
            if (group == 0x7FE0 && element == 0x0010) {
                return false;
            }

            int seqDepth = 1;
            while (pos + 8 <= bytes.size() && seqDepth > 0) {
                if (static_cast<uchar>(bytes[pos]) == 0xFE &&
                    static_cast<uchar>(bytes[pos + 1]) == 0xFF) {
                    const quint16 delimElem = readDicomU16(bytes, pos + 2, true);
                    const quint32 delimLen = readDicomU32(bytes, pos + 4, true);
                    pos += 8;
                    if (delimElem == 0xE000) {
                        if (delimLen != 0xFFFFFFFFu && delimLen <= quint32(bytes.size() - pos)) {
                            pos += int(delimLen);
                        } else if (delimLen == 0xFFFFFFFFu) {
                            seqDepth++;
                        }
                    } else if (delimElem == 0xE00D || delimElem == 0xE0DD) {
                        seqDepth--;
                    }
                } else {
                    pos++;
                }
            }
            if (seqDepth != 0) {
                return false;
            }
            continue;
        }

        if (valueLength > quint32(bytes.size() - pos)) {
            break;
        }

        const int valueOffset = pos;
        pos += int(valueLength);

        if (group == 0x0002 && element == 0x0010) {
            const QString transferSyntax = readDicomTextValue(bytes, valueOffset, valueLength);
            if (transferSyntax == "1.2.840.10008.1.2") {
                datasetExplicitVR = false;
                datasetLittleEndian = true;
            } else if (transferSyntax == "1.2.840.10008.1.2.1") {
                datasetExplicitVR = true;
                datasetLittleEndian = true;
            } else {
                return false;
            }
        } else if (group == 0x0028 && element == 0x0002 && valueLength >= 2) {
            samplesPerPixel = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0004) {
            photometricInterpretation = readDicomTextValue(bytes, valueOffset, valueLength);
        } else if (group == 0x0028 && element == 0x0010 && valueLength >= 2) {
            rows = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0011 && valueLength >= 2) {
            cols = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0100 && valueLength >= 2) {
            bitsAllocated = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0103 && valueLength >= 2) {
            pixelRepresentation = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x7FE0 && element == 0x0010) {
            pixelDataOffset = valueOffset;
            pixelDataLength = valueLength;
            break;
        }
    }

    if (pixelDataOffset < 0 || rows <= 0 || cols <= 0 || !datasetLittleEndian || samplesPerPixel != 1) {
        return false;
    }
    if (bitsAllocated != 8 && bitsAllocated != 16) {
        return false;
    }

    const int bytesPerPixel = bitsAllocated / 8;
    const qint64 pixelCount = qint64(rows) * qint64(cols);
    const qint64 requiredBytes = pixelCount * bytesPerPixel;
    if (requiredBytes <= 0 || pixelDataLength < quint32(requiredBytes)) {
        return false;
    }

    const bool invert = (photometricInterpretation.trimmed().toUpper() == "MONOCHROME1");
    cv::Mat raw16(rows, cols, CV_16UC1);
    if (raw16.empty()) {
        return false;
    }

    qint64 pixelIndex = 0;
    for (int y = 0; y < rows; ++y) {
        quint16* line = reinterpret_cast<quint16*>(raw16.ptr(y));
        for (int x = 0; x < cols; ++x, ++pixelIndex) {
            const int sampleOffset = pixelDataOffset + int(pixelIndex * bytesPerPixel);
            quint16 value16 = 0;

            if (bitsAllocated == 8) {
                const uchar raw = static_cast<uchar>(bytes.at(sampleOffset));
                if (pixelRepresentation == 0) {
                    value16 = quint16(raw) * 257u;
                } else {
                    const int signedRaw = int(static_cast<qint8>(raw));
                    value16 = quint16(qBound(0, signedRaw + 128, 255) * 257);
                }
            } else {
                const quint16 word = readDicomU16(bytes, sampleOffset, true);
                if (pixelRepresentation == 0) {
                    value16 = word;
                } else {
                    const int signedWord = int(static_cast<qint16>(word));
                    value16 = quint16(qBound(0, signedWord + 32768, 65535));
                }
            }

            line[x] = invert ? quint16(65535u - value16) : value16;
        }
    }

    *out16 = raw16;
    return true;
}

bool decodeSegmentationSlice16(const QString& path, cv::Mat* out16, bool* native16 = nullptr) {
    if (!out16) {
        return false;
    }

    if (native16) {
        *native16 = false;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }

    const QByteArray bytes = file.readAll();
    if (bytes.isEmpty()) {
        return false;
    }

    std::vector<uchar> buffer(bytes.begin(), bytes.end());
    cv::Mat decoded = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (!decoded.empty()) {
        cv::Mat gray;
        if (decoded.channels() == 1) {
            gray = decoded;
        } else if (decoded.channels() == 3) {
            cv::cvtColor(decoded, gray, cv::COLOR_BGR2GRAY);
        } else if (decoded.channels() == 4) {
            cv::cvtColor(decoded, gray, cv::COLOR_BGRA2GRAY);
        }

        if (!gray.empty()) {
            if (gray.depth() == CV_16U) {
                *out16 = gray.clone();
                if (native16) {
                    *native16 = true;
                }
                return true;
            }

            if (gray.depth() == CV_8U) {
                gray.convertTo(*out16, CV_16UC1, 257.0);
                return true;
            }
        }
    }

    if (decodeUncompressedDicomMonochromeRaw16(bytes, out16)) {
        if (native16) {
            *native16 = true;
        }
        return true;
    }

    return false;
}

QImage decodeUncompressedDicomMonochrome(const QByteArray& bytes) {
    if (bytes.size() < 8) {
        return QImage();
    }

    int pos = 0;
    if (bytes.size() >= 132 && bytes.mid(128, 4) == "DICM") {
        pos = 132;
    }

    bool parsingMetaHeader = (pos == 132);
    bool datasetLittleEndian = true;

    // For files without a DICM preamble (ACR-NEMA / legacy DICOM), detect VR
    // encoding by probing the 2 bytes immediately after the first 4-byte tag.
    // In explicit VR those bytes are an uppercase two-letter VR code (e.g. "UL",
    // "CS").  In implicit VR little-endian (the historical default) they are
    // the low two bytes of a 4-byte binary length, which virtually never looks
    // like two uppercase ASCII letters.
    bool datasetExplicitVR = true;
    if (pos == 0 && bytes.size() >= 6) {
        const uchar vr0 = static_cast<uchar>(bytes[4]);
        const uchar vr1 = static_cast<uchar>(bytes[5]);
        const bool explicitVrProbe = (vr0 >= 'A' && vr0 <= 'Z' && vr1 >= 'A' && vr1 <= 'Z');
        datasetExplicitVR = explicitVrProbe;
    }

    int rows = 0;
    int cols = 0;
    int bitsAllocated = 0;
    int samplesPerPixel = 1;
    int pixelRepresentation = 0;
    QString photometricInterpretation = "MONOCHROME2";
    double rescaleSlope = 1.0;
    double rescaleIntercept = 0.0;

    int pixelDataOffset = -1;
    quint32 pixelDataLength = 0;

    while (pos + 8 <= bytes.size()) {
        if (parsingMetaHeader && readDicomU16(bytes, pos, true) != 0x0002) {
            parsingMetaHeader = false;
            continue;
        }

        const bool explicitVR = parsingMetaHeader ? true : datasetExplicitVR;
        const bool littleEndian = parsingMetaHeader ? true : datasetLittleEndian;

        if (pos + 8 > bytes.size()) {
            break;
        }

        const quint16 group = readDicomU16(bytes, pos, littleEndian);
        const quint16 element = readDicomU16(bytes, pos + 2, littleEndian);
        pos += 4;

        quint32 valueLength = 0;
        if (explicitVR) {
            if (pos + 2 > bytes.size()) {
                break;
            }
            const QByteArray vr = bytes.mid(pos, 2);
            pos += 2;

            if (dicomVrUsesLongLength(vr)) {
                if (pos + 6 > bytes.size()) {
                    break;
                }
                pos += 2; // reserved bytes
                valueLength = readDicomU32(bytes, pos, littleEndian);
                pos += 4;
            } else {
                if (pos + 2 > bytes.size()) {
                    break;
                }
                valueLength = readDicomU16(bytes, pos, littleEndian);
                pos += 2;
            }
        } else {
            if (pos + 4 > bytes.size()) {
                break;
            }
            valueLength = readDicomU32(bytes, pos, littleEndian);
            pos += 4;
        }

        if (valueLength == 0xFFFFFFFFu) {
            if (group == 0x7FE0 && element == 0x0010) {
                // Encapsulated (compressed) pixel data — not supported by this decoder.
                return QImage();
            }
            // Undefined-length sequence or item in the header: scan forward using FFFE
            // delimiter tags (group 0xFFFE is always little-endian in DICOM).
            int seqDepth = 1;
            while (pos + 8 <= bytes.size() && seqDepth > 0) {
                if (static_cast<uchar>(bytes[pos])     == 0xFE &&
                    static_cast<uchar>(bytes[pos + 1]) == 0xFF) {
                    const quint16 delimElem = readDicomU16(bytes, pos + 2, true);
                    const quint32 delimLen  = readDicomU32(bytes, pos + 4, true);
                    pos += 8;
                    if (delimElem == 0xE000) { // Item start
                        if (delimLen != 0xFFFFFFFFu && delimLen <= quint32(bytes.size() - pos)) {
                            pos += int(delimLen); // defined-length item: jump past content
                        } else if (delimLen == 0xFFFFFFFFu) {
                            seqDepth++; // undefined-length item needs its own E00D delimiter
                        }
                    } else if (delimElem == 0xE00D || delimElem == 0xE0DD) {
                        seqDepth--;
                    }
                } else {
                    pos++;
                }
            }
            if (seqDepth != 0) {
                return QImage(); // Unterminated sequence — bail out
            }
            continue;
        }

        if (valueLength > quint32(bytes.size() - pos)) {
            break;
        }

        const int valueOffset = pos;
        pos += int(valueLength);

        if (group == 0x0002 && element == 0x0010) {
            const QString transferSyntax = readDicomTextValue(bytes, valueOffset, valueLength);
            if (transferSyntax == "1.2.840.10008.1.2") {
                datasetExplicitVR = false;
                datasetLittleEndian = true;
            } else if (transferSyntax == "1.2.840.10008.1.2.1") {
                datasetExplicitVR = true;
                datasetLittleEndian = true;
            } else if (transferSyntax == "1.2.840.10008.1.2.2") {
                // Explicit VR Big Endian transfer syntax is not handled by this parser.
                return QImage();
            } else if (transferSyntax.startsWith("1.2.840.10008.1.2.4") || transferSyntax == "1.2.840.10008.1.2.5") {
                // JPEG/JPEG2000/RLE compressed pixel data is out-of-scope for this lightweight decoder.
                return QImage();
            }
        } else if (group == 0x0028 && element == 0x0002 && valueLength >= 2) {
            samplesPerPixel = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0004) {
            photometricInterpretation = readDicomTextValue(bytes, valueOffset, valueLength);
        } else if (group == 0x0028 && element == 0x0010 && valueLength >= 2) {
            rows = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0011 && valueLength >= 2) {
            cols = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0100 && valueLength >= 2) {
            bitsAllocated = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x0103 && valueLength >= 2) {
            pixelRepresentation = int(readDicomU16(bytes, valueOffset, littleEndian));
        } else if (group == 0x0028 && element == 0x1052) {
            rescaleIntercept = parseDicomDouble(readDicomTextValue(bytes, valueOffset, valueLength), 0.0);
        } else if (group == 0x0028 && element == 0x1053) {
            rescaleSlope = parseDicomDouble(readDicomTextValue(bytes, valueOffset, valueLength), 1.0);
        } else if (group == 0x7FE0 && element == 0x0010) {
            pixelDataOffset = valueOffset;
            pixelDataLength = valueLength;
            break;
        }
    }

    if (pixelDataOffset < 0 || rows <= 0 || cols <= 0) {
        return QImage();
    }
    if (!datasetLittleEndian) {
        return QImage();
    }
    if (samplesPerPixel != 1) {
        return QImage();
    }
    if (bitsAllocated != 8 && bitsAllocated != 16) {
        return QImage();
    }

    if (!std::isfinite(rescaleSlope) || qFuzzyIsNull(rescaleSlope)) {
        rescaleSlope = 1.0;
    }
    if (!std::isfinite(rescaleIntercept)) {
        rescaleIntercept = 0.0;
    }

    const int bytesPerPixel = bitsAllocated / 8;
    const qint64 pixelCount = qint64(rows) * qint64(cols);
    const qint64 requiredBytes = pixelCount * bytesPerPixel;
    if (requiredBytes <= 0 || pixelDataLength < quint32(requiredBytes)) {
        return QImage();
    }

    auto readScaledValue = [&](qint64 index) -> double {
        const int sampleOffset = pixelDataOffset + int(index * bytesPerPixel);
        if (bitsAllocated == 8) {
            const uchar raw = static_cast<uchar>(bytes.at(sampleOffset));
            const int signedRaw = static_cast<qint8>(raw);
            const double value = (pixelRepresentation == 0) ? double(raw) : double(signedRaw);
            return value * rescaleSlope + rescaleIntercept;
        }

        const quint16 word = readDicomU16(bytes, sampleOffset, true);
        const int signedWord = static_cast<qint16>(word);
        const double value = (pixelRepresentation == 0) ? double(word) : double(signedWord);
        return value * rescaleSlope + rescaleIntercept;
    };

    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    for (qint64 i = 0; i < pixelCount; ++i) {
        const double value = readScaledValue(i);
        minValue = qMin(minValue, value);
        maxValue = qMax(maxValue, value);
    }

    if (!std::isfinite(minValue) || !std::isfinite(maxValue)) {
        return QImage();
    }

    const bool invert = (photometricInterpretation.trimmed().toUpper() == "MONOCHROME1");
    const double dynamicRange = maxValue - minValue;
    const double scale = (dynamicRange > std::numeric_limits<double>::epsilon()) ? (255.0 / dynamicRange) : 0.0;

    QImage image(cols, rows, QImage::Format_Grayscale8);
    if (image.isNull()) {
        return QImage();
    }

    qint64 pixelIndex = 0;
    for (int y = 0; y < rows; ++y) {
        uchar* line = image.scanLine(y);
        for (int x = 0; x < cols; ++x, ++pixelIndex) {
            const double value = readScaledValue(pixelIndex);
            int gray = (scale > 0.0) ? int((value - minValue) * scale + 0.5) : 0;
            gray = qBound(0, gray, 255);
            line[x] = static_cast<uchar>(invert ? (255 - gray) : gray);
        }
    }

    return image;
}

QImage loadSliceImage(const QString& path, bool* usedFallback = nullptr) {
    if (usedFallback) {
        *usedFallback = false;
    }

    QImage image;
    if (image.load(path)) {
        return image;
    }

    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return QImage();
    }

    const QByteArray bytes = file.readAll();
    if (bytes.isEmpty()) {
        return QImage();
    }

    std::vector<uchar> buffer(bytes.begin(), bytes.end());
    cv::Mat decoded = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
    if (!decoded.empty()) {
        const QImage converted = cvMatToDisplayableQImage(decoded);
        if (!converted.isNull()) {
            if (usedFallback) {
                *usedFallback = true;
            }
            return converted;
        }
    }

    const QImage dicomImage = decodeUncompressedDicomMonochrome(bytes);
    if (!dicomImage.isNull()) {
        if (usedFallback) {
            *usedFallback = true;
        }
        return dicomImage;
    }

    return QImage();
}
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
    glWidget(new GLWidget(this))
{
    // Load stylesheet first
    QFile styleFile(":/styles/styles.qss");
    if(styleFile.open(QIODevice::ReadOnly)) {
        QString styleSheet = QLatin1String(styleFile.readAll());
        qApp->setStyleSheet(styleSheet);
    }

    // Create main tab widget
    mainTabs = new QTabWidget();
    setCentralWidget(mainTabs);

    // Create visualization tab
    vizTab = new QWidget();
    QVBoxLayout* vizLayout = new QVBoxLayout(vizTab);
    vizLayout->setContentsMargins(0, 0, 0, 0);
    vizLayout->setSpacing(0);
    vizLayout->addWidget(glWidget);
    mainTabs->addTab(vizTab, "3D Visualization");

    // Overlay legend — floats at top-right inside the 3D view, hidden until material colors on
    overlayLegendWidget = new QWidget(vizTab);
    overlayLegendWidget->setAttribute(Qt::WA_TranslucentBackground, false);
    overlayLegendWidget->setStyleSheet(
        "background-color: rgba(30, 30, 30, 210);"
        "border: 1px solid #555;"
        "border-radius: 6px;"
    );
    QVBoxLayout* overlayLayout = new QVBoxLayout(overlayLegendWidget);
    overlayLayout->setContentsMargins(10, 8, 12, 8);
    overlayLayout->setSpacing(6);
    // Row 1: Ceramic Bone
    QHBoxLayout* row1Layout = new QHBoxLayout();
    row1Layout->setSpacing(6);
    QLabel* oSwatch = new QLabel(overlayLegendWidget);
    oSwatch->setFixedSize(14, 14);
    oSwatch->setStyleSheet("background-color: rgb(50, 130, 255); border: 1px solid #888; border-radius: 2px;");
    QLabel* oLabel = new QLabel("Ceramic Bone", overlayLegendWidget);
    oLabel->setStyleSheet("color: #e0e0e0; font-size: 12px; background: transparent;");
    row1Layout->addWidget(oSwatch);
    row1Layout->addWidget(oLabel);
    overlayLayout->addLayout(row1Layout);
    // Row 2: Bone (non-colorized)
    QHBoxLayout* row2Layout = new QHBoxLayout();
    row2Layout->setSpacing(6);
    QLabel* boneSwatch = new QLabel(overlayLegendWidget);
    boneSwatch->setFixedSize(14, 14);
    boneSwatch->setStyleSheet("background-color: rgb(180, 180, 180); border: 1px solid #888; border-radius: 2px;");
    QLabel* boneLabel = new QLabel("Bone", overlayLegendWidget);
    boneLabel->setStyleSheet("color: #e0e0e0; font-size: 12px; background: transparent;");
    row2Layout->addWidget(boneSwatch);
    row2Layout->addWidget(boneLabel);
    overlayLayout->addLayout(row2Layout);
    overlayLegendWidget->adjustSize();
    overlayLegendWidget->setVisible(false);
    overlayLegendWidget->raise();
    vizTab->installEventFilter(this);

    // Create preview tab
    createPreviewTab();

    // Rest of initialization
    createMenu();
    createToolbar();
    setWindowTitle("MRI 3D Visualizer");
    resize(1280, 720);
    statusBar()->showMessage("Ready", 2000);

    loadingDialog = new QProgressDialog(this);
    loadingDialog->setWindowModality(Qt::WindowModal);
    loadingDialog->setWindowTitle("Mesh Generation Progress");
    loadingDialog->setLabelText("Preparing mesh generation...");
    loadingDialog->setCancelButton(nullptr);
    loadingDialog->setRange(0, 100);
    loadingDialog->setValue(0);
    loadingDialog->setMinimumDuration(0);
    loadingDialog->setAutoClose(false);
    loadingDialog->setAutoReset(false);
    loadingDialog->reset();

    connect(&meshGenerationWatcher, &QFutureWatcher<void>::started,
            this, &MainWindow::handleMeshGenerationStarted);
    connect(&meshGenerationWatcher, &QFutureWatcher<void>::finished,
            this, &MainWindow::handleMeshComputationFinished);
        connect(&otsuWatcher, &QFutureWatcher<int>::finished,
            this, &MainWindow::onOtsuComputationFinished);
    connect(glWidget, &GLWidget::meshUpdateComplete, // New signal
            this, &MainWindow::handleMeshRenderingFinished);
        connect(&previewBuildTimer, &QTimer::timeout,
            this, &MainWindow::appendPreviewThumbnails);
}

MainWindow::~MainWindow()
{
    delete loadingDialog;

}

void MainWindow::createPreviewTab() {
    previewTab = new QWidget();
    previewScroll = new QScrollArea(previewTab);
    previewContainer = new QWidget();
    previewLayout = new QGridLayout(previewContainer);

    previewScroll->setWidget(previewContainer);
    previewScroll->setWidgetResizable(true);

    QVBoxLayout* tabLayout = new QVBoxLayout(previewTab);
    tabLayout->addWidget(previewScroll);

    mainTabs->addTab(previewTab, "Image Preview");
}

void MainWindow::createMenu() {
    QMenuBar* menuBar = new QMenuBar(this);

    // File Menu
    QMenu* fileMenu = menuBar->addMenu("&File");
    QAction* openAct = fileMenu->addAction(QIcon(":/icons/open.png"), "&Open...");
    openAct->setShortcut(QKeySequence::Open);
    fileMenu->addSeparator();
    QAction* exitAct = fileMenu->addAction("E&xit");

    // View Menu
    QMenu* viewMenu = menuBar->addMenu("&View");
    QAction* toggleControlsAct = viewMenu->addAction("Show Controls");
    toggleControlsAct->setCheckable(true);
    toggleControlsAct->setChecked(true);

    // Connect signals
    connect(openAct, &QAction::triggered, this, &MainWindow::openDataset);
    connect(toggleControlsAct, &QAction::toggled, this, &MainWindow::toggleControls);
    connect(exitAct, &QAction::triggered, qApp, &QApplication::quit);

    setMenuBar(menuBar);
}

void MainWindow::createToolbar() {
    toolBar = addToolBar("Main Toolbar");
    toolBar->setIconSize(QSize(32, 32));
    toolBar->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
    toolBar->setMovable(false);

    // Create actions with consistent properties
    auto createAction = [](const QString& text, const QString& iconPath) {
        QAction* act = new QAction(QIcon(iconPath), text, nullptr);
        act->setStatusTip(text);
        return act;
    };

    QAction* openAct = createAction("Open Dataset", ":icons/icons/open.png");
    generate3DAct = createAction("Generate 3D", ":icons/icons/cube.png");
    QAction* resetAct = createAction("Reset View", ":icons/icons/reset.png");
    QAction* loadMeshAct = createAction("Load Mesh", ":icons/icons/up-loading.png");
    exportSTLAct = new QAction(QIcon(":icons/icons/material.png"), "Export STL", this);  // New export

    // Add to toolbar with spacers
    QWidget* leadingSpacer = new QWidget(this);
    leadingSpacer->setFixedWidth(5);
    toolBar->addWidget(leadingSpacer);

    toolBar->addAction(openAct);
    toolBar->addSeparator();
    toolBar->addAction(generate3DAct);
    toolBar->addSeparator();
    toolBar->addAction(resetAct);
    toolBar->addSeparator();
    toolBar->addAction(loadMeshAct);
    toolBar->addSeparator();
    toolBar->addAction(exportSTLAct);  // Add export button

    // Connect signals
    connect(openAct, &QAction::triggered, this, &MainWindow::openDataset);
    connect(generate3DAct, &QAction::triggered, this, &MainWindow::generateMesh);
    connect(resetAct, &QAction::triggered, [this]() { glWidget->resetView(); });
    connect(loadMeshAct, &QAction::triggered, this, &MainWindow::loadMesh);
    connect(exportSTLAct, &QAction::triggered, this, &MainWindow::exportSTL);

    // Segmentation controls (single 16-bit mode)
    thresholdLabel = new QLabel("Threshold:", this);
    thresholdLabel->setMinimumWidth(70);

    threshold16SpinBox = new QSpinBox(this);
    threshold16SpinBox->setRange(0, 65535);
    threshold16SpinBox->setSingleStep(64);
    threshold16SpinBox->setValue(currentThreshold16);
    threshold16SpinBox->setMinimumWidth(88);
    threshold16SpinBox->setMaximumWidth(106);

    threshold16AutoButton = new QPushButton("Auto Threshold (Otsu)", this);
    threshold16AutoButton->setMinimumWidth(150);
    threshold16AutoButton->setVisible(true);

    // Create container widget for segmentation controls
    QWidget* segmentationWidget = new QWidget(this);
    QVBoxLayout* segmentationLayout = new QVBoxLayout(segmentationWidget);
    segmentationLayout->setContentsMargins(0, 2, 0, 2);
    segmentationLayout->setSpacing(3);

    // First row: Threshold controls
    QHBoxLayout* controlsLayout = new QHBoxLayout();
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->setSpacing(5);
    controlsLayout->addWidget(thresholdLabel, 0);
    controlsLayout->addWidget(threshold16SpinBox, 0);
    controlsLayout->addWidget(threshold16AutoButton, 0);
    controlsLayout->addStretch(1);

    segmentationLayout->addLayout(controlsLayout);

    // Second row: Material Colors checkbox sits directly under the Threshold label.
    materialColorsCheckBox = new QCheckBox("Material Colors", this);
    materialColorsCheckBox->setChecked(false);
    materialColorsCheckBox->setEnabled(false);
    segmentationLayout->addWidget(materialColorsCheckBox);

    // (Legend is shown inside the 3D view as an overlay; no toolbar legend)

    toolBar->addSeparator();
    QWidget* toolbarSpacer = new QWidget(this);
    toolbarSpacer->setFixedWidth(28);
    toolBar->addWidget(toolbarSpacer);
    toolBar->addWidget(segmentationWidget);

    connect(threshold16SpinBox,
            QOverload<int>::of(&QSpinBox::valueChanged),
            this,
            &MainWindow::onThreshold16Changed);
    connect(threshold16AutoButton, &QPushButton::clicked, this, &MainWindow::onThreshold16Auto);

    // Add material visualization controls
    addMaterialVisualizationControls();

    syncThresholdControls();
}


void MainWindow::openDataset() {
    if (isExportingStl) {
        statusBar()->showMessage("STL export is still running. Please wait.", 3000);
        return;
    }

    QFileDialog dialog(this);
    dialog.setWindowTitle("Load MRI Dataset");
    dialog.setFileMode(QFileDialog::Directory); // Allow selecting a directory
    dialog.setOption(QFileDialog::ShowDirsOnly, true); // Only show directories

    if (dialog.exec() == QDialog::Accepted) {
        const QStringList selected = dialog.selectedFiles();
        if (selected.isEmpty()) {
            QMessageBox::warning(this, "No Folder Selected", "No dataset folder was selected.");
            return;
        }
        QString folderPath = selected.first(); // Get the selected folder path
        QDir dir(folderPath);

        // Filter for image files and enforce numeric ordering by slice number in filenames.
        QStringList filters = {"*.bmp", "*.dcm", "*.dicom", "*.png", "*.jpg", "*.tif", "*.tiff"};

        QFileInfoList files = dir.entryInfoList(filters, QDir::Files, QDir::NoSort);

        // Projection folders (raw acquisition frames) can look numerically valid but
        // do not represent tomographic Z slices. Auto-switch to a reconstruction folder when found.
        const QString recCandidate = findLikelyReconstructionDir(dir);
        const bool hasRecCandidate = !recCandidate.isEmpty();
        if (looksLikeProjectionSeries(files, dir, hasRecCandidate)) {
            QString message = "The selected folder looks like projection/raw acquisition data, not reconstructed slices.";
            if (hasRecCandidate) {
                message += "\n\nSwitch to the detected reconstruction folder?\n" + recCandidate;
            } else {
                message += "\n\nPlease select a reconstruction folder (often *_rec / recon / reconstruction).";
            }

            if (!hasRecCandidate) {
                QMessageBox::warning(this, "Projection Folder Detected", message);
                return;
            }

            const QMessageBox::StandardButton choice = QMessageBox::question(
                this,
                "Projection Folder Detected",
                message,
                QMessageBox::Yes | QMessageBox::No,
                QMessageBox::Yes
            );

            if (choice == QMessageBox::Yes) {
                dir = QDir(recCandidate);
                folderPath = recCandidate;
                files = dir.entryInfoList(filters, QDir::Files, QDir::NoSort);
                qDebug() << "Auto-switched dataset folder to reconstruction path:" << recCandidate
                         << "| image count:" << files.size();
            }
        }

        int trailingNumericCount = 0;
        for (const QFileInfo& fileInfo : files) {
            bool hasTrailingNumber = false;
            extractLastNumber(fileInfo.completeBaseName(), &hasTrailingNumber);
            if (hasTrailingNumber) {
                ++trailingNumericCount;
            }
        }

        // If most files are numbered slices, drop non-numbered helpers (for example *_arc.tif).
        if (!files.isEmpty() && trailingNumericCount >= qMax(16, int(files.size() * 0.70))) {
            QFileInfoList numericOnly;
            numericOnly.reserve(trailingNumericCount);
            for (const QFileInfo& fileInfo : files) {
                bool hasTrailingNumber = false;
                extractLastNumber(fileInfo.completeBaseName(), &hasTrailingNumber);
                if (hasTrailingNumber) {
                    numericOnly.append(fileInfo);
                }
            }
            if (!numericOnly.isEmpty()) {
                qDebug() << "Filtered non-slice files:" << (files.size() - numericOnly.size())
                         << "kept" << numericOnly.size() << "numbered slices";
                files = numericOnly;
            }
        }

        std::sort(files.begin(), files.end(), numericFileLess);

        QStringList filePaths;
        filePaths.reserve(files.size());
        for (const QFileInfo& fileInfo : files) {
            filePaths.append(fileInfo.absoluteFilePath());
        }

        if (!files.isEmpty()) {
            qDebug() << "Slice ordering sample:" << files.first().fileName() << "..." << files.last().fileName();
        }

        if (!filePaths.isEmpty()) {
            loadImages(filePaths); // Load all images in the folder
        } else {
            QMessageBox::warning(this, "No Images Found", "The selected folder does not contain any valid images.");
        }
    }
}

void MainWindow::loadImages(const QStringList& filePaths) {
    importTimer.start();
    currentImagePaths = filePaths;
    syncThresholdControls();

    loadedImages.clear();
    originalImages.clear();
    segmentationSlices16.clear();
    currentSegmentationVolume16.clear();
    hasSegmentation16Data = false;
    loadedImages.reserve(filePaths.size());
    originalImages.reserve(filePaths.size());
    segmentationSlices16.reserve(filePaths.size());

    // Reset Material Colors state when new dataset is loaded
    if (materialColorsCheckBox) {
        materialColorsCheckBox->blockSignals(true);
        materialColorsCheckBox->setChecked(false);
        materialColorsCheckBox->setEnabled(false);
        materialColorsCheckBox->blockSignals(false);
    }
    if (materialLegendWidget)
        materialLegendWidget->setVisible(false);
    materialColorsEnabled = false;
    glWidget->setMaterialColorsEnabled(false);

    if (mainTabs && mainTabs->count() > 1) {
        mainTabs->setTabEnabled(1, false);
    }
    setEnabled(false);

    QVector<int> sliceIndices(filePaths.size());
    std::iota(sliceIndices.begin(), sliceIndices.end(), 0);
    const double threshold = currentThreshold;
    const int totalSlices = filePaths.size();

    // Create a progress dialog
    auto* progressDialog = new QProgressDialog("Loading images...", "Cancel", 0, filePaths.size(), this);
    progressDialog->setWindowTitle("Loading Progress");
    progressDialog->setWindowModality(Qt::WindowModal); // Block UI interaction until preview loading completes
    progressDialog->setMinimumDuration(0); // Show immediately
    progressDialog->setAutoClose(false);
    progressDialog->setAutoReset(false);
    progressDialog->show();
    centerDialogOnWidget(progressDialog, this);
    activeImportDialog = progressDialog;

    auto* loadPool = new QThreadPool(this);
    const int idealThreads = QThread::idealThreadCount();
    // Import is I/O + decode heavy; allowing more workers than the previous 1-2 cap
    // significantly reduces load time on large stacks while avoiding runaway threading.
    const int workerThreads = qBound(4, qMax(4, idealThreads), 12);
    loadPool->setMaxThreadCount(workerThreads);

    auto* watcher = new QFutureWatcher<LoadedSliceResult>(this);
    connect(watcher, &QFutureWatcher<LoadedSliceResult>::progressRangeChanged,
            progressDialog, &QProgressDialog::setRange);
    connect(watcher, &QFutureWatcher<LoadedSliceResult>::progressValueChanged,
            progressDialog, &QProgressDialog::setValue);
    connect(watcher, &QFutureWatcher<LoadedSliceResult>::progressValueChanged,
            this, [progressDialog, totalSlices](int value) {
                progressDialog->setLabelText(QString("Loading image %1/%2").arg(value).arg(totalSlices));
            });
    connect(progressDialog, &QProgressDialog::canceled, watcher, &QFutureWatcher<LoadedSliceResult>::cancel);

            connect(watcher, &QFutureWatcher<LoadedSliceResult>::finished, this,
                [this, watcher, progressDialog, loadPool]() {

        if (progressDialog) {
            progressDialog->setLabelText("Finalizing loaded slices...");
            // Switch to busy indicator because this stage does not expose mapped progress.
            progressDialog->setRange(0, 0);
            progressDialog->setValue(0);
            QApplication::processEvents();
        }

        const QFuture<LoadedSliceResult> future = watcher->future();
        auto* finalizeWatcher = new QFutureWatcher<ImportFinalizeResult>(this);
        connect(finalizeWatcher, &QFutureWatcher<ImportFinalizeResult>::finished, this,
            [this, finalizeWatcher, watcher, progressDialog, loadPool]() {
            ImportFinalizeResult result = finalizeWatcher->result();

            originalImages = std::move(result.originalImages);
            loadedImages = std::move(result.loadedImages);
            segmentationSlices16 = std::move(result.segmentationSlices16);
            hasSegmentation16Data = result.hasSegmentation16Data;

            if (!loadedImages.isEmpty()) {
                if (loadedImages.size() >= 4) {
                    const int step = qMax(1, loadedImages.size() / 120);
                    double diffSum = 0.0;
                    int diffCount = 0;
                    for (int i = 0; i + step < loadedImages.size(); i += step) {
                        diffSum += meanAbsDiffSampled(loadedImages[i], loadedImages[i + step]);
                        ++diffCount;
                    }

                    const double avgSliceDiff = diffCount > 0 ? (diffSum / diffCount) : 0.0;
                    qDebug() << "Slice-stack consistency metric (mean abs diff):" << avgSliceDiff;
                    if (avgSliceDiff < 2.0) {
                        qWarning() << "Slices are extremely similar across the stack."
                                   << "Please verify this dataset is a true tomographic volume.";
                    }
                }

                const QString msg = (result.failedCount > 0)
                    ? QString("Loaded %1 image(s). %2 file(s) could not be decoded. Seg16: %3/%4 native. Threshold remains %5.")
                        .arg(loadedImages.size()).arg(result.failedCount).arg(result.native16Count).arg(segmentationSlices16.size()).arg(currentThreshold16)
                    : QString("Images loaded successfully (%1 slices). Seg16 native: %2/%3. Threshold remains %4.")
                        .arg(loadedImages.size()).arg(result.native16Count).arg(segmentationSlices16.size()).arg(currentThreshold16);
                qDebug().noquote() << "-------";
                qDebug().noquote() << QString("Import dataset: %1 s").arg(double(importTimer.elapsed()) / 1000.0, 0, 'f', 2);
                qDebug().noquote() << msg;
                qDebug().noquote() << "-------";
                updateImagePreviews();
            } else {
                qDebug().noquote() << "-------";
                qDebug().noquote() << QString("Import dataset: %1 s").arg(double(importTimer.elapsed()) / 1000.0, 0, 'f', 2);
                qDebug().noquote() << QString("No valid images loaded (%1 file(s) failed to decode). "
                            "Compressed DICOM (JPEG/JPEG2000) is not supported.").arg(result.failedCount);
                qDebug().noquote() << "-------";
                updateImagePreviews();
            }
        });

        finalizeWatcher->setFuture(QtConcurrent::run([future]() -> ImportFinalizeResult {
            ImportFinalizeResult result;
            const int resultCount = future.resultCount();
            result.originalImages.reserve(resultCount);
            result.loadedImages.reserve(resultCount);
            result.segmentationSlices16.reserve(resultCount);

            for (int i = 0; i < resultCount; ++i) {
                const LoadedSliceResult slice = future.resultAt(i);
                if (!slice.valid) {
                    ++result.failedCount;
                    continue;
                }

                result.originalImages.append(slice.original);
                result.loadedImages.append(slice.processed);
                if (slice.hasNative16) {
                    ++result.native16Count;
                }
                result.segmentationSlices16.append(slice.segmentation16);
            }

            if (!result.segmentationSlices16.isEmpty()) {
                cv::Rect sharedCropRect;
                bool haveSharedCrop = false;

                const int totalSegSlices = result.segmentationSlices16.size();
                const int sampleCount = qMin(totalSegSlices, 96);
                QVector<int> sampleIndices;
                sampleIndices.reserve(sampleCount);
                if (sampleCount == totalSegSlices) {
                    for (int i = 0; i < totalSegSlices; ++i) {
                        sampleIndices.append(i);
                    }
                } else {
                    const double step = double(totalSegSlices - 1) / double(sampleCount - 1);
                    for (int i = 0; i < sampleCount; ++i) {
                        sampleIndices.append(int(std::llround(step * i)));
                    }
                }

                // Crop detection is expensive on large datasets; sample representative slices.
                for (int sampleIndex : sampleIndices) {
                    const cv::Rect detectedRect = detectSegmentationCropRect(result.segmentationSlices16[sampleIndex]);
                    if (detectedRect.width <= 0 || detectedRect.height <= 0) {
                        continue;
                    }
                    if (!haveSharedCrop) {
                        sharedCropRect = detectedRect;
                        haveSharedCrop = true;
                    } else {
                        sharedCropRect |= detectedRect;
                    }
                }

                if (haveSharedCrop && sharedCropRect.width > 0 && sharedCropRect.height > 0) {
                    const cv::Rect imageBounds(0, 0, result.segmentationSlices16.first().cols, result.segmentationSlices16.first().rows);
                    sharedCropRect &= imageBounds;

                    if (sharedCropRect.width > 0 && sharedCropRect.height > 0) {
                        const double fullArea = double(imageBounds.width) * double(imageBounds.height);
                        const double cropArea = double(sharedCropRect.width) * double(sharedCropRect.height);
                        const bool nearFullFrame = (fullArea > 0.0) && ((cropArea / fullArea) >= 0.94);

                        if (!nearFullFrame) {
                            QVector<cv::Mat> croppedSlices;
                            croppedSlices.reserve(result.segmentationSlices16.size());
                            for (const cv::Mat& slice16 : result.segmentationSlices16) {
                                croppedSlices.append(cropSegmentationSlice16(slice16, sharedCropRect));
                            }
                            result.segmentationSlices16 = croppedSlices;
                            qDebug() << "Applied shared 16-bit crop:" << QRect(sharedCropRect.x, sharedCropRect.y, sharedCropRect.width, sharedCropRect.height)
                                     << "for" << result.segmentationSlices16.size() << "slice(s)"
                                     << "(detected from" << sampleCount << "sampled slice(s))";
                        } else {
                            qDebug() << "Skipped shared crop: detected ROI is near full-frame"
                                     << QRect(sharedCropRect.x, sharedCropRect.y, sharedCropRect.width, sharedCropRect.height)
                                     << "(detected from" << sampleCount << "sampled slice(s))";
                        }
                    }
                }
            }

            result.hasSegmentation16Data = (!result.segmentationSlices16.isEmpty() && result.segmentationSlices16.size() == result.loadedImages.size());
            return result;
        }));
        watcher->deleteLater();
    });

    watcher->setFuture(QtConcurrent::mapped(loadPool, sliceIndices, [this, filePaths, threshold, totalSlices](int index) -> LoadedSliceResult {
        LoadedSliceResult result;
        result.index = index;

        try {
            if (index < 0 || index >= filePaths.size()) {
                qWarning() << "Slice index out of range during async load:" << index << "size:" << filePaths.size();
                return result;
            }

            // Fast path: decode native 16-bit slice first and reuse it for both segmentation and preview.
            cv::Mat decoded16;
            bool native16 = false;
            const bool haveSeg16 = decodeSegmentationSlice16(filePaths[index], &decoded16, &native16);

            QImage originalImg;
            bool usedFallback = false;

            if (haveSeg16 && !decoded16.empty()) {
                originalImg = cvMatToDisplayableQImage(decoded16);
                result.segmentation16 = decoded16;
                result.hasNative16 = native16;
            }

            if (originalImg.isNull()) {
                originalImg = loadSliceImage(filePaths[index], &usedFallback);
                if (originalImg.isNull()) {
                    qWarning() << "Failed to decode image (Qt/OpenCV):" << filePaths[index];
                    return result;
                }
            }

            if (usedFallback) {
                static std::atomic<int> fallbackLogCount{0};
                if (fallbackLogCount.fetch_add(1, std::memory_order_relaxed) < 5) {
                    qDebug() << "Loaded slice via fallback decoder:" << QFileInfo(filePaths[index]).fileName();
                }
            }

            result.original = originalImg;

            // If segmentation was already decoded for this slice, preprocessing is redundant
            // for reconstruction and only increases import time.
            if (haveSeg16 && !result.segmentation16.empty()) {
                result.processed = originalImg;
            } else {
                result.processed = preprocessLoadedImage(originalImg, filePaths[index], threshold, totalSlices);
                if (result.segmentation16.empty()) {
                    result.segmentation16 = qImageToSegmentationU16(originalImg);
                    result.hasNative16 = false;
                }
            }
            result.valid = !result.processed.isNull();
        } catch (const cv::Exception& e) {
            qWarning() << "OpenCV error while loading slice" << filePaths.value(index) << ":" << e.what();
        } catch (const std::exception& e) {
            qWarning() << "Standard exception while loading slice" << filePaths.value(index) << ":" << e.what();
        } catch (...) {
            qWarning() << "Unknown error while loading slice" << filePaths.value(index);
        }

        return result;
    }));

}

void MainWindow::updateImagePreviews() {
    // Clear previous previews
    QLayoutItem* item;
    while ((item = previewLayout->takeAt(0)) != nullptr) {
        if (item->widget()) {
            item->widget()->removeEventFilter(this);
            delete item->widget();
        }
        delete item;
    }

    const QVector<QImage>& previewImages =
        (!originalImages.isEmpty() && originalImages.size() == loadedImages.size())
            ? originalImages
            : loadedImages;

    if (previewImages.isEmpty()) {
        QLabel* emptyLabel = new QLabel(
            "No valid images loaded.\n\n"
            "Supported: uncompressed 8-bit or 16-bit monochrome DICOM\n"
            "(BMP, PNG, JPG, TIF are also supported).\n\n"
            "Note: JPEG or JPEG 2000 compressed DICOM files cannot be decoded.");
        emptyLabel->setAlignment(Qt::AlignCenter);
        emptyLabel->setWordWrap(true);
        emptyLabel->setStyleSheet("color: #a0a0a0; font-size: 13px; padding: 30px;");
        previewLayout->addWidget(emptyLabel, 0, 0);
        previewBuildInProgress = false;
        previewBuildImages.clear();
        previewBuildIndex = 0;
        finalizeImportUi();
        return;
    }

    previewBuildImages = previewImages;
    previewBuildIndex = 0;
    previewBuildInProgress = true;
    previewBuildTimer.start(1);
}

void MainWindow::appendPreviewThumbnails() {
    if (!previewBuildInProgress) {
        return;
    }

    const int thumbsPerRow = 4;
    const int thumbSize = 200;
    const int batchSize = (previewBuildImages.size() >= 800) ? 28 : 14;

    int added = 0;
    while (previewBuildIndex < previewBuildImages.size() && added < batchSize) {
        const int imageIndex = previewBuildIndex;
        QLabel* thumbLabel = new QLabel;
        QImage thumb = previewBuildImages[imageIndex].scaled(thumbSize, thumbSize,
                                                             Qt::KeepAspectRatio,
                                                             Qt::FastTransformation);

        thumbLabel->setPixmap(QPixmap::fromImage(thumb));
        thumbLabel->setToolTip(QString("Double-click to view full size\nSlice %1").arg(imageIndex + 1));
        thumbLabel->setAlignment(Qt::AlignCenter);
        thumbLabel->setStyleSheet("border: 1px solid #505050; margin: 2px;");
        thumbLabel->installEventFilter(this);
        thumbLabel->setProperty("imageIndex", imageIndex);
        previewLayout->addWidget(thumbLabel, imageIndex / thumbsPerRow, imageIndex % thumbsPerRow);

        ++previewBuildIndex;
        ++added;
    }

    if (activeImportDialog && !previewBuildImages.isEmpty()) {
        // Show determinate progress for thumbnail generation so users see forward motion.
        activeImportDialog->setRange(0, previewBuildImages.size());
        activeImportDialog->setValue(previewBuildIndex);
        activeImportDialog->setLabelText(
            QString("Preparing previews... %1/%2")
                .arg(previewBuildIndex)
                .arg(previewBuildImages.size())
        );
    }

    if (previewBuildIndex >= previewBuildImages.size()) {
        previewBuildTimer.stop();
        previewBuildInProgress = false;
        previewBuildImages.clear();
        finalizeImportUi();
    }
}

void MainWindow::finalizeImportUi() {
    if (activeImportDialog) {
        activeImportDialog->close();
        activeImportDialog->deleteLater();
        activeImportDialog = nullptr;
    }

    if (mainTabs && mainTabs->count() > 1) {
        mainTabs->setTabEnabled(1, true);
        if (!loadedImages.isEmpty()) {
            mainTabs->setCurrentIndex(1);
        }
    }

    setEnabled(true);
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    // Reposition overlay legend when the 3D view tab is resized
    if (watched == vizTab && event->type() == QEvent::Resize && overlayLegendWidget) {
        overlayLegendWidget->adjustSize();
        const int margin = 10;
        overlayLegendWidget->move(vizTab->width() - overlayLegendWidget->width() - margin, margin);
        overlayLegendWidget->raise();
    }

    if (event->type() == QEvent::MouseButtonDblClick) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            QLabel* label = qobject_cast<QLabel*>(watched);
            if (label) {
                bool ok = false;
                int index = label->property("imageIndex").toInt(&ok);
                const QVector<QImage>& previewImages =
                    (!originalImages.isEmpty() && originalImages.size() == loadedImages.size())
                        ? originalImages
                        : loadedImages;
                if (ok && index >= 0 && index < previewImages.size()) {
                    showFullSizeImage(previewImages[index]);
                    return true;
                }
            }
        }
    }
    return QMainWindow::eventFilter(watched, event);
}
void MainWindow::showFullSizeImage(const QImage& image) {
    QDialog previewDialog(this);
    previewDialog.setWindowTitle("Full Size Preview");
    previewDialog.setStyleSheet("background: #353535;");

    QLabel* imageLabel = new QLabel(&previewDialog);
    imageLabel->setPixmap(QPixmap::fromImage(image));
    imageLabel->setAlignment(Qt::AlignCenter);

    QScrollArea* scrollArea = new QScrollArea(&previewDialog);
    scrollArea->setWidget(imageLabel);
    scrollArea->setWidgetResizable(true);

    QVBoxLayout* layout = new QVBoxLayout(&previewDialog);
    layout->addWidget(scrollArea);

    // Add close button
    QPushButton* closeButton = new QPushButton("Close", &previewDialog);
    connect(closeButton, &QPushButton::clicked, &previewDialog, &QDialog::accept);
    layout->addWidget(closeButton);

    previewDialog.resize(800, 600);
    previewDialog.exec();
}

void MainWindow::createCentralWidget()
{
    // Already handled in constructor, can be empty
}

void MainWindow::toggleControls(bool visible)
{
    controlsDock->setVisible(visible);
}

void MainWindow::updateThreshold(double value)
{
    // Implement threshold update logic
    qDebug() << "Threshold:" << value;
}

QImage MainWindow::preprocessLoadedImage(const QImage& input, const QString& path, double thresholdOverride, int totalSlicesOverride) {
    if (input.isNull()) {
        qWarning() << "Invalid image for preprocessing:" << path;
        return QImage();
    }

    // Adaptive ROI detection: Use hardcoded crop as fallback, but detect content region for TIFF/variable-sized inputs.
    QRect cropRect(45, 48, 1417, 537);
    
    // Convert input to CV for content detection
    cv::Mat inputMat = QImageToCvMat(input);
    if (!inputMat.empty()) {
        // Apply Otsu to find content threshold dynamically
        cv::Mat grayInput = inputMat.channels() == 3 ? cv::Mat() : inputMat.clone();
        if (inputMat.channels() == 3) {
            cv::cvtColor(inputMat, grayInput, cv::COLOR_BGR2GRAY);
        }
        
        cv::Mat binaryContent;
        cv::threshold(grayInput, binaryContent, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        
        // Find contours indicating actual scan content
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryContent.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty()) {
            // Get bounding box of all significant contours.
            int minX = input.width(), maxX = 0;
            int minY = input.height(), maxY = 0;
            
            for (const auto& contour : contours) {
                cv::Rect boundingBox = cv::boundingRect(contour);
                const bool touchesBorder = (boundingBox.x <= 1)
                    || (boundingBox.y <= 1)
                    || (boundingBox.x + boundingBox.width >= input.width() - 1)
                    || (boundingBox.y + boundingBox.height >= input.height() - 1);
                if (touchesBorder) {
                    continue;
                }
                if (boundingBox.area() > (input.width() * input.height() / 500)) { // Filter noise
                    minX = qMin(minX, boundingBox.x);
                    maxX = qMax(maxX, boundingBox.x + boundingBox.width);
                    minY = qMin(minY, boundingBox.y);
                    maxY = qMax(maxY, boundingBox.y + boundingBox.height);
                }
            }
            
            // Use detected ROI if it's valid and larger than the hardcoded one
            if (minX < maxX && minY < maxY) {
                int detectedWidth = maxX - minX;
                int detectedHeight = maxY - minY;
                
                // Only switch to detected ROI if content region is reasonably large
                if (detectedWidth > 400 && detectedHeight > 150) {
                    // Add small margin and apply padding
                    int margin = qMax(5, qMin(detectedWidth, detectedHeight) / 50);
                    cropRect = QRect(
                        qMax(0, minX - margin),
                        qMax(0, minY - margin),
                        qMin(input.width(), detectedWidth + 2 * margin),
                        qMin(input.height(), detectedHeight + 2 * margin)
                    );
                }
            }
        }
    }
    
    cropRect = cropRect.intersected(input.rect());
    if (cropRect.isEmpty()) {
        qWarning() << "Crop ROI is outside image bounds for" << path;
        return QImage();
    }

    QImage croppedImage = input.copy(cropRect);
    cv::Mat gray = QImageToCvMat(croppedImage);

    // Small denoise and conservative thresholding preserve jaw structure better than a low global threshold.
    cv::Mat smoothed;
    cv::GaussianBlur(gray, smoothed, cv::Size(3, 3), 0.0, 0.0);

    cv::Mat binaryMask;
    const double otsuThreshold = cv::threshold(smoothed, binaryMask, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const double effectiveThreshold = (thresholdOverride >= 0.0) ? thresholdOverride : currentThreshold;
    const int totalSlices = (totalSlicesOverride > 0) ? totalSlicesOverride : qMax(1, currentImagePaths.size());
    const double manualThreshold = qMax(effectiveThreshold, 0.40) * 255.0;
    const double finalThreshold = qMax(manualThreshold, otsuThreshold * 0.92);
    const bool sparseStack = (totalSlices <= 48);
    const bool denseStack = (totalSlices >= 200);
    const QString stackProfile = sparseStack ? "sparse" : (denseStack ? "dense" : "balanced");
    const auto extractJawMask = [&](double thresholdValue) -> cv::Mat {
        cv::Mat mask;
        cv::threshold(smoothed, mask, thresholdValue, 255, cv::THRESH_BINARY);

        const cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        // Sparse stacks lose fine anatomy quickly; skip initial opening there.
        if (!sparseStack) {
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelSmall);
        }
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelSmall);

        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int componentCount = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
        if (componentCount <= 1) {
            return mask;
        }

        struct ComponentCandidate {
            int label = 0;
            int area = 0;
            double score = 0.0;
        };

        const int rows = mask.rows;
        const int cols = mask.cols;
        const int totalPixels = rows * cols;
        // Adapt minimum component area to dataset depth:
        // sparse stacks keep finer details, dense stacks reject tiny noise.
        const int minComponentArea = sparseStack
            ? qMax(8, totalPixels / 25000)
            : (denseStack ? qMax(180, totalPixels / 2200) : qMax(64, totalPixels / 4000));

        QVector<ComponentCandidate> candidates;
        candidates.reserve(componentCount - 1);

        for (int label = 1; label < componentCount; ++label) {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < minComponentArea) {
                continue;
            }

            const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
            const int top = stats.at<int>(label, cv::CC_STAT_TOP);
            const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
            const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
            const int centerX = left + width / 2;
            const int centerY = top + height / 2;

            const bool touchesLeft = (left <= 1);
            const bool touchesRight = (left + width >= cols - 1);
            const bool touchesTop = (top <= 1);
            const bool touchesBottom = (top + height >= rows - 1);
            const int borderTouchCount = int(touchesLeft) + int(touchesRight) + int(touchesTop) + int(touchesBottom);
            const bool thinHorizontalStrip = (height <= qMax(3, rows / 26)) && (width >= (cols * 3 / 5));

            // Reject frame/slab artifacts that dominate image borders.
            // For sparse stacks, be more lenient with border touching (jaw can legitimately extend to edges).
            const int borderRejectThreshold = sparseStack ? 4 : 3;
            const int borderAreaThreshold = sparseStack ? (totalPixels / 8) : (totalPixels / 12);
            const bool frameLike = (borderTouchCount >= 2)
                && ((width >= (cols * 3 / 5)) || (height >= (rows * 3 / 5)) || (area > borderAreaThreshold));
            if (frameLike || (borderTouchCount >= borderRejectThreshold && area > borderAreaThreshold)) {
                continue;
            }
            // Dense stacks often contain bright horizontal streak artifacts.
            // They are thin and span much of the width, unlike the jaw body.
            if (denseStack && thinHorizontalStrip && area < (totalPixels / 24)) {
                continue;
            }
            // Relax tall-component rejection for sparse stacks where jaw extends nearly full height
            const int heightRejectRatio = sparseStack ? 9 : 20;  // sparse: 9/10, dense: 17/20
            const int heightAreaThreshold = sparseStack ? (totalPixels / 10) : (totalPixels / 14);
            if (height > (rows * heightRejectRatio / 10) && area > heightAreaThreshold) {
                continue;
            }
            const int topRejectBand = sparseStack ? (rows / 10) : (rows / 8);  // Wider acceptance zone for sparse
            const int topAreaThreshold = sparseStack ? (totalPixels / 150) : (totalPixels / 220);
            if (centerY < topRejectBand && area > topAreaThreshold) {
                continue;
            }
            if (denseStack && centerY > (rows * 9 / 10) && area < (totalPixels / 100)) {
                continue;
            }

            const double areaScore = double(area) / double(totalPixels);
            const double widthScore = double(width) / double(cols);
            const double aspect = double(width) / double(qMax(1, height));
            const double aspectScore = qMin(aspect / 8.0, 1.0);
            const double centerScore = 1.0 - qAbs(double(centerX) - (double(cols) * 0.5)) / (double(cols) * 0.5 + 1e-6);
            const double score = (0.55 * areaScore) + (0.20 * widthScore) + (0.20 * aspectScore) + (0.05 * centerScore);
            candidates.append({label, area, score});
        }

        cv::Mat filteredMask = cv::Mat::zeros(mask.size(), CV_8UC1);
        int keptComponents = 0;
        if (!candidates.isEmpty()) {
            std::sort(candidates.begin(), candidates.end(), [](const ComponentCandidate& a, const ComponentCandidate& b) {
                return a.score > b.score;
            });

            // Sparse stacks need more fragments retained, dense stacks need stronger cleanup.
            const int targetKeepCount = sparseStack ? 7 : (denseStack ? 2 : 4);
            const int keepCount = qMin(targetKeepCount, candidates.size());
            for (int i = 0; i < keepCount; ++i) {
                filteredMask.setTo(255, labels == candidates[i].label);
            }
            keptComponents = keepCount;
        } else {
            // Fallback: keep the largest component if no candidate survived strict filtering.
            int maxLabel = 1;
            int maxArea = stats.at<int>(1, cv::CC_STAT_AREA);
            for (int label = 2; label < componentCount; ++label) {
                const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                if (area > maxArea) {
                    maxArea = area;
                    maxLabel = label;
                }
            }
            filteredMask.setTo(255, labels == maxLabel);
            keptComponents = 1;
        }

        const cv::Size closeSize = sparseStack ? cv::Size(11, 5) : (denseStack ? cv::Size(17, 9) : cv::Size(13, 7));
        const cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, closeSize);
        cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_CLOSE, kernelClose);
        cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_OPEN, kernelSmall);

        if (denseStack) {
            std::vector<int> rowOccupancy(rows, 0);
            for (int y = 0; y < rows; ++y) {
                rowOccupancy[y] = cv::countNonZero(filteredMask.row(y));
            }

            for (int y = 1; y < rows - 1; ++y) {
                const int currentCount = rowOccupancy[y];
                const int prevCount = rowOccupancy[y - 1];
                const int nextCount = rowOccupancy[y + 1];
                const bool wideRow = (currentCount >= (cols * 9 / 20));
                const bool isolatedSpike = (currentCount > (prevCount * 2)) && (currentCount > (nextCount * 2));
                if (wideRow && isolatedSpike) {
                    filteredMask.row(y).setTo(0);
                }
            }
        }

        return filteredMask;
    };

    binaryMask = extractJawMask(finalThreshold);

    const double fillRatio = double(cv::countNonZero(binaryMask)) / double(binaryMask.rows * binaryMask.cols + 1e-6);
    if (fillRatio > 0.22) {
        // Very dense masks usually indicate slab artifacts; retry with stricter threshold.
        binaryMask = extractJawMask(qMin(255.0, finalThreshold + 18.0));
    } else if ((sparseStack && fillRatio < 0.018) || (!sparseStack && fillRatio < 0.008)) {
        // Very sparse masks usually mean valid anatomy was over-filtered; retry with slightly relaxed threshold.
        const double sparseRelax = sparseStack ? 18.0 : 12.0;
        binaryMask = extractJawMask(qMax(0.0, finalThreshold - sparseRelax));

        const double relaxedFillRatio = double(cv::countNonZero(binaryMask)) / double(binaryMask.rows * binaryMask.cols + 1e-6);
        if ((sparseStack && relaxedFillRatio < 0.010) || (!sparseStack && relaxedFillRatio < 0.006)) {
            const double otsuScale = sparseStack ? 0.80 : 0.85;
            binaryMask = extractJawMask(qMax(0.0, otsuThreshold * otsuScale));
        }
    }

    return cvMatToQImage(binaryMask);
}

QImage MainWindow::loadMemoryMappedImage(const QString& path) {
    const QImage input = loadSliceImage(path);
    if (input.isNull()) {
        qWarning() << "Failed to decode image (Qt/OpenCV):" << path;
        return QImage();
    }

    return preprocessLoadedImage(input, path, currentThreshold, qMax(1, currentImagePaths.size()));
}

MainWindow::VolumeData MainWindow::convertToVolume(const QVector<QImage>& images, const ProgressCallback& progressCallback) {
    VolumeData volume;

    if (images.isEmpty()) {
        qWarning() << "No images to convert";
        return volume;
    }

    // Get source dimensions from first image
    const int srcWidth = images.first().width();
    const int srcHeight = images.first().height();
    const int srcDepth = images.size();
    qint64 filledVoxels = 0;

    qDebug() << "Converting images to volume. Dimensions:"
             << srcWidth << "x" << srcHeight << "x" << srcDepth;

    // Validate dimensions
    for (const QImage& img : images) {
        if (img.width() != srcWidth || img.height() != srcHeight) {
            qCritical() << "Image dimensions mismatch! Expected"
                        << srcWidth << "x" << srcHeight
                        << "got" << img.width() << "x" << img.height();
            return VolumeData();
        }
    }

    // Active production path: keep binary thresholded volume for stable presentation behavior.
    const int width = srcWidth;
    const int height = srcHeight;
    const int depth = srcDepth;

    // Allocate 3D array [z][y][x]
    volume.resize(depth);
    for (int z = 0; z < depth; ++z) {
        volume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            volume[z][y].resize(width);
        }
    }

    // Threshold application logic: convert the active threshold into the binary volume.
    const float thresholdValue = static_cast<float>(currentThreshold * 255.0);
    auto sampleBinaryPixel = [&](const QImage& img, int x, int y) -> float {
        return (qGray(img.pixel(x, y)) >= thresholdValue) ? 1.0f : 0.0f;
    };

    auto writeVolumeSlice = [&](int outZ, const QImage& imgA, const QImage* imgB, float lerpT) {
        for (int y = 0; y < height; ++y) {
            float* row = volume[outZ][y].data();
            for (int x = 0; x < width; ++x) {
                const float valueA = sampleBinaryPixel(imgA, x, y);
                float value = valueA;
                if (imgB) {
                    const float valueB = sampleBinaryPixel(*imgB, x, y);
                    value = ((1.0f - lerpT) * valueA) + (lerpT * valueB);
                }
                row[x] = value;
                if (value >= 0.5f) {
                    ++filledVoxels;
                }
            }
        }
    };

    for (int z = 0; z < depth; ++z) {
        writeVolumeSlice(z, images[z], nullptr, 0.0f);

        if (progressCallback) {
            const int progress = 5 + ((z + 1) * 70 / depth);
            progressCallback(progress, QString("Building 3D volume... %1/%2 slices").arg(z + 1).arg(depth));
        }
    }

    const qint64 totalVoxels = static_cast<qint64>(width) * height * depth;
    const double occupancy = totalVoxels > 0 ? (100.0 * static_cast<double>(filledVoxels) / static_cast<double>(totalVoxels)) : 0.0;
    qDebug() << "Binary volume occupancy:" << occupancy << "%";
    if (occupancy < 0.1 || occupancy > 99.9) {
        qWarning() << "Volume is nearly uniform; mesh can be empty. Try adjusting threshold.";
    }

    // // Save the 3D volume to a binary file (for debugging)
    // QFile volumeFile("qt_volume.bin");
    // if (volumeFile.open(QIODevice::WriteOnly)) {
    //     for (int z = 0; z < volume.size(); ++z) {
    //         for (int y = 0; y < volume[z].size(); ++y) {
    //             for (int x = 0; x < volume[z][y].size(); ++x) {
    //                 // Save binary values (0 or 1) as floats
    //                 float value = volume[z][y][x];
    //                 volumeFile.write(reinterpret_cast<const char*>(&value), sizeof(float));
    //             }
    //         }
    //     }
    //     volumeFile.close();
    // }

    // Active production path intentionally disables XY/Z smoothing and inter-slice blending
    // so the mesh input remains a strict thresholded binary field.

    if (progressCallback) {
        progressCallback(90, "Volume ready. Preparing GPU execution...");
    }

    qDebug() << "Volume created successfully.";
    return volume;
}

quint16 MainWindow::computeOtsuThreshold16(const VolumeData16& volume) {
    if (volume.isEmpty()) {
        return 32768;
    }

    const int zSize = volume.size();
    const int ySize = volume[0].size();
    const int xSize = volume[0][0].size();
    const qint64 totalPixels = qint64(xSize) * ySize * zSize;

    std::vector<qint64> histogram(65536, 0);

    for (int z = 0; z < zSize; ++z) {
        for (int y = 0; y < ySize; ++y) {
            const quint16* row = volume[z][y].constData();
            for (int x = 0; x < xSize; ++x) {
                histogram[row[x]]++;
            }
        }
    }

    double sumTotal = 0.0;
    for (int i = 0; i < 65536; ++i) {
        sumTotal += static_cast<double>(i) * static_cast<double>(histogram[i]);
    }

    qint64 sumBackground = 0;
    qint64 weightBackground = 0;
    double maxVariance = 0.0;
    quint16 optimalThreshold = 32768;

    for (int t = 0; t < 65536; ++t) {
        weightBackground += histogram[t];
        if (weightBackground == 0) continue;

        const qint64 weightForeground = totalPixels - weightBackground;
        if (weightForeground == 0) break;

        sumBackground += static_cast<qint64>(t) * histogram[t];

        const double muBackground = static_cast<double>(sumBackground) / static_cast<double>(weightBackground);
        const double muForeground = (sumTotal - static_cast<double>(sumBackground)) / static_cast<double>(weightForeground);

        const double variance = static_cast<double>(weightBackground) * static_cast<double>(weightForeground)
                                * (muBackground - muForeground) * (muBackground - muForeground);

        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = static_cast<quint16>(t);
        }
    }

    qDebug() << "Volume-level Otsu16 threshold:" << optimalThreshold << "variance:" << maxVariance;
    return optimalThreshold;
}

MainWindow::VolumeData MainWindow::convertToVolume16(const QVector<cv::Mat>& slices16, int threshold16, float smoothingIntensity, const ProgressCallback& progressCallback) {
    Q_UNUSED(smoothingIntensity);
    VolumeData volume;

    if (slices16.isEmpty()) {
        qWarning() << "No 16-bit slices to convert";
        return volume;
    }

    const int srcWidth = slices16.first().cols;
    const int srcHeight = slices16.first().rows;
    const int srcDepth = slices16.size();

    qDebug() << "Converting 16-bit slices to binary volume. Dimensions:"
             << srcWidth << "x" << srcHeight << "x" << srcDepth;

    for (int i = 0; i < slices16.size(); ++i) {
        if (slices16[i].cols != srcWidth || slices16[i].rows != srcHeight) {
            qCritical() << "16-bit slice dimension mismatch at index" << i;
            return VolumeData();
        }
    }

    const int width = srcWidth;
    const int height = srcHeight;
    const int depth = srcDepth;
    const bool sparseInput = (srcDepth <= 48);
    const bool denseInput = (srcDepth >= 200);

    volume.resize(depth);
    for (int z = 0; z < depth; ++z) {
        volume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            volume[z][y].resize(width);
        }
    }

    if (progressCallback) {
        progressCallback(8, QString("Running adaptive 16-bit segmentation (threshold=%1)...").arg(threshold16));
    }

    ImageProcessor::AdaptiveSettings adaptiveSettings;
    adaptiveSettings.globalWeight = denseInput ? 0.46 : (sparseInput ? 0.60 : 0.66);
    // Prefer adaptive per-slice thresholds; candidate scoring below decides if a polarity/mode is unstable.
    adaptiveSettings.useUniformThreshold = false;
    const int polarityHint = inferForegroundPolarityHint(slices16);
    adaptiveSettings.polarityMode = (polarityHint == 1 || polarityHint == 2) ? polarityHint : 0;
    adaptiveSettings.preserveGrayscale = true;
    adaptiveSettings.lowOccupancyThreshold = denseInput ? 0.05 : 0.60;
    adaptiveSettings.highOccupancyThreshold = denseInput ? 78.0 : 98.0;

    qDebug() << "Adaptive polarity hint (0=auto,1=dark,2=bright):" << polarityHint
             << "| applied mode:" << adaptiveSettings.polarityMode;

    auto computeMaskBorderOccupancyPct = [&](const QVector<cv::Mat>& masks) -> double {
        if (masks.isEmpty()) {
            return 0.0;
        }

        const int borderBandX = qMax(2, width / 16);
        const int borderBandY = qMax(2, height / 16);
        qint64 borderCount = 0;
        qint64 borderPixels = 0;

        for (int z = 0; z < masks.size(); ++z) {
            const cv::Mat& mask = masks[z];
            if (mask.empty()) {
                continue;
            }
            for (int y = 0; y < height; ++y) {
                const uchar* row = mask.ptr<uchar>(y);
                for (int x = 0; x < width; ++x) {
                    const bool inBorder = (x < borderBandX) || (x >= width - borderBandX)
                        || (y < borderBandY) || (y >= height - borderBandY);
                    if (!inBorder) {
                        continue;
                    }
                    ++borderPixels;
                    if (row[x] > 0) {
                        ++borderCount;
                    }
                }
            }
        }

        return (borderPixels > 0)
            ? (100.0 * double(borderCount) / double(borderPixels))
            : 0.0;
    };

    auto computeAdjacentIoU = [&](const QVector<cv::Mat>& masks) -> double {
        if (masks.size() < 2) {
            return 0.0;
        }

        const int sampleStep = qMax(1, masks.size() / 140);
        double iouSum = 0.0;
        int iouCount = 0;
        for (int z = 0; z + sampleStep < masks.size(); z += sampleStep) {
            const cv::Mat& a = masks[z];
            const cv::Mat& b = masks[z + sampleStep];
            if (a.empty() || b.empty() || a.size() != b.size()) {
                continue;
            }

            cv::Mat interMask;
            cv::Mat unionMask;
            cv::bitwise_and(a, b, interMask);
            cv::bitwise_or(a, b, unionMask);
            const double interCount = double(cv::countNonZero(interMask));
            const double unionCount = double(cv::countNonZero(unionMask));
            if (unionCount <= 0.0) {
                continue;
            }

            iouSum += (interCount / unionCount);
            ++iouCount;
        }

        return (iouCount > 0) ? (iouSum / double(iouCount)) : 0.0;
    };

    ImageProcessor::ProcessingMetrics metrics;
    const int boundedThreshold = qBound(0, threshold16, 65535);
    QVector<cv::Mat> adaptiveMasks;
    double adaptiveBorderOcc = 0.0;

    auto adaptiveProgress = [&](int pct, const QString& stageMessage) {
        if (!progressCallback) {
            return;
        }
        // Map adaptive stage [0..100] into overall mesh-prep [8..18] so users see progress.
        const int mapped = 8 + ((qBound(0, pct, 100) * 10) / 100);
        progressCallback(mapped, stageMessage);
    };

    // Run adaptive pipeline once, then only retry with opposite polarity when metrics indicate instability.
    QVector<cv::Mat> adaptiveMasksOutput = imageProcessor.processSlices16Adaptive(
        slices16,
        boundedThreshold,
        adaptiveSettings,
        &metrics,
        adaptiveProgress
    );

    if (adaptiveMasksOutput.size() != depth) {
        qCritical() << "Adaptive 16-bit segmentation output size mismatch:" << adaptiveMasksOutput.size() << "expected" << depth;
        return VolumeData();
    }

    adaptiveMasks = adaptiveMasksOutput;
    adaptiveBorderOcc = computeMaskBorderOccupancyPct(adaptiveMasks);

    const double baseAdjacentIoU = computeAdjacentIoU(adaptiveMasks);
    auto scoreSegmentation = [&](const ImageProcessor::ProcessingMetrics& m, double borderOcc, double adjacentIoU) -> double {
        const double occPenalty = denseInput
            ? ((m.occupancy < 4.0)
                ? ((4.0 - m.occupancy) * 3.0)
                : ((m.occupancy > 75.0)
                    ? ((m.occupancy - 75.0) * 1.6)
                    : (qAbs(m.occupancy - 28.0) * 0.25)))
            : qAbs(m.occupancy - 18.0);
        const double borderPenalty = borderOcc * (denseInput ? 1.0 : 1.2);
        const double continuityPenalty = qMax(0.0, (denseInput ? 0.12 : 0.16) - adjacentIoU) * (denseInput ? 120.0 : 120.0);
        const double sliceBalancePenalty = (double(m.slicesTooSparse) + double(m.slicesTooDense)) * 0.03;
        return occPenalty + borderPenalty + continuityPenalty + sliceBalancePenalty;
    };

    // Keep runtime bounded: only do one extra polarity pass when metrics strongly indicate a bad segmentation.
    const bool suspiciousSegmentation = (metrics.occupancy < (denseInput ? 0.05 : 0.35))
        || (metrics.occupancy > (denseInput ? 88.0 : 48.0))
        || (adaptiveBorderOcc > (denseInput ? 82.0 : 45.0))
        || (baseAdjacentIoU < (denseInput ? 0.035 : 0.08))
        || (denseInput && metrics.slicesTooSparse > (depth / 3))
        || (denseInput && metrics.slicesTooDense > ((depth * 3) / 4));
    if (suspiciousSegmentation) {
        if (progressCallback) {
            progressCallback(19, "Validating polarity candidate...");
        }
        const int oppositePolarity = (adaptiveSettings.polarityMode == 1) ? 2 : 1;
        ImageProcessor::AdaptiveSettings altSettings = adaptiveSettings;
        altSettings.polarityMode = oppositePolarity;

        ImageProcessor::ProcessingMetrics altMetrics;
        QVector<cv::Mat> altMasks = imageProcessor.processSlices16Adaptive(
            slices16,
            boundedThreshold,
            altSettings,
            &altMetrics
        );

        if (altMasks.size() == depth) {
            const double altBorderOcc = computeMaskBorderOccupancyPct(altMasks);
            const double altAdjacentIoU = computeAdjacentIoU(altMasks);
            const double baseScore = scoreSegmentation(metrics, adaptiveBorderOcc, baseAdjacentIoU);
            const double altScore = scoreSegmentation(altMetrics, altBorderOcc, altAdjacentIoU);

            qDebug() << "Adaptive fallback candidate"
                     << "| polarity" << oppositePolarity
                     << "| occ" << altMetrics.occupancy
                     << "| border" << altBorderOcc
                     << "| iou" << altAdjacentIoU
                     << "| score" << altScore
                     << "| baseScore" << baseScore;

            if (altScore + 1.0 < baseScore) {
                adaptiveMasks = std::move(altMasks);
                metrics = altMetrics;
                adaptiveBorderOcc = altBorderOcc;
                adaptiveSettings.polarityMode = oppositePolarity;
            }
        }
    }

    qDebug() << "Adaptive 16-bit segmentation:"
             << "| occupancy" << metrics.occupancy << "%"
             << "| border" << adaptiveBorderOcc << "%"
             << "| polarity" << adaptiveSettings.polarityMode;

    if (adaptiveMasks.size() != depth) {
        qCritical() << "Adaptive 16-bit segmentation output size mismatch:" << adaptiveMasks.size() << "expected" << depth;
        return VolumeData();
    }

    QVector<cv::Mat> masksForVolume = adaptiveMasks;

    // Dense clean stacks frequently fail from inter-slice discontinuities: thin gaps create
    // ribbon-like tearing in top/front views. Apply permissive Z-bridging before meshing.
    if (denseInput && depth >= 3) {
        if (progressCallback) {
            progressCallback(16, "Applying dense-stack Z consistency bridging...");
        }

        const int zPasses = 2;
        for (int pass = 0; pass < zPasses; ++pass) {
            QVector<cv::Mat> bridgedMasks;
            bridgedMasks.reserve(depth);
            for (int z = 0; z < depth; ++z) {
                bridgedMasks.append(masksForVolume[z].clone());
            }

            for (int z = 1; z < depth - 1; ++z) {
                const cv::Mat& prev = masksForVolume[z - 1];
                const cv::Mat& cur = masksForVolume[z];
                const cv::Mat& next = masksForVolume[z + 1];
                cv::Mat& dst = bridgedMasks[z];

                for (int y = 0; y < height; ++y) {
                    const uchar* pRow = prev.ptr<uchar>(y);
                    const uchar* cRow = cur.ptr<uchar>(y);
                    const uchar* nRow = next.ptr<uchar>(y);
                    uchar* dRow = dst.ptr<uchar>(y);

                    for (int x = 0; x < width; ++x) {
                        const bool p = (pRow[x] > 0);
                        const bool c = (cRow[x] > 0);
                        const bool n = (nRow[x] > 0);

                        // Fill single-slice holes and suppress single-slice spikes along Z.
                        if (!c && p && n) {
                            dRow[x] = 255;
                        } else if (c && !p && !n) {
                            dRow[x] = 0;
                        } else {
                            dRow[x] = c ? 255 : 0;
                        }
                    }
                }
            }

            masksForVolume.swap(bridgedMasks);
        }
    }

    // Remove scanner-frame strips: persistent thin border lines can survive per-slice cleanup
    // and become slab/wall artifacts after Marching Cubes.
    if (!masksForVolume.isEmpty() && width >= 24 && height >= 24) {
        const int borderBandX = qMax(2, width / 16);
        const int borderBandY = qMax(2, height / 16);
        const int maxStripWidthX = qMax(3, width / 24);
        const int maxStripWidthY = qMax(3, height / 24);
        const int totalSamplesPerColumn = qMax(1, depth * height);
        const int totalSamplesPerRow = qMax(1, depth * width);
        std::vector<int> columnForeground(width, 0);
        std::vector<int> columnSlicePresence(width, 0);
        std::vector<int> rowForeground(height, 0);
        std::vector<int> rowSlicePresence(height, 0);

        auto computeBorderOccupancy = [&](const QVector<cv::Mat>& masks) -> double {
            qint64 borderCount = 0;
            qint64 borderPixels = 0;
            for (int z = 0; z < masks.size(); ++z) {
                const cv::Mat& mask = masks[z];
                for (int y = 0; y < height; ++y) {
                    const uchar* row = mask.ptr<uchar>(y);
                    for (int x = 0; x < width; ++x) {
                        const bool inBorder = (x < borderBandX) || (x >= width - borderBandX)
                            || (y < borderBandY) || (y >= height - borderBandY);
                        if (!inBorder) {
                            continue;
                        }
                        ++borderPixels;
                        if (row[x] > 0) {
                            ++borderCount;
                        }
                    }
                }
            }
            return (borderPixels > 0)
                ? (100.0 * double(borderCount) / double(borderPixels))
                : 0.0;
        };

        const double borderOccBefore = computeBorderOccupancy(masksForVolume);
        const bool shouldRunBorderStripCleanup = (borderOccBefore >= 45.0);

        if (!shouldRunBorderStripCleanup) {
            qDebug() << "Skipping border-strip suppression: border occupancy too low:" << borderOccBefore;
        }

        if (shouldRunBorderStripCleanup) {
            for (int z = 0; z < depth; ++z) {
                const cv::Mat& mask = masksForVolume[z];
                std::vector<int> perSliceForeground(width, 0);
                std::vector<int> perSliceRowForeground(height, 0);

                for (int y = 0; y < height; ++y) {
                    const uchar* row = mask.ptr<uchar>(y);
                    for (int x = 0; x < width; ++x) {
                        if (row[x] > 0) {
                            ++perSliceForeground[x];
                            ++perSliceRowForeground[y];
                        }
                    }
                }

                for (int x = 0; x < width; ++x) {
                    const int fgCount = perSliceForeground[x];
                    columnForeground[x] += fgCount;

                    // Count slices where this column forms a noticeable vertical trace.
                    if (fgCount >= qMax(1, height / 4)) {
                        ++columnSlicePresence[x];
                    }
                }

                for (int y = 0; y < height; ++y) {
                    const int fgCount = perSliceRowForeground[y];
                    rowForeground[y] += fgCount;

                    // Count slices where this row forms a noticeable horizontal trace.
                    if (fgCount >= qMax(1, width / 4)) {
                        ++rowSlicePresence[y];
                    }
                }
            }

        std::vector<char> removeColumn(width, 0);
        auto markPersistentBorderColumnRun = [&](int xBegin, int xEnd) {
            int runStart = -1;
            for (int x = xBegin; x <= xEnd; ++x) {
                const double occupancyRatio = double(columnForeground[x]) / double(totalSamplesPerColumn);
                const double slicePresenceRatio = double(columnSlicePresence[x]) / double(qMax(1, depth));

                // Border rods can flicker slice-to-slice; accept either strong occupancy
                // or frequent per-slice presence as persistent evidence.
                const bool persistent = (occupancyRatio >= 0.35)
                    || (slicePresenceRatio >= 0.55 && occupancyRatio >= 0.18);

                if (persistent) {
                    if (runStart < 0) {
                        runStart = x;
                    }
                } else if (runStart >= 0) {
                    const int runEnd = x - 1;
                    const int runWidth = runEnd - runStart + 1;
                    if (runWidth <= maxStripWidthX) {
                        for (int xi = runStart; xi <= runEnd; ++xi) {
                            removeColumn[xi] = 1;
                        }
                    }
                    runStart = -1;
                }
            }

            if (runStart >= 0) {
                const int runEnd = xEnd;
                const int runWidth = runEnd - runStart + 1;
                if (runWidth <= maxStripWidthX) {
                    for (int xi = runStart; xi <= runEnd; ++xi) {
                        removeColumn[xi] = 1;
                    }
                }
            }
        };

        std::vector<char> removeRow(height, 0);
        auto markPersistentBorderRowRun = [&](int yBegin, int yEnd) {
            int runStart = -1;
            for (int y = yBegin; y <= yEnd; ++y) {
                const double occupancyRatio = double(rowForeground[y]) / double(totalSamplesPerRow);
                const double slicePresenceRatio = double(rowSlicePresence[y]) / double(qMax(1, depth));

                const bool persistent = (occupancyRatio >= 0.35)
                    || (slicePresenceRatio >= 0.55 && occupancyRatio >= 0.18);

                if (persistent) {
                    if (runStart < 0) {
                        runStart = y;
                    }
                } else if (runStart >= 0) {
                    const int runEnd = y - 1;
                    const int runWidth = runEnd - runStart + 1;
                    if (runWidth <= maxStripWidthY) {
                        for (int yi = runStart; yi <= runEnd; ++yi) {
                            removeRow[yi] = 1;
                        }
                    }
                    runStart = -1;
                }
            }

            if (runStart >= 0) {
                const int runEnd = yEnd;
                const int runWidth = runEnd - runStart + 1;
                if (runWidth <= maxStripWidthY) {
                    for (int yi = runStart; yi <= runEnd; ++yi) {
                        removeRow[yi] = 1;
                    }
                }
            }
        };

            markPersistentBorderColumnRun(0, borderBandX - 1);
            markPersistentBorderColumnRun(width - borderBandX, width - 1);
            markPersistentBorderRowRun(0, borderBandY - 1);
            markPersistentBorderRowRun(height - borderBandY, height - 1);

            int removedColumns = 0;
            for (int x = 0; x < width; ++x) {
                if (!removeColumn[x]) {
                    continue;
                }
                ++removedColumns;
                for (int z = 0; z < depth; ++z) {
                    masksForVolume[z].col(x).setTo(0);
                }
            }

            int removedRows = 0;
            for (int y = 0; y < height; ++y) {
                if (!removeRow[y]) {
                    continue;
                }
                ++removedRows;
                for (int z = 0; z < depth; ++z) {
                    masksForVolume[z].row(y).setTo(0);
                }
            }

            if (removedColumns > 0 || removedRows > 0) {
                qDebug() << "Removed persistent border-strip artifacts. Columns:" << removedColumns
                         << "Rows:" << removedRows;
            }
        }

        const double borderOccAfter = computeBorderOccupancy(masksForVolume);
        qDebug() << "Border occupancy before/after strip suppression (%):"
                 << borderOccBefore << "->" << borderOccAfter;
    }

    // Final safety-net: remove border-only components that never enter the interior ROI.
    // This catches persistent scanner strips even when they are fragmented slice-to-slice.
    if (!masksForVolume.isEmpty() && width >= 24 && height >= 24 && adaptiveBorderOcc >= 55.0) {
        const int interiorX0 = width / 8;
        const int interiorX1 = width - interiorX0;
        const int interiorY0 = height / 8;
        const int interiorY1 = height - interiorY0;
        const cv::Rect interiorRect(interiorX0, interiorY0,
                                    qMax(1, interiorX1 - interiorX0),
                                    qMax(1, interiorY1 - interiorY0));

        int removedComponentsTotal = 0;
        int affectedSlices = 0;
        for (int z = 0; z < depth; ++z) {
            cv::Mat& mask = masksForVolume[z];
            if (mask.empty()) {
                continue;
            }

            cv::Mat labels, stats, centroids;
            const int cc = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
            if (cc <= 1) {
                continue;
            }

            int removedInSlice = 0;
            for (int label = 1; label < cc; ++label) {
                const int area = stats.at<int>(label, cv::CC_STAT_AREA);
                if (area <= qMax(8, (width * height) / 5000)) {
                    continue;
                }

                const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
                const int top = stats.at<int>(label, cv::CC_STAT_TOP);
                const int compWidth = stats.at<int>(label, cv::CC_STAT_WIDTH);
                const int compHeight = stats.at<int>(label, cv::CC_STAT_HEIGHT);
                const cv::Rect bbox(left, top, compWidth, compHeight);

                const bool touchesBorder = (left <= 1)
                    || (top <= 1)
                    || (left + compWidth >= width - 1)
                    || (top + compHeight >= height - 1);
                if (!touchesBorder) {
                    continue;
                }

                const bool intersectsInterior = ((bbox & interiorRect).area() > 0);
                if (intersectsInterior) {
                    continue;
                }

                mask.setTo(0, labels == label);
                ++removedInSlice;
            }

            if (removedInSlice > 0) {
                ++affectedSlices;
                removedComponentsTotal += removedInSlice;
            }
        }

        if (removedComponentsTotal > 0) {
            qDebug() << "Final border-only component cleanup removed"
                     << removedComponentsTotal << "components across"
                     << affectedSlices << "slices";
        }
    }

    const qint64 totalVoxels = static_cast<qint64>(width) * height * depth;
    qint64 filledVoxels = 0;

    for (int z = 0; z < depth; ++z) {
        const cv::Mat& mask = masksForVolume[z];
        if (mask.empty() || mask.cols != width || mask.rows != height) {
            qWarning() << "Adaptive mask invalid at slice" << z << "size:" << mask.cols << "x" << mask.rows;
            continue;
        }

        for (int y = 0; y < height; ++y) {
            const uchar* maskRow = mask.ptr<uchar>(y);
            float* dstRow = volume[z][y].data();
            for (int x = 0; x < width; ++x) {
                const float value = (maskRow[x] > 0) ? 1.0f : 0.0f;
                dstRow[x] = value;
                if (value >= 0.5f) {
                    ++filledVoxels;
                }
            }
        }

        if (progressCallback && ((z + 1) % 8 == 0 || z + 1 == depth)) {
            const int progress = 20 + (((z + 1) * 60) / qMax(1, depth));
            progressCallback(progress, QString("Building adaptive 16-bit volume... %1/%2").arg(z + 1).arg(depth));
        }
    }

    const double occupancy = totalVoxels > 0 ? (100.0 * static_cast<double>(filledVoxels) / static_cast<double>(totalVoxels)) : 0.0;
    qDebug() << "16-bit adaptive segmentation threshold (input):" << boundedThreshold;
    qDebug() << "16-bit adaptive segmentation metrics:" << ImageProcessor::metricsToString(metrics);
    qDebug() << "16-bit adaptive binary volume occupancy:" << occupancy << "%";

    if (progressCallback) {
        progressCallback(80, QString("Adaptive segmentation occupancy %1%")
            .arg(occupancy, 0, 'f', 2));
    }

    if (occupancy < 0.1 || occupancy > 99.9) {
        qWarning() << "16-bit volume is nearly uniform with this threshold; mesh may be empty.";
    }

    if (progressCallback) {
        progressCallback(90, "16-bit volume ready. Preparing GPU execution...");
    }

    qDebug() << "16-bit volume created successfully.";
    return volume;
}

MainWindow::VolumeData16 MainWindow::buildRawVolume16(const QVector<cv::Mat>& slices16) const {
    VolumeData16 volume;

    if (slices16.isEmpty()) {
        return volume;
    }

    const int width = slices16.first().cols;
    const int height = slices16.first().rows;
    const int depth = slices16.size();

    volume.resize(depth);
    for (int z = 0; z < depth; ++z) {
        volume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            volume[z][y].resize(width);
            const quint16* row = slices16[z].ptr<quint16>(y);
            for (int x = 0; x < width; ++x) {
                volume[z][y][x] = row ? row[x] : 0;
            }
        }
    }

    return volume;
}

// histogram-based heuristic to estimate material thresholds for 16-bit segmentation volumes
void MainWindow::estimateMaterialThresholds(const VolumeData16& volume16, int& ceramicThreshold16, int& boneThreshold16) const {
    ceramicThreshold16 = 200;
    boneThreshold16 = 100;

    if (volume16.isEmpty() || volume16[0].isEmpty() || volume16[0][0].isEmpty()) {
        return;
    }

    QVector<int> histogram(256, 0);
    qint64 totalSamples = 0;

    for (const auto& slice : volume16) {
        for (const auto& row : slice) {
            for (quint16 value : row) {
                if (value == 0) {
                    continue;
                }
                const int normalized = qBound(0, int(qRound(value / 257.0)), 255);
                ++histogram[normalized];
                ++totalSamples;
            }
        }
    }

    if (totalSamples <= 0) {
        return;
    }

    auto percentileFromHistogram = [&](double percentile) -> int {
        const qint64 target = qMax<qint64>(1, qRound(totalSamples * percentile));
        qint64 cumulative = 0;
        for (int i = 0; i < histogram.size(); ++i) {
            cumulative += histogram[i];
            if (cumulative >= target) {
                return i;
            }
        }
        return 255;
    };

    boneThreshold16 = qBound(10, percentileFromHistogram(0.60), 230);

    // *** KEY FIX: Use currentThreshold16 (the Otsu/user bone threshold used by
    // Marching Cubes) as the lower cutoff for ceramic analysis.
    //
    // Previous approach (percentile of ALL voxels) failed because:
    //   - 95th pct of 835M voxels (mostly air/soft tissue) lands at ~114 normalized
    //   - currentThreshold16=28180 -> normalized=110  (bone/background boundary)
    //   - ceramic threshold 114 was only 4 above bone -> classifies all dense bone as ceramic
    //
    // Correct approach: build a RESTRICTED histogram of only voxels brighter than the
    // Otsu bone cutoff, then take the 75th percentile of THAT bright subset.
    // This means "top 25% of foreground-bright voxels" = ceramic/metal density.
    const int boneCutoffNorm = qBound(boneThreshold16,
                                      int(qRound(currentThreshold16 / 257.0)),
                                      240);

    // Exclude scanner/metal artifact voxels (intensity >= 238 normalized, ~61000 raw 16-bit)
    // from the ceramic threshold calculation.  Datasets with metal hardware or MRI saturation
    // can have millions of voxels at [248-255] that inflate the 75th-pct ceramic threshold
    // far above the actual scaffold signal range, causing the intensity gate to reject most
    // real scaffold wall vertices.  INTENSITY_MAX=240 in classification already gates these;
    // match that cutoff here so the threshold reflects actual scaffold, not artifact, density.
    const int artifactCap = 238;  // ≈ INTENSITY_MAX(240)/257*255

    qint64 brightTotal = 0;
    for (int i = boneCutoffNorm; i < artifactCap; ++i) {
        brightTotal += histogram[i];
    }

    int ceramicCandidate = boneCutoffNorm + 10;  // safe fallback
    if (brightTotal > 0) {
        const qint64 ceramicTarget = qMax<qint64>(1, qRound(brightTotal * 0.75));
        qint64 brightCumulative = 0;
        for (int i = boneCutoffNorm; i < artifactCap; ++i) {
            brightCumulative += histogram[i];
            if (brightCumulative >= ceramicTarget) {
                ceramicCandidate = i;
                break;
            }
        }
    }

    ceramicThreshold16 = qBound(boneCutoffNorm + 5, ceramicCandidate, 255);

    // --- Diagnostic logging ---
    qDebug().noquote() << "-----[ Threshold Estimation ]-----";
    qDebug() << "  Classification: 16-bit raw data normalized to 8-bit (0-255) for comparison";
    qDebug() << "  Voxels analyzed (non-zero):" << totalSamples;
    qDebug() << "  Bone threshold (60th pct of all voxels, 0-255):" << boneThreshold16;
    qDebug() << "  currentThreshold16 (Otsu/user, raw 16-bit):" << currentThreshold16
             << " => normalized:" << boneCutoffNorm;
    qDebug() << "  Bright voxels (>= boneCutoff):" << brightTotal
             << QString("(%1% of foreground)").arg(100.0 * brightTotal / double(totalSamples), 0, 'f', 2);
    qDebug() << "  Ceramic threshold (75th pct of bright voxels, 0-255):" << ceramicThreshold16;
    qDebug() << "  Histogram tail (8-bin sums, 0-255 normalized):";
    for (int i = 80; i < 256; i += 8) {
        qint64 sum = 0;
        for (int j = i; j < qMin(i + 8, 256); ++j) sum += histogram[j];
        if (sum > 0) {
            const QString marker = (i <= boneCutoffNorm && boneCutoffNorm < i + 8) ? " <-- bone cutoff"
                                 : (i <= ceramicThreshold16 && ceramicThreshold16 < i + 8) ? " <-- ceramic threshold"
                                 : "";
            qDebug().noquote() << QString("    [%1-%2]: %3%4").arg(i,3).arg(i+7,3).arg(sum).arg(marker);
        }
    }
    qDebug().noquote() << "-----[ Threshold Estimation End ]-----";
}

 void MainWindow::updateIsoLevel(int value) {
     qDebug() << "Entering updateIsoLevel";

     if (isoChangeTimer.elapsed() < 100) {
         qDebug() << "ISO change too fast, skipping";
         return; // 100ms cooldown
     }
     isoChangeTimer.restart();

     float isoLevel = value / 1000.0f;
     qDebug() << "Generating mesh with ISO:" << isoLevel;

     if (currentVolume.isEmpty()) {
         qWarning() << "No volume data!";
         return;
     }

    qDebug() << "Generating mesh...";
    currentMesh = MarchingCubes::generateMesh(
        currentVolume,
        isoLevel,
        voxelSpacingX,
        voxelSpacingY,
        voxelSpacingZ
    );
    if (currentMesh.vertices.isEmpty()) {
        qDebug() << "Full-volume meshing returned empty result. Falling back to streaming.";
        currentMesh = MarchingCubes::generateMeshStreaming(
            currentVolume,
            isoLevel,
            8,
            voxelSpacingX,
            voxelSpacingY,
            voxelSpacingZ
        );
    }
     qDebug() << "Generated mesh:"
              << currentMesh.vertices.size() << "vertices,"
              << currentMesh.indices.size() / 3 << "triangles";

     qDebug() << "Updating GLWidget...";
     if (mainTabs && mainTabs->count() > 0) {
         mainTabs->setCurrentIndex(0);
     }
     
     // Apply material classification if enabled and available
     if (!currentSegmentationVolume16.isEmpty() && materialColorsEnabled) {
         MarchingCubes::Mesh classifiedMesh = classifyMeshByMaterial(currentMesh, currentSegmentationVolume16);
         glWidget->updateMeshWithMaterials(classifiedMesh.vertices, classifiedMesh.indices, classifiedMesh.vertexColors);
     } else {
         glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
     }
     
     glWidget->update();

     qDebug() << "Exiting updateIsoLevel";
 }


 // Convert QImage to cv::Mat
 cv::Mat MainWindow::QImageToCvMat(const QImage& img) {
     if (img.isNull()) {
         return cv::Mat();
     }

     if (img.format() == QImage::Format_Grayscale8 || img.format() == QImage::Format_Indexed8) {
         cv::Mat mat(img.height(), img.width(), CV_8UC1, const_cast<uchar*>(img.constBits()), img.bytesPerLine());
         return mat.clone();
     }

     QImage gray = img.convertToFormat(QImage::Format_Grayscale8);
     cv::Mat mat(gray.height(), gray.width(), CV_8UC1, const_cast<uchar*>(gray.constBits()), gray.bytesPerLine());
     return mat.clone();
 }

 // Convert cv::Mat to QImage
 QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
     if (mat.type() == CV_8UC1) {
         // Ensure returned QImage owns its memory after function returns.
         return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8).copy();
     } else if (mat.type() == CV_8UC3) {
         return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped().copy();
     } else {
         qWarning() << "Unsupported CV Mat type:" << mat.type();
         return QImage();
     }
 }


 void MainWindow::exportSTL() {
     if (isExportingStl) {
         statusBar()->showMessage("STL export already in progress...", 2500);
         return;
     }

     if (currentMesh.vertices.isEmpty() || currentMesh.indices.isEmpty()) {
         statusBar()->showMessage("No mesh to export!", 3000);
         return;
     }

     const int triangleCount = currentMesh.indices.size() / 3;
     const qint64 estimatedBytes = estimateBinaryStlBytes(triangleCount);
     logExportEstimate(triangleCount, estimatedBytes);

     ExportOptions exportOptions;
     if (!promptExportOptions(this, triangleCount, estimatedBytes, &exportOptions)) {
         return;
     }

     qDebug() << "[STL Export] Start | vertices:" << currentMesh.vertices.size()
              << "indices:" << currentMesh.indices.size()
              << "preset:" << presetName(exportOptions.preset)
              << "simplify:" << (exportOptions.simplifyBeforeExport ? "yes" : "no")
              << "targetFaces:" << exportOptions.targetFaceCount
              << "aggressiveness:" << exportOptions.aggressiveness;

     QString fileName = QFileDialog::getSaveFileName(this, "Save STL File", "", "STL Files (*.stl)");
     if (fileName.isEmpty()) {
         return;
     }
     if (!fileName.endsWith(".stl", Qt::CaseInsensitive)) {
         fileName += ".stl";
     }
     qDebug() << "[STL Export] Target path:" << fileName;

     stlTimer.start();

     isExportingStl = true;
     if (generate3DAct) generate3DAct->setEnabled(false);
     if (exportSTLAct) exportSTLAct->setEnabled(false);
     if (threshold16SpinBox) threshold16SpinBox->setEnabled(false);
     if (threshold16AutoButton) threshold16AutoButton->setEnabled(false);

     auto* exportDialog = new QProgressDialog("Preparing STL export...", "Cancel", 0, 100, this);
     exportDialog->setWindowTitle("STL Export Progress");
     exportDialog->setWindowModality(Qt::NonModal);
     exportDialog->setMinimumDuration(0);
     exportDialog->setAutoClose(false);
     exportDialog->setAutoReset(false);
     exportDialog->show();
     centerDialogOnWidget(exportDialog, this);

     QPointer<MainWindow> weakWindow(this);
     QPointer<QProgressDialog> weakDialog(exportDialog);
     auto cancelRequested = std::make_shared<std::atomic_bool>(false);
     connect(exportDialog, &QProgressDialog::canceled, this, [cancelRequested]() {
         cancelRequested->store(true, std::memory_order_relaxed);
     });

     auto finishUi = [this, exportDialog]() {
         isExportingStl = false;
         if (generate3DAct) generate3DAct->setEnabled(true);
         if (exportSTLAct) exportSTLAct->setEnabled(true);
         if (threshold16SpinBox) threshold16SpinBox->setEnabled(true);
         if (threshold16AutoButton) threshold16AutoButton->setEnabled(true);
         if (exportDialog) {
             exportDialog->close();
             exportDialog->deleteLater();
         }
     };

     auto startBinaryExport = [this, fileName, weakWindow, weakDialog, cancelRequested, finishUi](MarchingCubes::Mesh meshToExport) mutable {
         if (cancelRequested->load(std::memory_order_relaxed)) {
             finishUi();
             return;
         }

         auto* watcher = new QFutureWatcher<StlExportResult>(this);
         connect(watcher, &QFutureWatcher<StlExportResult>::finished, this,
                 [this, watcher, finishUi]() {
             const StlExportResult result = watcher->result();

             finishUi();

             if (!result.success) {
                 qWarning() << "[STL Export] Failed:" << result.errorMessage;
                 QMessageBox::warning(this,
                                      "Export Failed",
                                      result.errorMessage.isEmpty()
                                          ? "STL export failed."
                                          : result.errorMessage);
                 watcher->deleteLater();
                 return;
             }

             qDebug().noquote() << "-------";
             qDebug().noquote() << QString("Export STL: %1 s").arg(double(stlTimer.elapsed()) / 1000.0, 0, 'f', 2);
             qDebug() << "[STL Export] Success | triangles:" << result.writtenTriangles
                      << "bytes:" << result.bytesOnDisk;
             if (result.skippedInvalidTriangles > 0) {
                 qWarning() << "[STL Export] Skipped" << result.skippedInvalidTriangles << "invalid triangle(s).";
             }
             if (result.skippedDegenerateTriangles > 0) {
                 qWarning() << "[STL Export] Skipped" << result.skippedDegenerateTriangles << "degenerate triangle(s).";
             }
             qDebug() << "[STL Export] Completed with" << result.writtenTriangles << "triangle(s).";
             qDebug().noquote() << "-------";
             watcher->deleteLater();
         });

         watcher->setFuture(QtConcurrent::run([fileName,
                                               verticesCopy = meshToExport.vertices,
                                               indicesCopy = meshToExport.indices,
                                               weakWindow,
                                               weakDialog,
                                               cancelRequested]() mutable -> StlExportResult {
             return exportMeshToBinaryStlFile(
                 fileName,
                 verticesCopy,
                 indicesCopy,
                 [weakWindow, weakDialog](int progress, const QString& message) {
                     if (!weakWindow) {
                         return;
                     }

                     QMetaObject::invokeMethod(weakWindow.data(), [weakDialog, progress, message]() {
                         if (!weakDialog) {
                             return;
                         }
                         weakDialog->setValue(qBound(0, progress, 100));
                         weakDialog->setLabelText(message);
                     }, Qt::QueuedConnection);
                 },
                 [cancelRequested, weakDialog]() {
                     return cancelRequested->load(std::memory_order_relaxed) || (weakDialog && weakDialog->wasCanceled());
                 });
         }));
     };

     const MarchingCubes::Mesh meshCopy = currentMesh;
     const bool simplifyRequested = exportOptions.simplifyBeforeExport && exportOptions.preset != ExportPreset::Ultra;
     if (!simplifyRequested) {
         qDebug().noquote() << QString("[STL Export] Simplification skipped. checkbox=%1 preset=%2")
                                 .arg(exportOptions.simplifyBeforeExport ? "true" : "false")
                                 .arg(presetName(exportOptions.preset));
         startBinaryExport(meshCopy);
         return;
     }

     exportDialog->setValue(0);
     exportDialog->setLabelText("Simplifying mesh with QEM...\nPlease wait...");
     exportDialog->setRange(0, 0);
     QApplication::processEvents();

     auto* simplifyWatcher = new QFutureWatcher<MeshSimplifier::SimplifyReport>(this);
     connect(simplifyWatcher, &QFutureWatcher<MeshSimplifier::SimplifyReport>::finished, this,
             [this, simplifyWatcher, startBinaryExport, cancelRequested, meshCopy, exportDialog]() mutable {
         if (cancelRequested->load(std::memory_order_relaxed)) {
             isExportingStl = false;
             if (generate3DAct) generate3DAct->setEnabled(true);
             if (exportSTLAct) exportSTLAct->setEnabled(true);
             if (threshold16SpinBox) threshold16SpinBox->setEnabled(true);
             if (threshold16AutoButton) threshold16AutoButton->setEnabled(true);
             if (exportDialog) {
                 exportDialog->close();
                 exportDialog->deleteLater();
             }
             simplifyWatcher->deleteLater();
             return;
         }

         const MeshSimplifier::SimplifyReport report = simplifyWatcher->result();
         simplifyWatcher->deleteLater();
         qDebug().noquote() << QString("[Simplifier] completed. success=%1 inputFaces=%2 outputFaces=%3 message=%4")
                                 .arg(report.success ? "true" : "false")
                                 .arg(report.inputFaceCount)
                                 .arg(report.outputFaceCount)
                                 .arg(report.message);

         if (report.outputFaceCount == report.inputFaceCount) {
             qWarning() << "[Simplifier] Triangle count did not change; export will proceed with the original mesh unless OpenMesh produced a reduced mesh.";
         }

         if (!report.success) {
             if (MeshSimplifier::isOpenMeshAvailable()) {
                 isExportingStl = false;
                 if (generate3DAct) generate3DAct->setEnabled(true);
                 if (exportSTLAct) exportSTLAct->setEnabled(true);
                 if (threshold16SpinBox) threshold16SpinBox->setEnabled(true);
                 if (threshold16AutoButton) threshold16AutoButton->setEnabled(true);
                 if (exportDialog) {
                     exportDialog->close();
                     exportDialog->deleteLater();
                 }
                 QMessageBox::warning(this, "Simplification Failed", report.message.isEmpty() ? "Mesh simplification failed." : report.message);
                 return;
             }

             qWarning() << "[Simplifier]" << report.message << "Proceeding without simplification.";
             startBinaryExport(meshCopy);
             return;
         }

         qDebug() << "[Simplifier] input faces:" << report.inputFaceCount
                  << "output faces:" << report.outputFaceCount
                  << "backend:" << MeshSimplifier::backendName();
         startBinaryExport(report.mesh);
     });

     if (exportDialog) {
         exportDialog->setLabelText("Simplifying mesh with QEM...\nPlease wait...");
     }
     simplifyWatcher->setFuture(QtConcurrent::run([meshCopy,
                                                   targetFaces = exportOptions.targetFaceCount,
                                                   aggressiveness = exportOptions.aggressiveness]() {
         return MeshSimplifier::simplifyMeshDetailed(meshCopy, targetFaces, aggressiveness);
     }));
 }

 void MainWindow::generateMesh() {

     if (isExportingStl) {
         statusBar()->showMessage("STL export is still running. Please wait.", 3000);
         return;
     }

     if (isGeneratingMesh) return;
     if (isOtsuRunning) {
         statusBar()->showMessage("Otsu is running. Please wait...", 2500);
         return;
     }

     isGeneratingMesh = true;
     statusBar()->showMessage("Starting mesh generation...");
    // Reset dialog state here (not in finalize) to avoid re-show via setValue
    loadingDialog->reset();
    loadingDialog->setRange(0, 100);
    loadingDialog->setValue(0);
    loadingDialog->setLabelText("Generating 3D mesh...");
    loadingDialog->show();
    centerDialogOnWidget(loadingDialog, this);
    QApplication::processEvents();

    meshTimer.start();

     // Threshold selection logic: keep the active threshold aligned with the UI before meshing.
     currentThreshold16 = threshold16SpinBox ? threshold16SpinBox->value() : currentThreshold16;
     currentThreshold = qBound(0.0, double(currentThreshold16) / 65535.0, 1.0);

     QFuture<void> future = QtConcurrent::run([this]() {
         const int sliceCount = loadedImages.size();
         const float targetIso = 0.50f;
         const QString profile = (sliceCount <= 48) ? "sparse" : ((sliceCount >= 200) ? "dense" : "balanced");
         qDebug() << "Reconstruction profile:" << profile
                  << "slices:" << sliceCount
                  << "segmentation mode:" << (hasSegmentation16Data ? "native 16-bit slices" : "old thresholded slices")
                  << "iso:" << targetIso
                  << "spacing:" << voxelSpacingX << voxelSpacingY << voxelSpacingZ;

         auto reportProgress = [this](int progress, const QString& message) {
             QMetaObject::invokeMethod(this, [this, progress, message]() {
                 updateLoadingDialog(progress, message);
             }, Qt::QueuedConnection);
         };

         reportProgress(1, "Preparing volume data...");

         VolumeData volume;
         if (hasSegmentation16Data && !segmentationSlices16.isEmpty() && segmentationSlices16.size() == loadedImages.size()) {
             qDebug() << "Using native 16-bit segmentation path for mesh generation.";
             currentSegmentationVolume16 = buildRawVolume16(segmentationSlices16);
             volume = convertToVolume16(segmentationSlices16, currentThreshold16, 1.0f, reportProgress);
         } else {
             // Fallback path for non-16-bit stacks.
             const QVector<QImage>* meshSource = &loadedImages;
             if (meshSource->isEmpty()) {
                 qWarning() << "No source images are available; cannot generate mesh.";
                 QMetaObject::invokeMethod(this, [this]() {
                     statusBar()->showMessage("No source images available. Reload dataset.", 4000);
                 }, Qt::QueuedConnection);
                 return;
             }

             qDebug() << "Using 8-bit thresholded image path for mesh generation.";
             currentSegmentationVolume16.clear();
             volume = convertToVolume(*meshSource, reportProgress);
         }

         if (!volume.isEmpty()) {
             // Mesh generation logic: production path is a single full-volume pass first.
             reportProgress(84, "Running Marching Cubes (full-volume GPU)...");
             reconstructTimer.start();
             pendingMesh = MarchingCubes::generateMesh(
                 volume,
                 targetIso,
                 voxelSpacingX,
                 voxelSpacingY,
                 voxelSpacingZ
             );

             if (pendingMesh.vertices.isEmpty()) {
                 reportProgress(88, "Falling back to streaming GPU meshing...");
                 pendingMesh = MarchingCubes::generateMeshStreaming(
                     volume,
                     targetIso,
                     8,
                     voxelSpacingX,
                     voxelSpacingY,
                     voxelSpacingZ
                 );
             }
             reconstructElapsed = reconstructTimer.elapsed();

             reportProgress(90, "Mesh computation complete...");
         }
     });

     meshGenerationWatcher.setFuture(future);
 }

 void MainWindow::handleMeshGenerationStarted() {
     if (!isGeneratingMesh) {
         return;
     }
     updateLoadingDialog(0, "Generating 3D mesh...");
     loadingDialog->show();
     centerDialogOnWidget(loadingDialog, this);
     QApplication::processEvents();
 }

 void MainWindow::handleMeshComputationFinished() {
     if (!pendingMesh.vertices.isEmpty()) {
        // Save threshold as good threshold for warm-start on next dataset
        imageProcessor.saveLastGoodThreshold(currentThreshold16);
        
        // Ensure the OpenGL tab is active before uploading buffers.
        if (mainTabs && mainTabs->count() > 0) {
            mainTabs->setCurrentIndex(0);
        }
        updateLoadingDialog(92, "Uploading mesh to renderer...");
        currentMesh = pendingMesh; // Update currentMesh for export
        
        // Enable Material Colors controls now that mesh exists
        if (materialColorsCheckBox) {
            materialColorsCheckBox->setEnabled(true);
            // Auto-check Material Colors after mesh generation
            materialColorsCheckBox->blockSignals(true);
            materialColorsCheckBox->setChecked(true);
            materialColorsCheckBox->blockSignals(false);
        }
        // Signal must be set BEFORE the upload calls because meshUpdateComplete is emitted
        // synchronously inside updateMesh/updateMeshWithMaterials, which triggers
        // handleMeshRenderingFinished before execution returns here.
        waitingForMeshUploadCompletion = true;
        isGeneratingMesh = false;

        // Classify mesh by material and upload with colors
        if (!currentSegmentationVolume16.isEmpty()) {
            qDebug() << "Classifying mesh with materials from segmentation volume...";
            currentMesh = classifyMeshByMaterial(currentMesh, currentSegmentationVolume16);
            glWidget->updateMeshWithMaterials(currentMesh.vertices, currentMesh.indices, currentMesh.vertexColors);
        } else {
            // If no segmentation volume, upload without materials
            glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
        }

        // Mark generation complete and enable material overlay
        materialColorsEnabled = true;
        glWidget->setMaterialColorsEnabled(true);
        // Show overlay legend (blockSignals above prevented the toggle signal from firing)
        if (materialLegendWidget)
            materialLegendWidget->setVisible(true);
        if (overlayLegendWidget && vizTab) {
            overlayLegendWidget->adjustSize();
            const int margin = 10;
            overlayLegendWidget->move(vizTab->width() - overlayLegendWidget->width() - margin, margin);
            overlayLegendWidget->raise();
            overlayLegendWidget->setVisible(true);
        }

        qDebug().noquote() << "-------";
        qDebug().noquote() << QString("Reconstruction (MC): %1 s").arg(double(reconstructElapsed) / 1000.0, 0, 'f', 2);
        qDebug().noquote() << QString("Generate 3D (total): %1 s").arg(double(meshTimer.elapsed()) / 1000.0, 0, 'f', 2);
     } else {
         handleMeshRenderingFinished();
     }
 }

 void MainWindow::handleMeshRenderingFinished() {
    const bool shouldFinalize = (waitingForMeshUploadCompletion || isGeneratingMesh);
    isGeneratingMesh = false;
    waitingForMeshUploadCompletion = false;

    if (shouldFinalize) {
        finalizeMeshProgressDialog();
    }

     if (pendingMesh.vertices.isEmpty()) {
         QMessageBox::warning(this, "Mesh Generation Failed",
                              "No mesh vertices were generated.\n\n"
                              "Possible causes:\n"
                              "- GPU memory is insufficient for current volume size\n"
                              "- Threshold produced nearly uniform volume\n"
                              "- CUDA execution failed\n\n"
                              "Try reducing image/crop size or adjusting threshold.");
     } else {
         statusBar()->showMessage("Mesh generation completed", 3000);
     }
     pendingMesh = MarchingCubes::Mesh(); // Clear stored mesh
 }

void MainWindow::updateLoadingDialog(int progress, const QString& message) {
    if (!loadingDialog || !isGeneratingMesh) {
        return;
    }

    loadingDialog->setValue(qBound(0, progress, 100));
    if (!message.isEmpty()) {
        loadingDialog->setLabelText(message);
    }
}

void MainWindow::finalizeMeshProgressDialog() {
    if (!loadingDialog) {
        return;
    }
    // Close and hide ONLY — do NOT call setValue/reset here.
    // QProgressDialog with minimumDuration=0 re-shows itself whenever setValue is called
    // after close/hide, so we defer the reset to just before the next show() in generateMesh().
    loadingDialog->close();
    loadingDialog->hide();
}

 void MainWindow::saveProcessedImage(const QImage& image, const QString& filePath) {
     if (image.save(filePath)) {
         qDebug() << "Processed image saved to:" << filePath;
     } else {
         qWarning() << "Failed to save processed image to:" << filePath;
     }
 }

 void MainWindow::loadMesh() {
     QFileDialog dialog(this, "Select Mesh File(s)");
     dialog.setFileMode(QFileDialog::ExistingFiles);
     dialog.setNameFilters({"STL Files (*.stl)", "Text Files (*.txt)"});
     dialog.selectNameFilter("STL Files (*.stl)");

     if (dialog.exec() != QDialog::Accepted) {
         qWarning() << "No mesh file selected.";
         return;
     }

     const QStringList selectedPaths = dialog.selectedFiles();
     if (selectedPaths.isEmpty()) {
         qWarning() << "No mesh file selected.";
         return;
     }

     bool hasStl = false;
     for (const QString& p : selectedPaths) {
         if (QFileInfo(p).suffix().compare("stl", Qt::CaseInsensitive) == 0) {
             hasStl = true;
             break;
         }
     }

     QString verticesPath;
     QString facesPath;
     QString sourceDescription;

     if (hasStl) {
         if (selectedPaths.size() != 1) {
             QMessageBox::warning(this, "Load Mesh Failed", "Please select exactly one STL file.");
             return;
         }
         verticesPath = selectedPaths.first();
         sourceDescription = QString("STL: %1").arg(QFileInfo(verticesPath).fileName());
     } else {
         // TXT legacy mode: allow selecting both vertices and faces in one dialog.
         if (selectedPaths.size() == 2) {
             const QString a = selectedPaths[0];
             const QString b = selectedPaths[1];
             const QString aBase = QFileInfo(a).completeBaseName().toLower();
             const QString bBase = QFileInfo(b).completeBaseName().toLower();

             const bool aLooksVertex = aBase.contains("vert");
             const bool bLooksVertex = bBase.contains("vert");
             const bool aLooksFace = aBase.contains("face") || aBase.contains("tri");
             const bool bLooksFace = bBase.contains("face") || bBase.contains("tri");

             if (aLooksVertex && bLooksFace) {
                 verticesPath = a;
                 facesPath = b;
             } else if (bLooksVertex && aLooksFace) {
                 verticesPath = b;
                 facesPath = a;
             } else {
                 // Fallback to selection order.
                 verticesPath = a;
                 facesPath = b;
             }
         } else if (selectedPaths.size() == 1) {
             verticesPath = selectedPaths.first();
             facesPath = QFileDialog::getOpenFileName(this, "Select Faces File", "", "Text Files (*.txt)");
             if (facesPath.isEmpty()) {
                 qWarning() << "No faces file selected.";
                 return;
             }
         } else {
             QMessageBox::warning(this, "Load Mesh Failed", "Please select one STL file or two TXT files (vertices + faces).");
             return;
         }

         sourceDescription = QString("TXT: %1 + %2")
             .arg(QFileInfo(verticesPath).fileName())
             .arg(QFileInfo(facesPath).fileName());
     }

     auto* progressDialog = new QProgressDialog("Importing mesh...", QString(), 0, 0, this);
     progressDialog->setWindowTitle("Mesh Import");
     progressDialog->setWindowModality(Qt::WindowModal);
     progressDialog->setCancelButton(nullptr);
     progressDialog->setMinimumDuration(0);
     progressDialog->setAutoClose(false);
     progressDialog->setAutoReset(false);
     progressDialog->setLabelText(QString("Parsing %1\nPlease wait...").arg(sourceDescription));
    progressDialog->adjustSize();
     progressDialog->show();
     centerDialogOnWidget(progressDialog, glWidget ? static_cast<QWidget*>(glWidget) : static_cast<QWidget*>(this));
     QPointer<QProgressDialog> centerLater = progressDialog;
     QTimer::singleShot(0, this, [this, centerLater]() {
         if (centerLater) {
             centerDialogOnWidget(centerLater, glWidget ? static_cast<QWidget*>(glWidget) : static_cast<QWidget*>(this));
         }
     });
     setEnabled(false);

    meshLoadTimer.start();

     auto* watcher = new QFutureWatcher<LoadedMeshResult>(this);
     connect(watcher, &QFutureWatcher<LoadedMeshResult>::finished, this,
             [this, watcher, progressDialog]() {
         const LoadedMeshResult result = watcher->result();

         if (!result.ok) {
             if (progressDialog) {
                 progressDialog->close();
                 progressDialog->deleteLater();
             }
             setEnabled(true);
             QMessageBox::warning(this, "Load Mesh Failed", result.error.isEmpty() ? "Unknown mesh loading error." : result.error);
             watcher->deleteLater();
             return;
         }

         currentMesh.vertices = result.vertices;
         currentMesh.indices = result.indices;
         pendingMesh = currentMesh;

         if (mainTabs && mainTabs->count() > 0) {
             mainTabs->setCurrentIndex(0);
         }

         if (progressDialog) {
             progressDialog->setLabelText("Uploading mesh to renderer...\nPlease wait...");
             progressDialog->adjustSize();
             centerDialogOnWidget(progressDialog, glWidget ? static_cast<QWidget*>(glWidget) : static_cast<QWidget*>(this));
         }
         QApplication::processEvents();

         QPointer<QProgressDialog> uploadDialog = progressDialog;
         connect(glWidget, &GLWidget::meshUpdateComplete, progressDialog,
                 [this, watcher, uploadDialog]() {
             if (uploadDialog) {
                 uploadDialog->close();
                 uploadDialog->deleteLater();
             }
             setEnabled(true);
             qDebug().noquote() << "-------";
             qDebug().noquote() << QString("Load mesh: %1 s").arg(double(meshLoadTimer.elapsed()) / 1000.0, 0, 'f', 2);
             statusBar()->showMessage(
                 QString("Loaded mesh (%1 vertices, %2 triangles)")
                     .arg(currentMesh.vertices.size())
                     .arg(currentMesh.indices.size() / 3),
                 5000
             );
             qDebug().noquote() << "-------";
             watcher->deleteLater();
         });

         QTimer::singleShot(0, this, [this]() {
             glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
             glWidget->update();
         });
     });

     const bool importStl = hasStl;
     watcher->setFuture(QtConcurrent::run([verticesPath, facesPath, importStl, sourceDescription]() -> LoadedMeshResult {
         LoadedMeshResult result;
         result.sourceDescription = sourceDescription;

         QString error;
         if (importStl) {
             if (!loadStlMesh(verticesPath, &result.vertices, &result.indices, &error)) {
                 result.error = QString("Failed to parse STL file.\n\n%1").arg(error);
                 return result;
             }
         } else {
             if (!loadTxtMeshPair(verticesPath, facesPath, &result.vertices, &result.indices, &error)) {
                 result.error = QString("Failed to parse TXT mesh.\n\n%1").arg(error);
                 return result;
             }
         }

         if (result.vertices.isEmpty() || result.indices.isEmpty()) {
             result.error = "Mesh file contains no renderable triangles.";
             return result;
         }

         // Imported meshes can include thin detached/border slab artifacts.
         keepLargestTriangleComponent(&result.vertices, &result.indices);
         removeBorderPlaneSlab(&result.vertices, &result.indices);

         result.ok = true;
         return result;
     }));
 }


void MainWindow::syncThresholdControls() {
    threshold16SpinBox->setVisible(true);
    threshold16AutoButton->setVisible(true);
    thresholdLabel->setText("Threshold:");
}

// Threshold selection logic: manual updates own the active threshold directly.
void MainWindow::onThreshold16Changed(int value) {
    currentThreshold16 = value;
    currentThreshold = qBound(0.0, double(value) / 65535.0, 1.0);
    thresholdLabel->setText("Threshold:");
}

// Threshold selection logic: Otsu only suggests the threshold value.
void MainWindow::onThreshold16Auto() {
    if (isOtsuRunning) {
        statusBar()->showMessage("Otsu threshold is already computing...", 2500);
        return;
    }

    if (!hasSegmentation16Data || segmentationSlices16.isEmpty()) {
        QMessageBox::information(
            this,
            "Auto Threshold Unavailable",
            "No images are loaded, so an automatic threshold cannot be calculated.\n\n"
            "Please load a dataset first, then click Auto again.");
        return;
    }

    isOtsuRunning = true;
    if (generate3DAct) {
        generate3DAct->setEnabled(false);
    }
    threshold16AutoButton->setEnabled(false);
    
    // Use QUICK threshold computation (instant if warm-start available)
    int warmStart = imageProcessor.loadLastGoodThreshold();
    if (warmStart > 0) {
        // Warm-start available: instant feedback
        statusBar()->showMessage("Using previous good threshold as warm-start...", 1500);
        threshold16SpinBox->setValue(warmStart);
        currentThreshold = qBound(0.0, double(warmStart) / 65535.0, 1.0);
        isOtsuRunning = false;
        if (generate3DAct) {
            generate3DAct->setEnabled(true);
        }
        threshold16AutoButton->setEnabled(true);
        thresholdLabel->setText(QString("Threshold (16-bit): %1 (warm-start)").arg(warmStart));
        return;
    }

    // No warm-start: compute Otsu in background (much faster than adaptive processing)
    statusBar()->showMessage("Computing global Otsu threshold...", 2500);
    const QVector<cv::Mat> slicesCopy = segmentationSlices16;
    
    otsuWatcher.setFuture(QtConcurrent::run([slicesCopy]() {
        return ImageProcessor::suggestThresholdQuick(slicesCopy, 0);
    }));
}

void MainWindow::onOtsuComputationFinished() {
    const int suggestedThreshold = otsuWatcher.future().result();
    const int boundedThreshold = qBound(0, suggestedThreshold, 65535);

    threshold16SpinBox->setValue(boundedThreshold);
    currentThreshold = qBound(0.0, double(boundedThreshold) / 65535.0, 1.0);
    isOtsuRunning = false;
    if (generate3DAct) {
        generate3DAct->setEnabled(true);
    }
    threshold16AutoButton->setEnabled(true);

    statusBar()->showMessage(
        QString("Threshold suggested: %1. Click Generate 3D to rebuild the old thresholded mesh path.")
            .arg(boundedThreshold),
        5000);
}

// ============================================================================
// Material Visualization and Segmentation
// ============================================================================

void MainWindow::addMaterialVisualizationControls() {
    // Checkbox is created in createToolbar() as the second row of the segmentation widget.
    // Only wire up the signal connection here.
    if (!materialColorsCheckBox) return;
    connect(materialColorsCheckBox, &QCheckBox::toggled, this, &MainWindow::onMaterialColorsToggled);
}

void MainWindow::onMaterialColorsToggled(bool enabled) {
    // Simply toggle overlay visibility - no mesh recalculation or upload
    materialColorsEnabled = enabled;
    glWidget->setMaterialColorsEnabled(enabled);
    glWidget->update();
    if (materialLegendWidget)
        materialLegendWidget->setVisible(enabled);
    // Overlay legend inside the 3D view
    if (overlayLegendWidget && vizTab) {
        if (enabled) {
            overlayLegendWidget->adjustSize();
            const int margin = 10;
            overlayLegendWidget->move(vizTab->width() - overlayLegendWidget->width() - margin, margin);
            overlayLegendWidget->raise();
        }
        overlayLegendWidget->setVisible(enabled);
    }
    qDebug() << "Material Colors overlay:" << (enabled ? "ON" : "OFF");
}

MarchingCubes::Mesh MainWindow::classifyMeshByMaterial(const MarchingCubes::Mesh& inputMesh,
                                                       const VolumeData16& volume16) {
    MarchingCubes::Mesh outputMesh = inputMesh;
    outputMesh.vertexColors.clear();
    outputMesh.vertexMaterials.clear();

    int ceramicThresholdLocal = 200;
    int boneThresholdLocal = 100;
    estimateMaterialThresholds(volume16, ceramicThresholdLocal, boneThresholdLocal);

    QColor ceramicColor(50, 130, 255);
    const int eventStride = qMax(100000, inputMesh.vertices.size() / 24);

    if (volume16.isEmpty() || volume16[0].isEmpty() || volume16[0][0].isEmpty()) {
        return outputMesh;
    }
    const int volDepth  = volume16.size();
    const int volHeight = volume16[0].size();
    const int volWidth  = volume16[0][0].size();
    const int borderGuard = 20;

    const float safeSpacingX = qMax(0.0001f, voxelSpacingX);
    const float safeSpacingY = qMax(0.0001f, voxelSpacingY);
    const float safeSpacingZ = qMax(0.0001f, voxelSpacingZ);

    // Safe sampler: border-guarded, returns intensity in 0-255 float, or -1 for invalid.
    auto sampleF = [&](int px, int py, int pz) -> float {
        if (px < borderGuard || px >= volWidth  - borderGuard ||
            py < borderGuard || py >= volHeight - borderGuard ||
            pz < borderGuard || pz >= volDepth  - borderGuard)
            return -1.0f;
        return static_cast<float>(volume16[pz][py][px]) / 257.0f;
    };

    // =========================================================================
    // Step 1: Precompute per-vertex area-weighted normals from face indices.
    //
    // MC typically produces outward normals.  We do NOT assume a direction —
    // instead we sample BOTH ±normal and take the maximum Variance-to-Mean Ratio (VMR).  This makes the
    // classifier robust to whichever way the MC normal happens to point.
    // =========================================================================
    qDebug() << "  [classify] Precomputing normals for"
             << inputMesh.vertices.size() << "vertices /"
             << (inputMesh.indices.size() / 3) << "faces...";
    QVector<QVector3D> vertexNormals(inputMesh.vertices.size(), QVector3D(0.0f, 0.0f, 0.0f));
    {
        const int numIdx = inputMesh.indices.size();
        const int nv     = inputMesh.vertices.size();
        for (int fi = 0; fi + 2 < numIdx; fi += 3) {
            const unsigned int i0 = inputMesh.indices[fi];
            const unsigned int i1 = inputMesh.indices[fi + 1];
            const unsigned int i2 = inputMesh.indices[fi + 2];
            if ((int)i0 >= nv || (int)i1 >= nv || (int)i2 >= nv) continue;
            const QVector3D e1 = inputMesh.vertices[i1] - inputMesh.vertices[i0];
            const QVector3D e2 = inputMesh.vertices[i2] - inputMesh.vertices[i0];
            const QVector3D fn = QVector3D::crossProduct(e1, e2);  // area-weighted
            vertexNormals[i0] += fn;
            vertexNormals[i1] += fn;
            vertexNormals[i2] += fn;
            if (fi % 3000000 == 0 && fi > 0)
                QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
        for (QVector3D& n : vertexNormals) {
            const float len = n.length();
            if (len > 1e-6f) n /= len;
        }
    }
    qDebug() << "  [classify] Normals done.";

    // =========================================================================
    // Step 2: VMR + Periodicity feature-based classification
    //
    // PRIMARY DISCRIMINATOR: Variance-to-mean ratio (VMR = σ²/(μ+1))
    //
    // We sample N steps of stepMM along +normal and -normal independently and
    // compute VMR for each direction separately, then take max(VMR+, VMR-).
    //
    // WHY separate directions (not bidirectional):
    //   Bidirectional: both bone-surface AND porous-scaffold show a bright↔dark
    //   transition, giving spuriously high VMR for cortical bone.
    //
    //   Unidirectional: cortical bone going inward is UNIFORM (dense→dense→dense),
    //   VMR ≈ 0-10.  Porous scaffold going inward alternates wall/pore/wall,
    //   VMR ≈ 40-150.  Clean separation without needing to know normal direction.
    //
    // WHY max(VMR+, VMR-):
    //   We do not know whether the MC normal points into solid or into air.
    //   max() picks whichever direction actually crosses the porous lattice.
    //
    // SECONDARY: Periodicity (direction reversals in the VMR-winning sample array)
    //   Porous scaffold:  2-4 reversals in 5 samples (wall→pore→wall→pore)
    //   Cortical bone:    0-1 reversals in 5 samples (gradual thickness decrease)
    //   Helps reject thick trabecular bone which can have moderate VMR.
    //
    // TERTIARY: Intensity gate — vertex must be at or above bone threshold.
    //
    // WHY the old weights failed (images 1-3 showed scattered false positives):
    //   Cut-plane surface vertex: sampling crosses bone→air in ONE step.
    //     VMR ≈ 100 (huge), periodicity = 0 reversals.
    //     Old score: 0.65*1.0 + 0.20*1.0 + 0.15*0.0 = 0.85 → WRONG ceramic label.
    //   Porous scaffold vertex: wall→pore→wall in 3 steps.
    //     Observed from log: VMR 12-20, periodicity = 1.00 reversal.
    //     Old VMR_NORMALIZE=60 made vmrNorm=12/60=0.20 — classifier was nearly blind.
    //     Fix: VMR_NORMALIZE calibrated to 15 so VMR=15 → vmrNorm=1.0.
    //
    // Hard rules (before scoring):
    //   1. centerVal >= ceramicThreshold * 0.80  (intensity gate: blocks soft tissue)
    //   2. periodicityScore > 0.0  (>= 1 reversal REQUIRED: blocks single-edge cut-planes)
    //      Cut-plane vertex: bone→air in 1 step → outCnt<2 in -dir → VMR-=0 → rev=0 → REJECTED
    //
    // Weights: 0.15 * intensity + 0.45 * vmrNorm + 0.40 * periodicity >= 0.55
    //
    // Re-calibrated for actual log values (VMR 12-20, periodicity 1.0):
    //   Cortical bone uniform (VMR=2,  rev=0): 0.15*0.43 + 0.45*0.13 + 0.40*0 = 0.12 → NO
    //   Trabeculae           (VMR=8,  rev=1): 0.15*0.43 + 0.45*0.53 + 0.40*0.5 = 0.50 → NO
    //   Porous scaffold      (VMR=15, rev=2): 0.15*0.43 + 0.45*1.00 + 0.40*1.0 = 0.91 → YES
    //   Scaffold (min case)  (VMR=12, rev=1): 0.15*0.43 + 0.45*0.80 + 0.40*0.5 = 0.62 → YES
    // =========================================================================
    const float stepMM        = 0.5f;   // mm per step: 0.5mm ≈ 1.7 voxels (X/Y) / 1.0 voxel (Z)
    const int   numSteps      = 8;      // samples at 0.5..4.0mm — covers full pore-wall-pore cycle
    // *** Calibrated from log data: scaffold VMR observed at 12-20, NOT 40-60 as assumed.
    // VMR_NORMALIZE=60 caused vmrNorm≈0.2 for all scaffold → only 258/38M classified.
    // Set to 15 so that observed scaffold VMR saturates the feature: VMR=15 → vmrNorm=1.0.
    const float VMR_NORMALIZE   = 15.0f;
    const float VMR_MAX_GATE    = 50.0f;   // Real scaffold VMR = 12-20; artifact edges produce 100-250.
                                            // Anything above 50 is a scanner saturation artefact.
    const float INTENSITY_MAX   = 240.0f;  // Near-saturated voxels (253/255) are metal/scanner artefacts,
                                            // not scaffold material.  Real scaffold sits at 120-160.
    const float SCORE_THRESHOLD = 0.55f;   // Restored: per-direction scoring makes this tight gate safe again.

    // Compute VMR along a single direction; fills outSamples[0..outCnt-1].
    auto computeVMR = [&](const QVector3D& origin, const QVector3D& dir,
                          float step, int steps,
                          float* outSamples, int& outCnt) -> float {
        outCnt = 0;
        float sum = 0.0f;
        for (int s = 1; s <= steps; ++s) {
            const float dist = s * step;
            const int sx = static_cast<int>(qRound((origin.x() + dir.x() * dist) / safeSpacingX));
            const int sy = static_cast<int>(qRound((origin.y() + dir.y() * dist) / safeSpacingY));
            const int sz = static_cast<int>(qRound((origin.z() + dir.z() * dist) / safeSpacingZ));
            const float v = sampleF(sx, sy, sz);
            if (v >= 0.0f) { outSamples[outCnt++] = v; sum += v; }
        }
        if (outCnt < 2) return 0.0f;
        const float mean = sum / outCnt;
        float var = 0.0f;
        for (int i = 0; i < outCnt; ++i) { const float d = outSamples[i] - mean; var += d * d; }
        var /= outCnt;
        return var / (mean + 1.0f);  // VMR: σ²/(μ+1)
    };

    // Count direction reversals in an array — periodicity measure.
    // Hysteresis ±3.0: suppresses voxel-quantization jitter while detecting pore
    // transitions (scaffold wall↔pore spans 20-80 intensity units in this CT).
    auto countReversals = [](const float* arr, int cnt) -> int {
        int rev = 0;
        for (int i = 1; i < cnt - 1; ++i) {
            const bool up0 = arr[i]   > arr[i-1] + 3.0f;
            const bool dn0 = arr[i]   < arr[i-1] - 3.0f;
            const bool up1 = arr[i+1] > arr[i]   + 3.0f;
            const bool dn1 = arr[i+1] < arr[i]   - 3.0f;
            if ((up0 && dn1) || (dn0 && up1)) ++rev;
        }
        return rev;
    };

    int ceramicCount       = 0;
    int borderSkipped      = 0;
    int deepCeramicPrinted = 0;
    const int deepZMin     = volDepth / 10;

    qDebug().noquote() << "-----[ Material Classification Start ]-----";
    qDebug() << "  Algorithm: max(VMR+, VMR-) + Periodicity via ±normal sampling";
    qDebug() << "  Volume (W x H x D):" << volWidth << "x" << volHeight << "x" << volDepth;
    qDebug() << "  Thresholds (0-255): ceramic=" << ceramicThresholdLocal
             << "  bone=" << boneThresholdLocal;
    qDebug() << "  Spacing X:" << safeSpacingX << " Y:" << safeSpacingY << " Z:" << safeSpacingZ;
    qDebug() << "  Vertices:" << inputMesh.vertices.size();
    qDebug() << "  stepMM:" << stepMM << "  numSteps:" << numSteps
             << "  VMR_NORMALIZE:" << VMR_NORMALIZE
             << "  SCORE_THRESHOLD:" << SCORE_THRESHOLD;
    qDebug() << "  Calibrated from log: scaffold VMR=12-20, VMR_NORMALIZE=15 so vmrNorm saturates at VMR=15";
    qDebug() << "  Hard rules: (1) centerVal >= Otsu*0.70   (2) periodicityScore > 0";

    // Intensity gate is now applied per-vertex inside the classify loop,
    // using ceramicThresholdLocal * 0.65 (see inside the loop below).

    // Per-ray scratch buffer for computeVMR (numSteps=8, +1 margin)
    float rayBuf[9];
    int   rayCnt = 0;
    const int nVerts = inputMesh.vertices.size();

    // Gate-failure diagnostics (logged after loop to diagnose future datasets).
    int failArtifactCnt  = 0;
    int failIntensityCnt = 0;
    int failReversalsCnt = 0;  // kept for log format compatibility; always 0 (gate removed)
    int failScoreCnt     = 0;

    for (int vi = 0; vi < nVerts; ++vi) {
        const QVector3D& vertex = inputMesh.vertices[vi];

        int vx = static_cast<int>(qRound(vertex.x() / safeSpacingX));
        int vy = static_cast<int>(qRound(vertex.y() / safeSpacingY));
        int vz = static_cast<int>(qRound(vertex.z() / safeSpacingZ));
        vx = qBound(0, vx, volWidth  - 1);
        vy = qBound(0, vy, volHeight - 1);
        vz = qBound(0, vz, volDepth  - 1);

        MaterialType material = MaterialType::Background;
        QColor color = ceramicColor;
        color.setAlphaF(0.0f);

        const bool nearBorder = (vx < borderGuard || vx >= volWidth  - borderGuard ||
                                 vy < borderGuard || vy >= volHeight - borderGuard ||
                                 vz < borderGuard || vz >= volDepth  - borderGuard);
        if (nearBorder) {
            ++borderSkipped;
        } else {
            // 6-connected max-neighbor intensity.
            // WHY: MC vertices sit on the iso-surface boundary (~threshold intensity).
            // qRound() snaps the vertex to either the air side (low, e.g. 50) or the
            // material side (high, e.g. 140). Scaffold pore-surface vertices routinely
            // snap to the air side, so sampleF(vx,vy,vz) returns sub-threshold values
            // and ALL scaffold vertices fail the intensity gate  (Dataset 3: 56.9M blocked).
            // Taking the max of 6 face-neighbours guarantees we see the adjacent scaffold
            // wall voxel (intensity ≥ ceramicThreshold), correctly identifying the vertex
            // as 'near scaffold material' regardless of which side the vertex snapped to.
            float centerVal = sampleF(vx, vy, vz);
            const float cnx1 = sampleF(vx+1, vy,   vz  ); if (cnx1 > centerVal) centerVal = cnx1;
            const float cnx2 = sampleF(vx-1, vy,   vz  ); if (cnx2 > centerVal) centerVal = cnx2;
            const float cny1 = sampleF(vx,   vy+1, vz  ); if (cny1 > centerVal) centerVal = cny1;
            const float cny2 = sampleF(vx,   vy-1, vz  ); if (cny2 > centerVal) centerVal = cny2;
            const float cnz1 = sampleF(vx,   vy,   vz+1); if (cnz1 > centerVal) centerVal = cnz1;
            const float cnz2 = sampleF(vx,   vy,   vz-1); if (cnz2 > centerVal) centerVal = cnz2;
            // (sampleF returns -1 for border/OOB, which is correctly ignored by the max)

            const QVector3D& nrm   = vertexNormals[vi];
            const bool validNormal = (nrm.lengthSquared() > 1e-6f);

            // ---------------------------------------------------------------
            // Omni-directional VMR sampling: ±normal + ±X + ±Y + ±Z = up to 8 dirs.
            //
            // WHY: scaffold surface normals typically point radially outward from the
            // ring, perpendicular to the pore axis. Sampling only along ±normal means
            // most scaffold vertices cross bone→air (not wall→pore→wall), producing
            // VMR with 0 reversals and getting rejected. Adding axis-aligned directions
            // guarantees at least one direction crosses the pore-wall periodicity,
            // regardless of how the MC normal is oriented.
            // ---------------------------------------------------------------
            QVector3D dirs[8];
            int ndirs = 0;
            if (validNormal) { dirs[ndirs++] =  nrm; dirs[ndirs++] = -nrm; }
            dirs[ndirs++] = QVector3D( 1, 0, 0);
            dirs[ndirs++] = QVector3D(-1, 0, 0);
            dirs[ndirs++] = QVector3D( 0, 1, 0);
            dirs[ndirs++] = QVector3D( 0,-1, 0);
            dirs[ndirs++] = QVector3D( 0, 0, 1);
            dirs[ndirs++] = QVector3D( 0, 0,-1);

            // Per-direction scoring: compute score for each direction independently,
            // then take the BEST-SCORING direction.  This ensures VMR and periodicity
            // used in the final decision come from the SAME physical direction.
            //
            // WHY independent tracking was wrong:
            //   bestVMR may come from the bone/air edge (VMR=40, rev=0)
            //   while bestPeriodicity may come from a noisy direction (VMR=3, rev=1).
            //   Combined: vmrNorm=1.0, periodicity=0.5 → score=0.70 → false ceramic.
            // WHY per-direction scoring is correct:
            //   Bone/air edge direction: score = 0.15*I + 0.45*1.0 + 0.40*0 = 0.52 < 0.55 → FAIL.
            //   Real scaffold direction: score = 0.15*I + 0.45*VMR + 0.40*per > 0.55 → PASS.
            //   VMR_MAX_GATE applied per-direction: artifact directions are skipped.
            const float intensityNorm = (centerVal >= 0.0f) ? centerVal / 255.0f : 0.0f;
            float bestScore       = 0.0f;
            float bestVMR         = 0.0f;   // for logging
            float periodicityScore = 0.0f;  // for logging

            for (int d = 0; d < ndirs; ++d) {
                const float vmrD = computeVMR(vertex, dirs[d], stepMM, numSteps, rayBuf, rayCnt);
                if (vmrD > VMR_MAX_GATE) continue; // artifact-level direction: skip
                const int rev = countReversals(rayBuf, rayCnt);
                const float perD     = qMin(float(rev) / 2.0f, 1.0f);
                const float vmrNormD = qMin(vmrD / VMR_NORMALIZE, 1.0f);
                const float scoreD   = 0.15f * intensityNorm + 0.45f * vmrNormD + 0.40f * perD;
                if (scoreD > bestScore) {
                    bestScore      = scoreD;
                    bestVMR        = vmrD;
                    periodicityScore = perD;
                }
            }

            // Near-saturated intensity = scanner/metal artefact (never real scaffold).
            const bool isArtifact = (centerVal > INTENSITY_MAX);

            if (isArtifact) {
                ++failArtifactCnt;
            } else {
                // Hard gate: max-neighbor intensity >= 85% of ceramic threshold.
                // With 6-connected max-neighbor, scaffold wall voxels (>= ceramicThreshold)
                // are always adjacent to pore surface vertices, so this gate is strict
                // without missing real scaffold.
                // Dataset 1: ceramicThreshold=130 → gate=110.5. Dataset 3: 143 → gate=121.6.
                const float intensityGate = ceramicThresholdLocal * 0.85f;
                const bool passIntensity  = (centerVal >= intensityGate);

                // No separate reversals gate: per-direction scoring handles it naturally.
                // A direction with 0 reversals scores ≤0.52 → fails threshold 0.55.

                const bool isCeramic = passIntensity && (bestScore >= SCORE_THRESHOLD);

                if (!passIntensity) ++failIntensityCnt;
                else if (!isCeramic) ++failScoreCnt;

                if (isCeramic) {
                    material = MaterialType::Ceramic;
                    const float softBlend = qBound(0.20f, 0.15f + 0.60f * bestScore, 0.75f);
                    color.setAlphaF(softBlend);
                    ++ceramicCount;

                    if (ceramicCount <= 10) {
                        qDebug() << "  [Ceramic hit" << ceramicCount << "]"
                                 << " voxel(" << vx << "," << vy << "," << vz << ")"
                                 << " intensity:" << qRound(centerVal)
                                 << " bestVMR:" << QString::number(bestVMR, 'f', 1)
                                 << " periodicity:" << QString::number(periodicityScore, 'f', 2)
                                 << " score:" << QString::number(bestScore, 'f', 3);
                    }
                    if (deepCeramicPrinted < 5 && vz > deepZMin) {
                        ++deepCeramicPrinted;
                        qDebug() << "  [Deep ceramic hit" << deepCeramicPrinted << "]"
                                 << " voxel(" << vx << "," << vy << "," << vz << ")"
                                 << " bestVMR:" << QString::number(bestVMR, 'f', 1)
                                 << " periodicity:" << QString::number(periodicityScore, 'f', 2)
                                 << " score:" << QString::number(bestScore, 'f', 3);
                    }
                }
            }
        }

        outputMesh.vertexMaterials.append(material);
        outputMesh.vertexColors.append(color);

        if ((vi + 1) % eventStride == 0) {
            qDebug() << "  [classify] progress:" << (vi + 1) << "/" << nVerts
                     << "  ceramic so far:" << ceramicCount;
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
        }
    }

    // =========================================================================
    // Post-classification: mesh-topology dilation (region growing on the mesh graph)
    //
    // =========================================================================
    // STEP 3: Volumetric 3D dilation — fill the entire scaffold region.
    //
    // Mesh-topology BFS only grows one edge per round; with sparse seeds it
    // converges long before covering the full scaffold surface.  Instead:
    //   3a) Project seed vertices into a 3-D voxel bitmap (seedVoxMap).
    //   3b) Apply a separable prefix-sum box max-filter (radius DILATE_R voxels)
    //       in X, Y, Z — O(W*H*D) total, no inner loops over radius.
    //   3c) Promote every non-ceramic mesh vertex whose voxel falls inside the
    //       dilated region to Ceramic.
    // This bridges all intra-scaffold gaps regardless of mesh topology and
    // covers every triangle on the scaffold surface uniformly.
    // =========================================================================
    const int mapVol = volWidth * volHeight * volDepth;
    const int seedsBefore = ceramicCount;

    // --- 3a: mark seed voxels ---
    std::vector<uint8_t> seedVoxMap(mapVol, 0);
    for (int vi2 = 0; vi2 < nVerts; ++vi2) {
        if (outputMesh.vertexMaterials[vi2] != MaterialType::Ceramic) continue;
        const QVector3D& sv = inputMesh.vertices[vi2];
        const int sx = qBound(0, qRound(sv.x() / safeSpacingX), volWidth  - 1);
        const int sy = qBound(0, qRound(sv.y() / safeSpacingY), volHeight - 1);
        const int sz = qBound(0, qRound(sv.z() / safeSpacingZ), volDepth  - 1);
        seedVoxMap[sz * volHeight * volWidth + sy * volWidth + sx] = 1;
    }
    qDebug() << "  [gate stats] artifact:" << failArtifactCnt
             << " intensity:" << failIntensityCnt
             << " reversals:" << failReversalsCnt
             << " score:" << failScoreCnt
             << " passed:" << ceramicCount;

    // --- 3b: adaptive DILATE_R based on seed density ---
    // Dense seeds need small dilation (seeds already cover scaffold surface well).
    // Sparse seeds need large dilation to bridge inter-seed gaps.
    // Calibrated so both datasets produce 10-40% scaffold coverage:
    //   Dataset 1: seedRate=0.72% → DILATE_R=4  (~1.2 mm at 0.3mm spacing)
    //   Dataset 3: seedRate=0.007% → DILATE_R=20 (~6 mm at 0.3mm spacing)
    const double seedRate = (nVerts > 0) ? double(seedsBefore) / nVerts : 0.0;
    int DILATE_R;
    if      (seedRate > 0.005) DILATE_R = 4;   // > 0.5%: dense seeds, small spread (~1.2 mm)
    else if (seedRate > 0.002) DILATE_R = 8;   // > 0.2%: moderate seeds (~2.4 mm)
    else if (seedRate > 0.001) DILATE_R = 14;  // > 0.1%: sparse seeds (~4.2 mm)
    else if (seedRate > 0.0001) DILATE_R = 20; // > 0.01%: very sparse (~6.0 mm)
    else                        DILATE_R = 25; // < 0.01%: isolated seeds (~7.5 mm)
    qDebug() << "  [vol-dilation] Adaptive DILATE_R:" << DILATE_R
             << " (seed rate:" << QString::number(seedRate * 100, 'f', 4) << "%)  Starting 3-D dilation...";
    const int maxDim   = std::max({volWidth, volHeight, volDepth});
    std::vector<uint8_t> dilWorkMap(mapVol, 0);  // intermediate
    std::vector<uint8_t> dilVoxMap (mapVol, 0);  // final dilated bitmap
    std::vector<int>     prefixSum (maxDim + 1, 0);

    // Pass 1 — dilate along X: seedVoxMap → dilWorkMap
    for (int z = 0; z < volDepth; ++z) {
        for (int y = 0; y < volHeight; ++y) {
            const int base = z * volHeight * volWidth + y * volWidth;
            prefixSum[0] = 0;
            for (int x = 0; x < volWidth; ++x)
                prefixSum[x + 1] = prefixSum[x] + (seedVoxMap[base + x] ? 1 : 0);
            for (int x = 0; x < volWidth; ++x) {
                const int lo = std::max(0, x - DILATE_R);
                const int hi = std::min(volWidth, x + DILATE_R + 1);
                if (prefixSum[hi] - prefixSum[lo] > 0)
                    dilWorkMap[base + x] = 1;
            }
        }
        if (z % 60 == 0)
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    // Pass 2 — dilate along Y: dilWorkMap → dilVoxMap
    for (int z = 0; z < volDepth; ++z) {
        for (int x = 0; x < volWidth; ++x) {
            prefixSum[0] = 0;
            for (int y = 0; y < volHeight; ++y)
                prefixSum[y + 1] = prefixSum[y]
                    + (dilWorkMap[z * volHeight * volWidth + y * volWidth + x] ? 1 : 0);
            for (int y = 0; y < volHeight; ++y) {
                const int lo = std::max(0, y - DILATE_R);
                const int hi = std::min(volHeight, y + DILATE_R + 1);
                if (prefixSum[hi] - prefixSum[lo] > 0)
                    dilVoxMap[z * volHeight * volWidth + y * volWidth + x] = 1;
            }
        }
        if (z % 60 == 0)
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }

    // Pass 3 — dilate along Z: dilVoxMap → dilWorkMap (reuse buffer)
    std::fill(dilWorkMap.begin(), dilWorkMap.end(), 0);
    for (int y = 0; y < volHeight; ++y) {
        for (int x = 0; x < volWidth; ++x) {
            prefixSum[0] = 0;
            for (int z = 0; z < volDepth; ++z)
                prefixSum[z + 1] = prefixSum[z]
                    + (dilVoxMap[z * volHeight * volWidth + y * volWidth + x] ? 1 : 0);
            for (int z = 0; z < volDepth; ++z) {
                const int lo = std::max(0, z - DILATE_R);
                const int hi = std::min(volDepth, z + DILATE_R + 1);
                if (prefixSum[hi] - prefixSum[lo] > 0)
                    dilWorkMap[z * volHeight * volWidth + y * volWidth + x] = 1;
            }
        }
        if (y % 100 == 0)
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }
    // dilWorkMap now holds the fully-dilated 3-D ceramic region bitmap.
    qDebug() << "  [vol-dilation] 3-D dilation complete (radius=" << DILATE_R << "voxels).";

    // --- 3c: promote non-ceramic vertices inside the dilated region ---
    int volPromoted = 0;
    for (int vi2 = 0; vi2 < nVerts; ++vi2) {
        if (outputMesh.vertexMaterials[vi2] == MaterialType::Ceramic) continue;
        const QVector3D& pv = inputMesh.vertices[vi2];
        const int px = qBound(0, qRound(pv.x() / safeSpacingX), volWidth  - 1);
        const int py = qBound(0, qRound(pv.y() / safeSpacingY), volHeight - 1);
        const int pz = qBound(0, qRound(pv.z() / safeSpacingZ), volDepth  - 1);
        if (!dilWorkMap[pz * volHeight * volWidth + py * volWidth + px]) continue;
        // Skip pure air (sampleF normalises 16-bit → 0-255; value < 5 is background).
        const float pval = sampleF(px, py, pz);
        if (pval < 5.0f) continue;
        outputMesh.vertexMaterials[vi2] = MaterialType::Ceramic;
        outputMesh.vertexColors[vi2].setAlphaF(0.55f);  // solid, opaque blue
        ++volPromoted;
        ++ceramicCount;
        if (vi2 % 4000000 == 0)
            QCoreApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
    }
    qDebug() << "  [vol-dilation] Promoted" << volPromoted << "vertices via volumetric dilation.";

    qDebug().noquote() << "-----[ Ceramic Bone Found ]-----";
    qDebug() << "  Ceramic seeds (VMR pass):   " << seedsBefore << "/" << nVerts;
    qDebug() << "  Ceramic after vol-dilation: " << ceramicCount << "/" << nVerts;
    qDebug() << "  Skipped (near border):      " << borderSkipped;
    if (nVerts > 0) {
        qDebug() << QString("  Ceramic coverage: %1%")
                        .arg(100.0 * ceramicCount / double(nVerts), 0, 'f', 2);
    }
    qDebug().noquote() << "-----[ Material Classification End ]-----";

    return outputMesh;
}

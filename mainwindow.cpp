#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "GLWidget.h"
#include <QFile>
#include <QThread>
#include <QListView>
#include <QTreeView>
#include <QAbstractItemView>
#include <QProgressDialog>
#include <QScrollArea>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include <QPushButton>
#include <QMouseEvent>
#include <QFileInfo>
#include <QEventLoop>
#include <QRegularExpression>
#include <QComboBox>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>
#include <cstring>
#include <cmath>
#include <limits>

namespace {
struct LoadedSliceResult {
    int index = -1;
    bool valid = false;
    QImage original;
    QImage processed;
    cv::Mat segmentation16;
    bool hasNative16 = false;
};

int extractLastNumber(const QString& name, bool* ok = nullptr) {
    static const QRegularExpression numberRegex("(\\d+)");
    QRegularExpressionMatchIterator it = numberRegex.globalMatch(name);
    int lastNumber = -1;

    while (it.hasNext()) {
        const QRegularExpressionMatch match = it.next();
        lastNumber = match.captured(1).toInt();
    }

    if (ok) {
        *ok = (lastNumber >= 0);
    }
    return lastNumber;
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

cv::Mat cropSegmentationSlice16(const cv::Mat& src) {
    if (src.empty()) {
        return cv::Mat();
    }

    const int x = 45;
    const int y = 48;
    const int w = 1417;
    const int h = 537;

    const int x0 = qBound(0, x, src.cols);
    const int y0 = qBound(0, y, src.rows);
    const int x1 = qBound(0, x + w, src.cols);
    const int y1 = qBound(0, y + h, src.rows);

    if (x1 <= x0 || y1 <= y0) {
        return src.clone();
    }

    return src(cv::Rect(x0, y0, x1 - x0, y1 - y0)).clone();
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

cv::Mat cleanupBinaryMask16(const cv::Mat& inputMask, bool sparseDepth, bool denseDepth) {
    if (inputMask.empty()) {
        return cv::Mat();
    }

    cv::Mat mask = inputMask.clone();
    const int rows = mask.rows;
    const int cols = mask.cols;
    const int totalPixels = rows * cols;

    // Hard suppress image borders to avoid slab artifacts from scanner frame/background.
    const int borderY = qMax(2, rows / 35);
    const int borderX = qMax(2, cols / 35);
    mask.rowRange(0, borderY).setTo(0);
    mask.rowRange(rows - borderY, rows).setTo(0);
    mask.colRange(0, borderX).setTo(0);
    mask.colRange(cols - borderX, cols).setTo(0);

    const cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    if (!sparseDepth) {
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

    struct Candidate {
        int label = 0;
        double score = 0.0;
    };

    const int minComponentArea = sparseDepth
        ? qMax(12, totalPixels / 25000)
        : (denseDepth ? qMax(220, totalPixels / 2200) : qMax(80, totalPixels / 4000));

    QVector<Candidate> candidates;
    candidates.reserve(componentCount - 1);

    for (int label = 1; label < componentCount; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minComponentArea) {
            continue;
        }

        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int compWidth = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int compHeight = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int centerX = left + compWidth / 2;
        const int centerY = top + compHeight / 2;

        const bool touchesLeft = (left <= 1);
        const bool touchesRight = (left + compWidth >= cols - 1);
        const bool touchesTop = (top <= 1);
        const bool touchesBottom = (top + compHeight >= rows - 1);
        const int borderTouchCount = int(touchesLeft) + int(touchesRight) + int(touchesTop) + int(touchesBottom);

        if (borderTouchCount >= 3 && area > (totalPixels / 12)) {
            continue;
        }

        const double areaScore = double(area) / double(totalPixels);
        const double centerXScore = 1.0 - qAbs(double(centerX) - (double(cols) * 0.5)) / (double(cols) * 0.5 + 1e-6);
        const double centerYScore = 1.0 - qAbs(double(centerY) - (double(rows) * 0.5)) / (double(rows) * 0.5 + 1e-6);
        const double score = (0.68 * areaScore) + (0.18 * centerXScore) + (0.14 * centerYScore);
        candidates.append({label, score});
    }

    cv::Mat filteredMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    if (!candidates.isEmpty()) {
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        const int targetKeepCount = sparseDepth ? 4 : (denseDepth ? 1 : 2);
        const int keepCount = qMin(targetKeepCount, candidates.size());
        for (int i = 0; i < keepCount; ++i) {
            filteredMask.setTo(255, labels == candidates[i].label);
        }
    } else {
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
    }

    const cv::Size closeSize = sparseDepth ? cv::Size(11, 5) : (denseDepth ? cv::Size(19, 9) : cv::Size(13, 7));
    const cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, closeSize);
    cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_CLOSE, kernelClose);
    cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_OPEN, kernelSmall);
    return filteredMask;
}

bool detectForegroundIsDark(const QVector<cv::Mat>& slices16) {
    if (slices16.isEmpty()) {
        return false;
    }

    double centerAccum = 0.0;
    double borderAccum = 0.0;
    qint64 centerCount = 0;
    qint64 borderCount = 0;

    const int zStep = qMax(1, slices16.size() / 24);
    for (int z = 0; z < slices16.size(); z += zStep) {
        cv::Mat u16;
        if (slices16[z].type() == CV_16UC1) {
            u16 = slices16[z];
        } else {
            slices16[z].convertTo(u16, CV_16UC1);
        }

        if (u16.empty()) {
            continue;
        }

        const int rows = u16.rows;
        const int cols = u16.cols;
        const int cx0 = cols / 5;
        const int cx1 = cols - cx0;
        const int cy0 = rows / 5;
        const int cy1 = rows - cy0;

        for (int y = 0; y < rows; ++y) {
            const quint16* row = reinterpret_cast<const quint16*>(u16.ptr(y));
            for (int x = 0; x < cols; ++x) {
                const bool inCenter = (x >= cx0 && x < cx1 && y >= cy0 && y < cy1);
                if (inCenter) {
                    centerAccum += row[x];
                    ++centerCount;
                } else {
                    borderAccum += row[x];
                    ++borderCount;
                }
            }
        }
    }

    if (centerCount == 0 || borderCount == 0) {
        return false;
    }

    const double centerMean = centerAccum / double(centerCount);
    const double borderMean = borderAccum / double(borderCount);
    return (centerMean + 300.0) < borderMean;
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
    QWidget* vizTab = new QWidget();
    QVBoxLayout* vizLayout = new QVBoxLayout(vizTab);
    vizLayout->addWidget(glWidget);
    mainTabs->addTab(vizTab, "3D Visualization");

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
    QToolBar* toolBar = addToolBar("Main Toolbar");
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
    thresholdLabel = new QLabel("Threshold (16-bit): 28180", this);
    thresholdLabel->setMinimumWidth(90);

    threshold16SpinBox = new QSpinBox(this);
    threshold16SpinBox->setRange(0, 65535);
    threshold16SpinBox->setSingleStep(64);
    threshold16SpinBox->setValue(currentThreshold16);
    threshold16SpinBox->setMinimumWidth(90);
    threshold16SpinBox->setMaximumWidth(110);

    threshold16AutoButton = new QPushButton("Auto", this);
    threshold16AutoButton->setMinimumWidth(60);
    threshold16AutoButton->setVisible(true);

    polarityModeComboBox = new QComboBox(this);
    polarityModeComboBox->addItem("Polarity: Auto");
    polarityModeComboBox->addItem("Polarity: Object Darker");
    polarityModeComboBox->addItem("Polarity: Object Brighter");
    polarityModeComboBox->setCurrentIndex(0);
    polarityModeComboBox->setMinimumWidth(165);

    previewMaskButton = new QPushButton("Preview Mask", this);
    previewMaskButton->setMinimumWidth(95);

    smoothingLabel = new QLabel("Smoothing: 0%", this);
    smoothingLabel->setMinimumWidth(95);
    
    smoothingIntensitySlider = new QSlider(Qt::Horizontal, this);
    smoothingIntensitySlider->setRange(0, 100);
    smoothingIntensitySlider->setValue(0);
    smoothingIntensitySlider->setMinimumWidth(90);
    smoothingIntensitySlider->setMaximumWidth(110);

    // Create container widget for segmentation controls
    QWidget* segmentationWidget = new QWidget(this);
    QVBoxLayout* segmentationLayout = new QVBoxLayout(segmentationWidget);
    segmentationLayout->setContentsMargins(0, 2, 0, 2);
    segmentationLayout->setSpacing(3);

    // First row: Mode, Threshold Label, Slider, Spinbox
    QHBoxLayout* controlsLayout = new QHBoxLayout();
    controlsLayout->setContentsMargins(0, 0, 0, 0);
    controlsLayout->setSpacing(5);
    controlsLayout->addWidget(new QLabel("Segmentation:", this), 0);
    controlsLayout->addWidget(new QLabel("16-bit", this), 0);
    controlsLayout->addWidget(thresholdLabel, 0);
    controlsLayout->addWidget(threshold16SpinBox, 0);

    // Second row: Auto, Preview, and Smoothing controls
    QHBoxLayout* buttonsLayout = new QHBoxLayout();
    buttonsLayout->setContentsMargins(0, 0, 0, 0);
    buttonsLayout->setSpacing(5);
    buttonsLayout->addWidget(threshold16AutoButton, 0);
    buttonsLayout->addWidget(polarityModeComboBox, 0);
    buttonsLayout->addWidget(previewMaskButton, 0);
    buttonsLayout->addWidget(new QLabel("", this), 0);  // Spacer
    buttonsLayout->addWidget(smoothingLabel, 0);
    buttonsLayout->addWidget(smoothingIntensitySlider, 0);
    buttonsLayout->addStretch(1);

    segmentationLayout->addLayout(controlsLayout);
    segmentationLayout->addLayout(buttonsLayout);

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
    connect(polarityModeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int idx) {
                const QString mode = (idx == 1) ? "manual dark-foreground"
                                                : (idx == 2) ? "manual bright-foreground"
                                                             : "auto polarity";
                statusBar()->showMessage(QString("Polarity mode set to %1").arg(mode), 2500);
            });
    connect(previewMaskButton, &QPushButton::clicked, this, &MainWindow::onPreviewMask);
    connect(smoothingIntensitySlider, QOverload<int>::of(&QSlider::valueChanged), this, &MainWindow::onSmoothingIntensityChanged);

        syncThresholdControls();
}


void MainWindow::openDataset() {
    QFileDialog dialog(this);
    dialog.setWindowTitle("Load MRI Dataset");
    dialog.setFileMode(QFileDialog::Directory); // Allow selecting a directory
    dialog.setOption(QFileDialog::ShowDirsOnly, true); // Only show directories

    if (dialog.exec() == QDialog::Accepted) {
        QString folderPath = dialog.selectedFiles().first(); // Get the selected folder path
        QDir dir(folderPath);

        // Filter for image files and enforce numeric ordering by slice number in filenames.
        QStringList filters = {"*.bmp", "*.dcm", "*.dicom", "*.png", "*.jpg", "*.tif"};
        QFileInfoList files = dir.entryInfoList(filters, QDir::Files, QDir::NoSort);
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

    QVector<int> sliceIndices(filePaths.size());
    std::iota(sliceIndices.begin(), sliceIndices.end(), 0);
    const double threshold = currentThreshold;
    const int totalSlices = filePaths.size();

    // Create a progress dialog
    QProgressDialog progressDialog("Loading images...", "Cancel", 0, filePaths.size(), this);
    progressDialog.setWindowTitle("Loading Progress");
    progressDialog.setWindowModality(Qt::WindowModal); // Block interaction with main window
    progressDialog.setMinimumDuration(0); // Show immediately
    progressDialog.show();

    QFutureWatcher<LoadedSliceResult> watcher;
    QEventLoop waitLoop;
    connect(&watcher, &QFutureWatcher<LoadedSliceResult>::finished, &waitLoop, &QEventLoop::quit);
    connect(&watcher, &QFutureWatcher<LoadedSliceResult>::progressRangeChanged,
            &progressDialog, &QProgressDialog::setRange);
    connect(&watcher, &QFutureWatcher<LoadedSliceResult>::progressValueChanged,
            &progressDialog, &QProgressDialog::setValue);
    connect(&watcher, &QFutureWatcher<LoadedSliceResult>::progressValueChanged,
            this, [this, &progressDialog, totalSlices](int value) {
                progressDialog.setLabelText(QString("Loading image %1/%2").arg(value).arg(totalSlices));
            });
    connect(&progressDialog, &QProgressDialog::canceled, &watcher, &QFutureWatcher<LoadedSliceResult>::cancel);

    watcher.setFuture(QtConcurrent::mapped(sliceIndices, [this, &filePaths, threshold, totalSlices](int index) -> LoadedSliceResult {
        LoadedSliceResult result;
        result.index = index;

        bool usedFallback = false;
        const QImage originalImg = loadSliceImage(filePaths[index], &usedFallback);
        if (originalImg.isNull()) {
            qWarning() << "Failed to decode image (Qt/OpenCV):" << filePaths[index];
            return result;
        }

        if (usedFallback) {
            static std::atomic<int> fallbackLogCount{0};
            if (fallbackLogCount.fetch_add(1, std::memory_order_relaxed) < 5) {
                qDebug() << "Loaded slice via fallback decoder:" << QFileInfo(filePaths[index]).fileName();
            }
        }

        result.original = originalImg;
        result.processed = preprocessLoadedImage(originalImg, filePaths[index], threshold, totalSlices);
        if (!decodeSegmentationSlice16(filePaths[index], &result.segmentation16, &result.hasNative16)) {
            result.segmentation16 = qImageToSegmentationU16(originalImg);
            result.hasNative16 = false;
        }
        // Keep the 16-bit mesh path aligned with the same ROI used in preview preprocessing.
        // Without this crop, scanner frame/slab background dominates meshing and hides anatomy.
        result.segmentation16 = cropSegmentationSlice16(result.segmentation16);

        result.valid = !result.processed.isNull();
        return result;
    }));

    waitLoop.exec();

    if (progressDialog.wasCanceled()) {
        loadedImages.clear();
        originalImages.clear();
        statusBar()->showMessage("Loading canceled.", 3000);
        return;
    }

    const QList<LoadedSliceResult> results = watcher.future().results();
    QVector<LoadedSliceResult> orderedResults(filePaths.size());
    for (const LoadedSliceResult& result : results) {
        if (result.index >= 0 && result.index < orderedResults.size()) {
            orderedResults[result.index] = result;
        }
    }

    int failedCount = 0;
    int native16Count = 0;
    for (const LoadedSliceResult& result : orderedResults) {
        if (!result.valid) {
            ++failedCount;
            continue;
        }
        originalImages.append(result.original);
        loadedImages.append(result.processed);
        if (result.hasNative16) {
            ++native16Count;
        }
        segmentationSlices16.append(result.segmentation16.clone());
    }

    hasSegmentation16Data = (!segmentationSlices16.isEmpty() && segmentationSlices16.size() == loadedImages.size());

    qDebug() << "Segmentation ingestion:" << segmentationSlices16.size() << "slice(s),"
             << native16Count << "native 16-bit";

    progressDialog.close();

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

        const QString msg = (failedCount > 0)
            ? QString("Loaded %1 image(s). %2 file(s) could not be decoded. Seg16: %3/%4 native. Threshold reset to %5.")
                .arg(loadedImages.size()).arg(failedCount).arg(native16Count).arg(segmentationSlices16.size()).arg(currentThreshold16)
            : QString("Images loaded successfully (%1 slices). Seg16 native: %2/%3. Threshold reset to %4.")
                .arg(loadedImages.size()).arg(native16Count).arg(segmentationSlices16.size()).arg(currentThreshold16);
        statusBar()->showMessage(msg, 5000);
        updateImagePreviews();
        mainTabs->setCurrentIndex(1);  // Switch to preview tab
    } else {
        statusBar()->showMessage(
            QString("No valid images loaded (%1 file(s) failed to decode). "
                    "Compressed DICOM (JPEG/JPEG2000) is not supported.").arg(failedCount),
            8000);
        updateImagePreviews();      // Shows the informative empty-state label
        mainTabs->setCurrentIndex(1); // Switch to preview tab so the message is visible
    }

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

    if (originalImages.isEmpty()) {
        QLabel* emptyLabel = new QLabel(
            "No valid images loaded.\n\n"
            "Supported: uncompressed 8-bit or 16-bit monochrome DICOM\n"
            "(BMP, PNG, JPG, TIF are also supported).\n\n"
            "Note: JPEG or JPEG 2000 compressed DICOM files cannot be decoded.");
        emptyLabel->setAlignment(Qt::AlignCenter);
        emptyLabel->setWordWrap(true);
        emptyLabel->setStyleSheet("color: #a0a0a0; font-size: 13px; padding: 30px;");
        previewLayout->addWidget(emptyLabel, 0, 0);
        return;
    }

    // Use originalImages instead of loadedImages
    const int thumbsPerRow = 4;
    const int thumbSize = 200;

    for (int i = 0; i < originalImages.size(); ++i) {
        QLabel* thumbLabel = new QLabel;
        QImage thumb = originalImages[i].scaled(thumbSize, thumbSize,
                                                Qt::KeepAspectRatio,
                                                Qt::SmoothTransformation);

        thumbLabel->setPixmap(QPixmap::fromImage(thumb));
        thumbLabel->setToolTip(QString("Double-click to view full size\nSlice %1").arg(i+1));
        thumbLabel->setAlignment(Qt::AlignCenter);
        thumbLabel->setStyleSheet("border: 1px solid #505050; margin: 2px;");

        // Install event filter for double-click
        thumbLabel->installEventFilter(this);
        thumbLabel->setProperty("imageIndex", i);  // Store image index

        previewLayout->addWidget(thumbLabel, i / thumbsPerRow, i % thumbsPerRow);
    }
}

bool MainWindow::eventFilter(QObject* watched, QEvent* event) {
    if (event->type() == QEvent::MouseButtonDblClick) {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            QLabel* label = qobject_cast<QLabel*>(watched);
            if (label) {
                int index = label->property("imageIndex").toInt();
                if (index >= 0 && index < originalImages.size()) {  // Check originalImages size
                    showFullSizeImage(originalImages[index]);  // Pass original image
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

    // Fixed ROI used by the existing workflow.
    QRect cropRect(45, 48, 1417, 537);
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
            if (borderTouchCount >= 3 && area > (totalPixels / 12)) {
                continue;
            }
            // Dense stacks often contain bright horizontal streak artifacts.
            // They are thin and span much of the width, unlike the jaw body.
            if (denseStack && thinHorizontalStrip && area < (totalPixels / 24)) {
                continue;
            }
            if (height > (rows * 17 / 20) && area > (totalPixels / 14)) {
                continue;
            }
            const int topRejectBand = sparseStack ? (rows / 14) : (rows / 8);
            if (centerY < topRejectBand && area > (totalPixels / 220)) {
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

        static std::atomic<int> segmentationLogCount{0};
        const int logIndex = segmentationLogCount.fetch_add(1, std::memory_order_relaxed);
        if (logIndex < 12) {
            qDebug() << "Mask profile" << stackProfile
                     << "kept" << keptComponents << "/" << candidates.size()
                     << "minArea" << minComponentArea
                     << "threshold" << thresholdValue;
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

    // Preserve full dataset fidelity for medical 3D reconstruction.
    const int width = srcWidth;
    const int height = srcHeight;
    const float smoothingIntensity = 1.0f;  // Legacy 8-bit path uses full smoothing
    const bool sparseInput = (srcDepth <= 48);

    QVector<int> zStepCounts;
    zStepCounts.reserve(qMax(0, srcDepth - 1));
    int depth = srcDepth;
    bool interpolatedSparseGaps = false;

    if (sparseInput && currentImagePaths.size() == srcDepth && srcDepth >= 2) {
        bool allNumbersValid = true;
        int totalGapDepth = 1;
        QVector<int> observedGaps;
        observedGaps.reserve(srcDepth - 1);

        for (int i = 0; i + 1 < currentImagePaths.size(); ++i) {
            bool okA = false;
            bool okB = false;
            const QFileInfo fileA(currentImagePaths[i]);
            const QFileInfo fileB(currentImagePaths[i + 1]);
            const int sliceA = extractLastNumber(fileA.completeBaseName(), &okA);
            const int sliceB = extractLastNumber(fileB.completeBaseName(), &okB);
            if (!okA || !okB) {
                allNumbersValid = false;
                break;
            }

            const int rawGap = qMax(1, sliceB - sliceA);
            const int clampedGap = qBound(1, rawGap, 8);
            zStepCounts.append(clampedGap);
            observedGaps.append(rawGap);
            totalGapDepth += clampedGap;
        }

        if (allNumbersValid && !observedGaps.isEmpty()) {
            QVector<int> sortedGaps = observedGaps;
            std::sort(sortedGaps.begin(), sortedGaps.end());
            const int medianGap = sortedGaps[sortedGaps.size() / 2];
            if (medianGap > 1) {
                depth = totalGapDepth;
                interpolatedSparseGaps = true;
                qDebug() << "Sparse stack appears subsampled; interpolating Z gaps."
                         << "median gap:" << medianGap
                         << "depth:" << srcDepth << "->" << depth;
            }
        }
    }

    const bool sparseDepth = sparseInput;
    const bool denseDepth = (!sparseInput && depth >= 200);
    const QString depthProfile = sparseDepth ? "sparse" : (denseDepth ? "dense" : "balanced");

    // Allocate 3D array [z][y][x]
    volume.resize(depth);
    for (int z = 0; z < depth; ++z) {
        volume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            volume[z][y].resize(width);
        }
    }

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

    if (interpolatedSparseGaps) {
        int outZ = 0;
        for (int srcZ = 0; srcZ + 1 < srcDepth; ++srcZ) {
            const int stepCount = qMax(1, zStepCounts.value(srcZ, 1));
            writeVolumeSlice(outZ++, images[srcZ], nullptr, 0.0f);
            for (int step = 1; step < stepCount; ++step) {
                const float lerpT = float(step) / float(stepCount);
                writeVolumeSlice(outZ++, images[srcZ], &images[srcZ + 1], lerpT);
            }

            if (progressCallback) {
                const int progress = 5 + (outZ * 70 / qMax(1, depth));
                progressCallback(progress, QString("Building interpolated 3D volume... %1/%2 slices").arg(outZ).arg(depth));
            }
        }
        writeVolumeSlice(depth - 1, images.last(), nullptr, 0.0f);
    } else {
        for (int z = 0; z < depth; ++z) {
            writeVolumeSlice(z, images[z], nullptr, 0.0f);

            if (progressCallback) {
                const int progress = 5 + ((z + 1) * 70 / depth);
                progressCallback(progress, QString("Building 3D volume... %1/%2 slices").arg(z + 1).arg(depth));
            }
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

    // ----------------------------------------------------------------
    // 3-D Gaussian smoothing: converts the hard binary 0/1 volume into
    // smooth gradients so Marching Cubes can interpolate surface positions
    // accurately instead of producing staircase banding.
    // ----------------------------------------------------------------

    // Pass 1 – XY per-slice blur (smooths ragged mask edges within each slice)
    if (progressCallback) {
        progressCallback(82, "Smoothing volume (XY pass)...");
    }
    const double baseSigma = sparseDepth ? 0.65 : (denseDepth ? 2.3 : 1.4);
    const double xySigma = baseSigma * smoothingIntensity;
    const cv::Size xyKernel = sparseDepth ? cv::Size(3, 3) : (xySigma < 0.1 ? cv::Size(3, 3) : cv::Size(5, 5));
    qDebug() << "Volume profile" << depthProfile
             << "XY sigma" << xySigma << "(base=" << baseSigma << "intensity=" << smoothingIntensity << ")"
             << "spacing" << voxelSpacingX << voxelSpacingY << voxelSpacingZ;
    for (int z = 0; z < depth; ++z) {
        cv::Mat sliceMat(height, width, CV_32F);
        for (int y = 0; y < height; ++y)
            std::memcpy(sliceMat.ptr<float>(y), volume[z][y].constData(), width * sizeof(float));
        if (xySigma > 0.1) {
            cv::GaussianBlur(sliceMat, sliceMat, xyKernel, xySigma);
        }
        for (int y = 0; y < height; ++y)
            std::memcpy(volume[z][y].data(), sliceMat.ptr<float>(y), width * sizeof(float));
    }

    // Sparse stacks can have large inter-slice jumps; bridge obvious one-slice gaps
    // while suppressing isolated one-slice speckles before Z smoothing.
    if (sparseDepth && depth >= 3) {
        if (progressCallback) {
            progressCallback(85, "Smoothing volume (sparse Z consistency)...");
        }
        VolumeData zConsistent = volume;
        for (int z = 1; z < depth - 1; ++z) {
            for (int y = 0; y < height; ++y) {
                const float* prevRow = volume[z - 1][y].constData();
                const float* currRow = volume[z][y].constData();
                const float* nextRow = volume[z + 1][y].constData();
                float* outRow = zConsistent[z][y].data();

                for (int x = 0; x < width; ++x) {
                    const float prevVal = prevRow[x];
                    const float currVal = currRow[x];
                    const float nextVal = nextRow[x];

                    const bool prevOn = (prevVal > 0.55f);
                    const bool currOn = (currVal > 0.55f);
                    const bool nextOn = (nextVal > 0.55f);

                    if (!currOn && prevOn && nextOn) {
                        outRow[x] = qMax(currVal, 0.72f);
                    } else if (currOn && !prevOn && !nextOn) {
                        outRow[x] = currVal * 0.35f;
                    }
                }
            }
        }
        volume.swap(zConsistent);
        qDebug() << "Sparse Z consistency bridge applied";
    }

    // Pass 2 – Z-direction 3-tap kernel [0.25, 0.5, 0.25]
    // Uses a rolling 2-plane buffer so extra memory is O(W*H) regardless of depth.
    if (progressCallback) {
        progressCallback(87, "Smoothing volume (Z pass)...");
    }
    if (depth >= 3) {
        const float zPrevNextW = sparseDepth ? 0.20f : 0.30f;
        const float zCenterW = 1.0f - (2.0f * zPrevNextW);
        // Multiple passes approximate a wider Gaussian in Z — crucial for dense stacks
        // where a single [0.25, 0.5, 0.25] pass leaves hard per-slice steps (staircase).
        const int zSmoothPasses = denseDepth ? 5 : (sparseDepth ? 2 : 2);
        qDebug() << "Z smoothing passes:" << zSmoothPasses << "weights:" << zPrevNextW << zCenterW << zPrevNextW;

        for (int pass = 0; pass < zSmoothPasses; ++pass) {
            std::vector<float> prevPlane(static_cast<std::size_t>(height) * width);
            std::vector<float> currPlane(static_cast<std::size_t>(height) * width);

            // Seed with the first (boundary) slice — left unmodified.
            for (int y = 0; y < height; ++y)
                std::memcpy(&prevPlane[y * width], volume[0][y].constData(), width * sizeof(float));

            for (int z = 1; z < depth - 1; ++z) {
                for (int y = 0; y < height; ++y)
                    std::memcpy(&currPlane[y * width], volume[z][y].constData(), width * sizeof(float));

                for (int y = 0; y < height; ++y) {
                    const float* prev = &prevPlane[y * width];
                    const float* curr = &currPlane[y * width];
                    const float* next = volume[z + 1][y].constData();
                    float* out  = volume[z][y].data();
                    for (int x = 0; x < width; ++x)
                        out[x] = zPrevNextW * prev[x] + zCenterW * curr[x] + zPrevNextW * next[x];
                }

                std::swap(prevPlane, currPlane);
            }
        }
    }

    // Suppress weak residual streaks after smoothing, mostly for dense stacks.
    // This keeps strong bone responses while removing faint banding layers.
    const float lowCut = sparseDepth ? 0.06f : (denseDepth ? 0.24f : 0.18f);
    const float highCut = sparseDepth ? 0.92f : (denseDepth ? 0.78f : 0.80f);
    const float invRange = 1.0f / qMax(1e-6f, highCut - lowCut);
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            float* row = volume[z][y].data();
            for (int x = 0; x < width; ++x) {
                const float v = (row[x] - lowCut) * invRange;
                row[x] = qBound(0.0f, v, 1.0f);
            }
        }
    }
    qDebug() << "Post-smooth clamp:" << lowCut << "to" << highCut;

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
    VolumeData volume;

    if (slices16.isEmpty()) {
        qWarning() << "No 16-bit slices to convert";
        return volume;
    }

    const int srcWidth = slices16.first().cols;
    const int srcHeight = slices16.first().rows;
    const int srcDepth = slices16.size();

    qDebug() << "Converting 16-bit slices to scalar volume. Dimensions:"
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
    const bool sparseDepth = sparseInput;
    const bool denseDepth = denseInput;

    volume.resize(depth);
    for (int z = 0; z < depth; ++z) {
        volume[z].resize(height);
        for (int y = 0; y < height; ++y) {
            volume[z][y].resize(width);
        }
    }

    if (progressCallback) {
        progressCallback(8, QString("Starting binary 16-bit processing pipeline (threshold=%1)...").arg(threshold16));
    }

    const int finalThreshold = qBound(0, threshold16, 65535);
    bool foregroundIsDark = false;
    if (polarityModeComboBox) {
        const int polarityMode = polarityModeComboBox->currentIndex();
        if (polarityMode == 1) {
            foregroundIsDark = true;
        } else if (polarityMode == 2) {
            foregroundIsDark = false;
        } else {
            foregroundIsDark = ImageProcessor::detectForegroundPolarity(slices16);
        }
    } else {
        foregroundIsDark = ImageProcessor::detectForegroundPolarity(slices16);
    }

    QVector<int> sliceIndices(depth);
    for (int z = 0; z < depth; ++z) {
        sliceIndices[z] = z;
    }

    const qint64 totalVoxels = static_cast<qint64>(width) * height * depth;
    std::atomic<qint64> filledVoxels{0};
    std::atomic<int> completedBinarySlices{0};

    auto binarizeSlice = [&](int z) {
        const cv::Mat& src16 = slices16[z];
        qint64 localFilledVoxels = 0;

        for (int y = 0; y < height; ++y) {
            const quint16* srcRow = src16.ptr<quint16>(y);
            float* dstRow = volume[z][y].data();
            for (int x = 0; x < width; ++x) {
                const bool foreground = foregroundIsDark ? (srcRow[x] <= finalThreshold) : (srcRow[x] >= finalThreshold);
                const float value = foreground ? 1.0f : 0.0f;
                dstRow[x] = value;
                if (value >= 0.5f) {
                    ++localFilledVoxels;
                }
            }
        }

        filledVoxels.fetch_add(localFilledVoxels, std::memory_order_relaxed);

        const int finished = completedBinarySlices.fetch_add(1, std::memory_order_relaxed) + 1;
        if (progressCallback && (finished % 8 == 0 || finished == depth)) {
            const int progress = 8 + ((finished * 12) / qMax(1, depth));
            progressCallback(progress, QString("Building 16-bit binary slices... %1/%2").arg(finished).arg(depth));
        }
    };

    if (depth >= 8) {
        QtConcurrent::blockingMap(sliceIndices, [&](int& z) {
            binarizeSlice(z);
        });
    } else {
        for (int z = 0; z < depth; ++z) {
            binarizeSlice(z);
        }
    }

    const double occupancy = totalVoxels > 0 ? (100.0 * static_cast<double>(filledVoxels.load(std::memory_order_relaxed)) / static_cast<double>(totalVoxels)) : 0.0;
    qDebug() << "16-bit threshold used:" << finalThreshold;
    qDebug() << "16-bit polarity:" << (foregroundIsDark ? "dark-foreground (<= threshold)" : "bright-foreground (>= threshold)");
    qDebug() << "16-bit binary volume occupancy:" << occupancy << "%";

    if (progressCallback) {
        progressCallback(80, QString("Binary threshold %1 | occupancy %2%")
            .arg(finalThreshold)
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
    currentMesh = MarchingCubes::generateMeshStreaming(
        currentVolume,
        isoLevel,
        8,
        voxelSpacingX,
        voxelSpacingY,
        voxelSpacingZ
    );
     qDebug() << "Generated mesh:"
              << currentMesh.vertices.size() << "vertices,"
              << currentMesh.indices.size() / 3 << "triangles";

     qDebug() << "Updating GLWidget...";
     if (mainTabs) {
         mainTabs->setCurrentIndex(0);
     }
     glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
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
     if (currentMesh.vertices.isEmpty() || currentMesh.indices.isEmpty()) {
         statusBar()->showMessage("No mesh to export!", 3000);
         return;
     }

     qDebug() << "Vertices count:" << currentMesh.vertices.size();
     qDebug() << "Indices count:" << currentMesh.indices.size();

     QString fileName = QFileDialog::getSaveFileName(this, "Save STL File", "", "STL Files (*.stl)");
     if (fileName.isEmpty()) return;

     QFile file(fileName);
     if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
         QMessageBox::warning(this, "Error", "Could not save file");
         return;
     }

     QTextStream out(&file);
     out << "solid Generated\n";

     int triangleCount = 0;
     for (int i = 0; i < currentMesh.indices.size(); i += 3) {
         if (i + 2 >= currentMesh.indices.size()) {
             qWarning() << "Incomplete triangle at index" << i;
             break;
         }

         const QVector3D& v0 = currentMesh.vertices[currentMesh.indices[i]];
         const QVector3D& v1 = currentMesh.vertices[currentMesh.indices[i + 1]];
         const QVector3D& v2 = currentMesh.vertices[currentMesh.indices[i + 2]];

         // Calculate normal
         QVector3D normal = QVector3D::normal(v1 - v0, v2 - v0);

         out << "facet normal "
             << normal.x() << " " << normal.y() << " " << normal.z() << "\n";
         out << " outer loop\n";
         out << "  vertex " << v0.x() << " " << v0.y() << " " << v0.z() << "\n";
         out << "  vertex " << v1.x() << " " << v1.y() << " " << v1.z() << "\n";
         out << "  vertex " << v2.x() << " " << v2.y() << " " << v2.z() << "\n";
         out << " endloop\n";
         out << "endfacet\n";

         triangleCount++;
     }

     out << "endsolid Generated\n";
     file.close();

     qDebug() << "Exported" << triangleCount << "triangles to STL file.";
     statusBar()->showMessage("STL exported successfully!", 3000);
 }

 void MainWindow::generateMesh() {

     if (isGeneratingMesh) return;
     if (isOtsuRunning) {
         statusBar()->showMessage("Otsu is running. Please wait...", 2500);
         return;
     }

     isGeneratingMesh = true;
     statusBar()->showMessage("Starting mesh generation...");

     currentThreshold16 = threshold16SpinBox ? threshold16SpinBox->value() : currentThreshold16;

     QFuture<void> future = QtConcurrent::run([this]() {
         const int sliceCount = loadedImages.size();
        const float targetIso = 0.50f;
         const QString profile = (sliceCount <= 48) ? "sparse" : ((sliceCount >= 200) ? "dense" : "balanced");
         qDebug() << "Reconstruction profile:" << profile
                  << "slices:" << sliceCount
                  << "segmentation mode:" << "16-bit"
                  << "iso:" << targetIso
                  << "spacing:" << voxelSpacingX << voxelSpacingY << voxelSpacingZ;

         auto reportProgress = [this](int progress, const QString& message) {
             QMetaObject::invokeMethod(this, [this, progress, message]() {
                 updateLoadingDialog(progress, message);
             }, Qt::QueuedConnection);
         };

         reportProgress(1, "Preparing volume data...");

         VolumeData volume;
         if (!hasSegmentation16Data || segmentationSlices16.isEmpty()) {
             qWarning() << "16-bit segmentation data is unavailable; cannot generate mesh.";
             QMetaObject::invokeMethod(this, [this]() {
                 statusBar()->showMessage("16-bit segmentation data unavailable. Reload dataset.", 4000);
             }, Qt::QueuedConnection);
             return;
         }

         volume = convertToVolume16(segmentationSlices16, currentThreshold16, currentSmoothingIntensity, reportProgress);

         if (!volume.isEmpty()) {
             reportProgress(92, "Running Marching Cubes on GPU...");
             pendingMesh = MarchingCubes::generateMeshStreaming(
                 volume,
                 targetIso,
                 8,
                 voxelSpacingX,
                 voxelSpacingY,
                 voxelSpacingZ
             );
             reportProgress(100, "Finalizing mesh...");
         }
     });

     meshGenerationWatcher.setFuture(future);
 }

 void MainWindow::handleMeshGenerationStarted() {
     updateLoadingDialog(0, "Generating 3D mesh...");
     loadingDialog->show();
     QApplication::processEvents();
 }

 void MainWindow::handleMeshComputationFinished() {
     if (!pendingMesh.vertices.isEmpty()) {
        // Save threshold as good threshold for warm-start on next dataset
        imageProcessor.saveLastGoodThreshold(lastMetrics.adaptiveThreshold);
        
        // Ensure the OpenGL tab is active before uploading buffers.
        if (mainTabs) {
            mainTabs->setCurrentIndex(0);
        }
        updateLoadingDialog(95, "Uploading mesh to renderer...");
        currentMesh = pendingMesh; // Update currentMesh for export
        glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
     } else {
         handleMeshRenderingFinished();
     }
 }

 void MainWindow::handleMeshRenderingFinished() {
     isGeneratingMesh = false;
     updateLoadingDialog(100, "Mesh upload complete");
     loadingDialog->hide();

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
    if (!loadingDialog) {
        return;
    }

    loadingDialog->setValue(qBound(0, progress, 100));
    if (!message.isEmpty()) {
        loadingDialog->setLabelText(message);
    }
}

 void MainWindow::saveProcessedImage(const QImage& image, const QString& filePath) {
     if (image.save(filePath)) {
         qDebug() << "Processed image saved to:" << filePath;
     } else {
         qWarning() << "Failed to save processed image to:" << filePath;
     }
 }

 void MainWindow::loadMesh() {
     // Open file dialogs to select vertices and faces files
     QString verticesPath = QFileDialog::getOpenFileName(this, "Select Vertices File", "", "Text Files (*.txt)");
     if (verticesPath.isEmpty()) {
         qWarning() << "No vertices file selected.";
         return;
     }

     QString facesPath = QFileDialog::getOpenFileName(this, "Select Faces File", "", "Text Files (*.txt)");
     if (facesPath.isEmpty()) {
         qWarning() << "No faces file selected.";
         return;
     }

     // Pass the file paths to the GLWidget
     glWidget->loadMeshFromFiles(verticesPath, facesPath);
 }


void MainWindow::syncThresholdControls() {
    threshold16SpinBox->setVisible(true);
    threshold16AutoButton->setVisible(true);
    if (polarityModeComboBox) {
        polarityModeComboBox->setVisible(true);
    }
    previewMaskButton->setVisible(true);
    thresholdLabel->setText(QString("Threshold (16-bit): %1").arg(currentThreshold16));
}

void MainWindow::onThreshold16Changed(int value) {
    currentThreshold16 = value;
    thresholdLabel->setText(QString("Threshold (16-bit): %1").arg(value));
}

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
    isOtsuRunning = false;
    if (generate3DAct) {
        generate3DAct->setEnabled(true);
    }
    threshold16AutoButton->setEnabled(true);

    statusBar()->showMessage(
        QString("Threshold suggested: %1. Click Generate 3D to apply full adaptive processing and create mesh.")
            .arg(boundedThreshold),
        5000);
}

void MainWindow::onSmoothingIntensityChanged(int value) {
    currentSmoothingIntensity = value / 100.0f;  // Convert 0-100 to 0.0-1.0
    if (smoothingLabel) {
        smoothingLabel->setText(QString("Smoothing: %1%").arg(value));
    }
    qDebug() << "Smoothing intensity changed to:" << currentSmoothingIntensity;
}

void MainWindow::onPreviewMask() {
    if (!hasSegmentation16Data || segmentationSlices16.isEmpty()) {
        QMessageBox::information(this,
                                 "Preview Mask Unavailable",
                                 "No images are loaded, so a mask preview cannot be shown.");
        return;
    }

    const int threshold = threshold16SpinBox ? threshold16SpinBox->value() : currentThreshold16;
    bool foregroundIsDark = detectForegroundIsDark(segmentationSlices16);
    const int polarityMode = polarityModeComboBox ? polarityModeComboBox->currentIndex() : 0;
    if (polarityMode == 1) {
        foregroundIsDark = true;
    } else if (polarityMode == 2) {
        foregroundIsDark = false;
    }
    const int depth = segmentationSlices16.size();
    const int midZ = depth / 2;
    cv::Mat src = segmentationSlices16[midZ];
    if (src.empty()) {
        QMessageBox::warning(this, "Preview Mask", "Selected slice is empty.");
        return;
    }

    cv::Mat u16;
    if (src.type() == CV_16UC1) {
        u16 = src;
    } else {
        src.convertTo(u16, CV_16UC1);
    }

    cv::Mat original8;
    cv::normalize(u16, original8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    cv::Mat binaryMask(u16.rows, u16.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < u16.rows; ++y) {
        const quint16* row = reinterpret_cast<const quint16*>(u16.ptr(y));
        uchar* out = binaryMask.ptr<uchar>(y);
        for (int x = 0; x < u16.cols; ++x) {
            const bool foreground = foregroundIsDark ? (row[x] <= threshold) : (row[x] >= threshold);
            out[x] = foreground ? 255 : 0;
        }
    }

    const bool sparseDepth = (depth <= 48);
    const bool denseDepth = (depth >= 200);
    const cv::Mat cleanedMask = cleanupBinaryMask16(binaryMask, sparseDepth, denseDepth);

    QDialog previewDialog(this);
    previewDialog.setWindowTitle("Threshold Mask Preview");
    previewDialog.resize(1050, 420);

    QVBoxLayout* rootLayout = new QVBoxLayout(&previewDialog);
    QLabel* infoLabel = new QLabel(
        QString("Slice %1/%2 | threshold=%3 | polarity=%4 | white pixels(before/after)=%5/%6")
            .arg(midZ + 1)
            .arg(depth)
            .arg(threshold)
            .arg(foregroundIsDark ? "<= (dark fg)" : ">= (bright fg)")
            .arg(cv::countNonZero(binaryMask))
            .arg(cv::countNonZero(cleanedMask)),
        &previewDialog);
    rootLayout->addWidget(infoLabel);

    QHBoxLayout* imagesLayout = new QHBoxLayout();
    auto makePane = [&](const QString& title, const cv::Mat& mat) {
        QWidget* panel = new QWidget(&previewDialog);
        QVBoxLayout* panelLayout = new QVBoxLayout(panel);
        panelLayout->setContentsMargins(0, 0, 0, 0);
        QLabel* titleLabel = new QLabel(title, panel);
        QLabel* imageLabel = new QLabel(panel);
        imageLabel->setAlignment(Qt::AlignCenter);
        imageLabel->setStyleSheet("border: 1px solid #555;");
        const QImage img = cvMatToQImage(mat);
        imageLabel->setPixmap(QPixmap::fromImage(img).scaled(320, 320, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        panelLayout->addWidget(titleLabel);
        panelLayout->addWidget(imageLabel);
        imagesLayout->addWidget(panel);
    };

    makePane("Original (normalized)", original8);
    makePane("Binary mask", binaryMask);
    makePane("Cleaned mask", cleanedMask);

    rootLayout->addLayout(imagesLayout);
    previewDialog.exec();
}

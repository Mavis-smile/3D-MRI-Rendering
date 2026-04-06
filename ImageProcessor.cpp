#include "ImageProcessor.h"
#include <QDebug>
#include <QStandardPaths>
#include <QFile>
#include <QSettings>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace {

// Helper: Compute histogram for 16-bit data
std::vector<qint64> computeHistogram16(const cv::Mat& slice16, int x0, int y0, int x1, int y1) {
    std::vector<qint64> hist(65536, 0);

    if (slice16.type() != CV_16UC1 || slice16.empty()) {
        return hist;
    }

    const int rows = slice16.rows;
    const int cols = slice16.cols;

    x0 = qMax(0, x0);
    y0 = qMax(0, y0);
    x1 = qMin(cols, x1);
    y1 = qMin(rows, y1);

    for (int y = y0; y < y1; ++y) {
        const quint16* row = slice16.ptr<quint16>(y);
        for (int x = x0; x < x1; ++x) {
            hist[row[x]]++;
        }
    }

    return hist;
}

// Helper: Percentile from histogram
int histogramPercentile(const std::vector<qint64>& histogram, qint64 total, double pct) {
    if (total <= 0) return 0;

    const qint64 target = qint64(std::llround(double(total) * pct));
    qint64 accum = 0;
    for (int i = 0; i < int(histogram.size()); ++i) {
        accum += histogram[i];
        if (accum > target) return i;
    }
    return int(histogram.size()) - 1;
}

// Helper: Compute Otsu from histogram
int otsuThresholdFromHistogram(const std::vector<qint64>& histogram) {
    qint64 total = 0;
    for (qint64 count : histogram) {
        total += count;
    }

    if (total <= 0) return 32768;

    double sumTotal = 0.0;
    for (int i = 0; i < int(histogram.size()); ++i) {
        sumTotal += double(i) * double(histogram[i]);
    }

    qint64 sumBackground = 0;
    qint64 weightBackground = 0;
    double maxVariance = 0.0;
    int optimalThreshold = 32768;

    for (int t = 0; t < int(histogram.size()); ++t) {
        weightBackground += histogram[t];
        if (weightBackground == 0) continue;

        const qint64 weightForeground = total - weightBackground;
        if (weightForeground == 0) break;

        sumBackground += qint64(t) * histogram[t];

        const double muBackground = double(sumBackground) / double(weightBackground);
        const double muForeground = (sumTotal - double(sumBackground)) / double(weightForeground);
        const double variance = double(weightBackground) * double(weightForeground)
                                * (muBackground - muForeground) * (muBackground - muForeground);

        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }

    return optimalThreshold;
}

// Helper: Per-slice Otsu with local ROI
int otsuSliceLocal(const cv::Mat& slice16, int borderPixels) {
    int x0 = borderPixels;
    int y0 = borderPixels;
    int x1 = slice16.cols - borderPixels;
    int y1 = slice16.rows - borderPixels;

    x0 = qMax(0, x0);
    y0 = qMax(0, y0);
    x1 = qMin(slice16.cols, x1);
    y1 = qMin(slice16.rows, y1);

    auto hist = computeHistogram16(slice16, x0, y0, x1, y1);
    return otsuThresholdFromHistogram(hist);
}

// Helper: Connected components with soft pruning
cv::Mat cleanupBinaryMaskSoft(
    const cv::Mat& inputMask,
    int /* sliceIndex */,
    int totalSlices,
    double confidenceLevel
) {
    if (inputMask.empty()) {
        return inputMask.clone();
    }

    cv::Mat mask = inputMask.clone();
    const int rows = mask.rows;
    const int cols = mask.cols;
    const int totalPixels = rows * cols;

    // Stack profile
    const bool sparseStack = (totalSlices <= 48);
    const bool denseStack = (totalSlices >= 200);

    // Border suppression is now conditional: only trigger when a slab/frame artifact dominates borders.
    const int probeY = qMax(1, rows / 24);
    const int probeX = qMax(1, cols / 24);
    cv::Mat borderProbe = cv::Mat::zeros(mask.size(), CV_8UC1);
    borderProbe.rowRange(0, probeY).setTo(255);
    borderProbe.rowRange(rows - probeY, rows).setTo(255);
    borderProbe.colRange(0, probeX).setTo(255);
    borderProbe.colRange(cols - probeX, cols).setTo(255);

    cv::Mat centerProbe = cv::Mat::zeros(mask.size(), CV_8UC1);
    const int centerX0 = cols / 4;
    const int centerX1 = cols - centerX0;
    const int centerY0 = rows / 4;
    const int centerY1 = rows - centerY0;
    centerProbe(cv::Rect(centerX0, centerY0, qMax(1, centerX1 - centerX0), qMax(1, centerY1 - centerY0))).setTo(255);

    const int borderPixels = qMax(1, cv::countNonZero(borderProbe));
    const int centerPixels = qMax(1, cv::countNonZero(centerProbe));
    const double borderOccupancy = double(cv::countNonZero(mask & borderProbe)) / double(borderPixels);
    const double centerOccupancy = double(cv::countNonZero(mask & centerProbe)) / double(centerPixels);

    const bool borderDominantArtifact = (borderOccupancy > 0.95) && (centerOccupancy < 0.15);
    if (borderDominantArtifact) {
        const double borderSuppression = 0.08 + (0.20 * confidenceLevel);
        const double borderDivisor = denseStack ? 140.0 : 100.0;
        const int borderY = qMax(1, int(std::round(rows / (borderDivisor / borderSuppression))));
        const int borderX = qMax(1, int(std::round(cols / (borderDivisor / borderSuppression))));

        mask.rowRange(0, borderY).setTo(0);
        mask.rowRange(rows - borderY, rows).setTo(0);
        mask.colRange(0, borderX).setTo(0);
        mask.colRange(cols - borderX, cols).setTo(0);
    }

    // Morphology adjusted by stack profile
    // Dense stacks should avoid open/close because they erase trabecular texture.
    const cv::Mat kernelSmall = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    if (!sparseStack && !denseStack) {
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernelSmall);
    }
    if (!denseStack) {
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernelSmall);
    }

    // Connected components with soft thresholding
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

    // Minimum component area: lower when confidence is low (keep more)
    const double minAreaFactor = 0.5 + (0.5 * confidenceLevel);  // 0.5-1.0
    const int minComponentArea = sparseStack
        ? qMax(8, int(totalPixels / (25000.0 / minAreaFactor)))
        : (denseStack ? qMax(8, int(totalPixels / (22000.0 / minAreaFactor)))
                      : qMax(60, int(totalPixels / (4000.0 / minAreaFactor))));

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

        const double bboxArea = double(qMax(1, compWidth * compHeight));
        const double bboxFillRatio = double(area) / bboxArea;
        const bool nearFullFrame = (compWidth >= (cols * 99 / 100)) && (compHeight >= (rows * 99 / 100));
        const bool frameLike = nearFullFrame && (bboxFillRatio < 0.35);

        if (frameLike) {
            continue;
        }

        // Reject scanner-strip artifacts: thin/elongated components anchored near borders.
        const bool thinVerticalStrip = (compHeight >= (rows * 55 / 100))
            && (compWidth <= qMax(2, cols / 45))
            && (double(compHeight) / double(qMax(1, compWidth)) >= 7.0);
        const bool thinHorizontalStrip = (compWidth >= (cols * 55 / 100))
            && (compHeight <= qMax(2, rows / 45))
            && (double(compWidth) / double(qMax(1, compHeight)) >= 7.0);

        // Moderate-width border slabs can still appear in some stacks; catch them too.
        const bool borderLocalVerticalSlab = (compHeight >= (rows * 45 / 100))
            && (compWidth <= qMax(4, cols / 14))
            && (double(compHeight) / double(qMax(1, compWidth)) >= 3.0)
            && (area <= qMax(1, totalPixels / 5));
        const bool borderLocalHorizontalSlab = (compWidth >= (cols * 45 / 100))
            && (compHeight <= qMax(4, rows / 14))
            && (double(compWidth) / double(qMax(1, compHeight)) >= 3.0)
            && (area <= qMax(1, totalPixels / 5));

        const bool nearLeftOrRight = (left <= cols / 12) || (left + compWidth >= cols - (cols / 12));
        const bool nearTopOrBottom = (top <= rows / 12) || (top + compHeight >= rows - (rows / 12));

        const bool rejectVerticalStrip = nearLeftOrRight && (thinVerticalStrip || borderLocalVerticalSlab);
        const bool rejectHorizontalStrip = nearTopOrBottom && (thinHorizontalStrip || borderLocalHorizontalSlab);
        if (rejectVerticalStrip || rejectHorizontalStrip) {
            static int stripRejectLogCount = 0;
            if (stripRejectLogCount < 24) {
                ++stripRejectLogCount;
                qDebug() << "Rejected border strip component"
                         << "label" << label
                         << "bbox" << left << top << compWidth << compHeight
                         << "area" << area
                         << "rows/cols" << rows << cols
                         << "vert?" << rejectVerticalStrip
                         << "horiz?" << rejectHorizontalStrip;
            }
            continue;
        }

        // Reject only extreme border-dominant thin structures, not compact anatomy touching borders.
        const int borderAreaGate = denseStack ? qMax(1, totalPixels / 2) : qMax(1, totalPixels / 8);
        const bool extremeBorderComponent = (borderTouchCount >= 4)
            && (area > borderAreaGate)
            && (bboxFillRatio < 0.25);
        if (extremeBorderComponent) {
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

        // Keep more components when confidence is low
        double keepCountFactor = 0.5 + (1.5 * confidenceLevel);  // 0.5-2.0
        int targetKeepCount = sparseStack ? qMax(2, int(4 * keepCountFactor))
                  : (denseStack ? qMax(16, int(26 * keepCountFactor)) : qMax(1, int(2 * keepCountFactor)));
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

    // Final morphology: keep dense stacks untouched to preserve pores and fine internal details.
    if (!denseStack) {
        const cv::Size closeSize = sparseStack ? cv::Size(11, 5) : cv::Size(13, 7);
        const cv::Mat kernelClose = cv::getStructuringElement(cv::MORPH_ELLIPSE, closeSize);
        cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_CLOSE, kernelClose);
        cv::morphologyEx(filteredMask, filteredMask, cv::MORPH_OPEN, kernelSmall);
    }

    return filteredMask;
}

}  // namespace

// ============================================================================
// ImageProcessor implementation
// ============================================================================

ImageProcessor::ImageProcessor(QObject* parent)
    : QObject(parent)
{
    lastGoodThreshold = loadLastGoodThreshold();
}

bool ImageProcessor::detectForegroundPolarity(const QVector<cv::Mat>& slices16) {
    if (slices16.isEmpty()) {
        return false;
    }

    double centerAccum = 0.0;
    double borderAccum = 0.0;
    qint64 centerCount = 0;
    qint64 borderCount = 0;

    const int zStep = qMax(1, slices16.size() / 24);
    for (int z = 0; z < slices16.size(); z += zStep) {
        const cv::Mat& mat = slices16[z];
        cv::Mat u16;
        if (mat.type() == CV_16UC1) {
            u16 = mat;
        } else {
            mat.convertTo(u16, CV_16UC1);
        }

        if (u16.empty()) continue;

        const int rows = u16.rows;
        const int cols = u16.cols;
        const int cx0 = cols / 5;
        const int cx1 = cols - cx0;
        const int cy0 = rows / 5;
        const int cy1 = rows - cy0;

        for (int y = 0; y < rows; ++y) {
            const quint16* row = u16.ptr<quint16>(y);
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
    return (centerMean + 300.0) < borderMean;  // Center darker than border
}

int ImageProcessor::computeOtsuThreshold16(
    const QVector<cv::Mat>& slices16,
    double borderCrop
) {
    if (slices16.isEmpty()) {
        return 32768;
    }

    std::vector<qint64> histogram(65536, 0);
    qint64 totalPixels = 0;

    for (const cv::Mat& slice : slices16) {
        if (slice.empty()) continue;

        cv::Mat u16;
        if (slice.type() == CV_16UC1) {
            u16 = slice;
        } else {
            slice.convertTo(u16, CV_16UC1);
        }

        // ROI with border crop
        const int x0 = int(u16.cols * borderCrop);
        const int y0 = int(u16.rows * borderCrop);
        const int x1 = u16.cols - x0;
        const int y1 = u16.rows - y0;

        auto sliceHist = computeHistogram16(u16, x0, y0, x1, y1);
        for (int i = 0; i < int(histogram.size()); ++i) {
            histogram[i] += sliceHist[i];
            totalPixels += sliceHist[i];
        }
    }

    if (totalPixels <= 0) {
        return 32768;
    }

    // Clip outliers to focus on main data range
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

    return otsuThresholdFromHistogram(clippedHistogram);
}

int ImageProcessor::suggestThresholdQuick(
    const QVector<cv::Mat>& slices16,
    int previousGoodThreshold
) {
    // Fast threshold suggestion for Auto button
    // Just use Otsu computation without per-slice processing

    if (slices16.isEmpty()) {
        return previousGoodThreshold > 0 ? previousGoodThreshold : 32768;
    }

    // If we have a warm-start, return it immediately (instant feedback)
    if (previousGoodThreshold > 0) {
        qDebug() << "Using warm-start threshold:" << previousGoodThreshold;
        return previousGoodThreshold;
    }

    // Otherwise compute global Otsu (fast: single histogram pass)
    int suggestedThreshold = computeOtsuThreshold16(slices16, 0.1);
    qDebug() << "Computed quick threshold via global Otsu:" << suggestedThreshold;
    return suggestedThreshold;
}

QVector<int> ImageProcessor::computeAdaptiveThresholdsPerSlice(
    const QVector<cv::Mat>& slices16,
    int globalThreshold,
    double globalWeight
) {
    QVector<int> thresholds;
    thresholds.reserve(slices16.size());

    globalWeight = qBound(0.5, globalWeight, 0.9);

    for (const cv::Mat& slice : slices16) {
        if (slice.empty()) {
            thresholds.append(globalThreshold);
            continue;
        }

        const int localThreshold = otsuSliceLocal(slice, 30);

        // Blend: favor global (more stable) but allow local to influence
        const double localWeight = 1.0 - globalWeight;
        int blended = int(std::round(globalWeight * globalThreshold + localWeight * localThreshold));
        blended = qBound(0, blended, 65535);
        thresholds.append(blended);
    }

    return thresholds;
}

bool ImageProcessor::checkOccupancyAndSuggestCorrection(
    const QVector<cv::Mat>& masks8,
    const QVector<cv::Mat>& slices16,
    int currentThreshold,
    bool foregroundIsDark,
    double lowBound,
    double highBound,
    int* outSuggestedThreshold
) {
    if (masks8.isEmpty() || slices16.isEmpty()) {
        return false;
    }

    qint64 totalForeground = 0;
    qint64 totalPixels = 0;

    for (const cv::Mat& mask : masks8) {
        if (mask.empty()) continue;
        totalForeground += cv::countNonZero(mask);
        totalPixels += mask.rows * mask.cols;
    }

    if (totalPixels <= 0) {
        return false;
    }

    const double occupancy = 100.0 * double(totalForeground) / double(totalPixels);

    // Check if correction is needed
    if (occupancy >= lowBound && occupancy <= highBound) {
        return false;  // Occupancy is acceptable
    }

    // Compute global Otsu to use as adjustment baseline
    int otsuThresh = computeOtsuThreshold16(slices16);

    if (outSuggestedThreshold) {
        if (occupancy < lowBound) {
            // Too sparse: loosen threshold to include more foreground.
            const double ratio = lowBound / qMax(0.1, occupancy);
            const double step = qBound(0.02, 0.15 * std::log(ratio + 1.0), 0.25);
            const int delta = qMax(256, int(currentThreshold * step));
            int adjusted = foregroundIsDark
                ? qBound(0, currentThreshold + delta, 65535)
                : qBound(0, currentThreshold - delta, 65535);

            if (foregroundIsDark) {
                adjusted = qMax(adjusted, otsuThresh);
            } else {
                adjusted = qMin(adjusted, otsuThresh);
            }
            *outSuggestedThreshold = adjusted;
        } else {
            // Too dense: tighten threshold to include less foreground.
            const double ratio = occupancy / qMin(99.9, highBound);
            const double step = qBound(0.02, 0.15 * std::log(ratio + 1.0), 0.25);
            const int delta = qMax(256, int(currentThreshold * step));
            int adjusted = foregroundIsDark
                ? qBound(0, currentThreshold - delta, 65535)
                : qBound(0, currentThreshold + delta, 65535);

            if (foregroundIsDark) {
                adjusted = qMin(adjusted, otsuThresh);
            } else {
                adjusted = qMax(adjusted, otsuThresh);
            }
            *outSuggestedThreshold = adjusted;
        }
    }

    return true;  // Correction recommended
}

cv::Mat ImageProcessor::softCleanupMask16(
    const cv::Mat& maskSlice,
    int sliceIndex,
    int totalSlices,
    double confidenceLevel
) {
    return cleanupBinaryMaskSoft(maskSlice, sliceIndex, totalSlices, confidenceLevel);
}

QString ImageProcessor::metricsToString(const ProcessingMetrics& metrics) {
    QString result = QString(
        "Threshold %1 | Adaptive %2 | %3-foreground | Occupancy %4% | Components %5 | "
        "Polarity %6 | Adjustment %7x"
    )
        .arg(metrics.globalThreshold)
        .arg(metrics.adaptiveThreshold)
        .arg(metrics.foregroundIsDark ? "Dark" : "Bright")
        .arg(metrics.occupancy, 0, 'f', 2)
        .arg(metrics.keptComponentCount)
        .arg(metrics.polarityLocked)
        .arg(metrics.autoAdjustmentRatio, 0, 'f', 2);

    if (metrics.slicesTooSparse > 0) {
        result.append(QString(" | %1 sparse slices").arg(metrics.slicesTooSparse));
    }
    if (metrics.slicesTooDense > 0) {
        result.append(QString(" | %1 dense slices").arg(metrics.slicesTooDense));
    }

    return result;
}

void ImageProcessor::saveLastGoodThreshold(int threshold) {
    lastGoodThreshold = threshold;
    QSettings settings;
    settings.setValue("ImageProcessor/lastGoodThreshold", threshold);
    qDebug() << "Saved good threshold for warm-start:" << threshold;
}

int ImageProcessor::loadLastGoodThreshold() const {
    QSettings settings;
    return settings.value("ImageProcessor/lastGoodThreshold", 0).toInt();
}

QVector<cv::Mat> ImageProcessor::processSlices16Adaptive(
    const QVector<cv::Mat>& slices16,
    int previousGoodThreshold,
    const AdaptiveSettings& settings,
    ProcessingMetrics* outMetrics,
    const ProgressCallback& progressCallback
) {
    QVector<cv::Mat> resultMasks;

    if (slices16.isEmpty()) {
        qWarning() << "processSlices16Adaptive: no input slices";
        return resultMasks;
    }

    qDebug() << "=== 16-bit Adaptive Processing Started ===";

    auto reportProgress = [&](int value, const QString& message) {
        if (progressCallback) {
            progressCallback(qBound(0, value, 100), message);
        }
    };
    reportProgress(2, "Analyzing 16-bit stack statistics...");

    // ===== Step 1: Polarity Detection (locked for entire volume) =====
    bool foregroundIsDark = detectForegroundPolarity(slices16);
    QString polaritySource = "auto-locked";
    if (settings.polarityMode == 1) {
        foregroundIsDark = true;
        polaritySource = "forced-dark";
    } else if (settings.polarityMode == 2) {
        foregroundIsDark = false;
        polaritySource = "forced-bright";
    }
    qDebug() << "Polarity mode:" << polaritySource
             << "->" << (foregroundIsDark ? "dark-foreground" : "bright-foreground");
    reportProgress(8, "Computing adaptive thresholds...");

    // ===== Step 2: Global Threshold (warm-start guarded by fresh Otsu) =====
    const int otsuThreshold = computeOtsuThreshold16(slices16, 0.1);
    int globalThreshold = otsuThreshold;
    if (previousGoodThreshold > 0) {
        const int diff = qAbs(previousGoodThreshold - otsuThreshold);
        if (diff <= 2048) {
            globalThreshold = previousGoodThreshold;
            qDebug() << "Warm-start accepted:" << previousGoodThreshold
                     << "(close to Otsu" << otsuThreshold << ")";
        } else {
            // Blend toward current data to avoid stale-threshold lock-in across different datasets.
            globalThreshold = int(std::llround(0.80 * double(otsuThreshold) + 0.20 * double(previousGoodThreshold)));
            qDebug() << "Warm-start differs from Otsu by" << diff
                     << "-> blended threshold:" << globalThreshold
                     << "(warm" << previousGoodThreshold << ", otsu" << otsuThreshold << ")";
        }
    } else {
        qDebug() << "Using fresh Otsu threshold:" << otsuThreshold;
    }

    // ===== Step 3: Adaptive Per-Slice Thresholds =====
    QVector<int> adaptiveThresholds;
    if (settings.useUniformThreshold) {
        adaptiveThresholds.reserve(slices16.size());
        for (int i = 0; i < slices16.size(); ++i) {
            adaptiveThresholds.append(globalThreshold);
        }
        qDebug() << "Uniform threshold mode enabled; reusing one threshold for all slices:" << globalThreshold;
    } else {
        adaptiveThresholds = computeAdaptiveThresholdsPerSlice(
            slices16,
            globalThreshold,
            settings.globalWeight
        );
    }

    // ===== Step 4: Apply Thresholding with Locked Polarity =====
    qint64 totalFilledVoxels = 0;
    qint64 totalVoxels = 0;
    int slicesTooSparse = 0;
    int slicesTooDense = 0;
    int totalKeptComponents = 0;

    const int sliceCount = slices16.size();

    reportProgress(14, "Thresholding and cleaning slices...");
    for (int z = 0; z < sliceCount; ++z) {
        const cv::Mat& slice16 = slices16[z];
        if (slice16.empty()) {
            resultMasks.append(cv::Mat());
            continue;
        }

        cv::Mat u16;
        if (slice16.type() == CV_16UC1) {
            u16 = slice16;
        } else {
            slice16.convertTo(u16, CV_16UC1);
        }

        const int threshold = adaptiveThresholds[z];
        const quint16 threshVal = static_cast<quint16>(qBound(0, threshold, 65535));

        // Vectorized thresholding is significantly faster than manual per-pixel loops.
        cv::Mat mask;
        cv::compare(u16, cv::Scalar::all(threshVal), mask,
                    foregroundIsDark ? cv::CMP_LE : cv::CMP_GE);

        // Soft cleanup with confidence adjustment
        const double sliceOccupancy = double(cv::countNonZero(mask)) / double(mask.rows * mask.cols);
        double confidence = 0.8;  // Default
        if (sliceOccupancy < 0.02) {
            confidence = 0.5;  // Low confidence if very sparse
            ++slicesTooSparse;
        } else if (sliceOccupancy > 0.95) {
            confidence = 0.6;  // Lower confidence if too dense
            ++slicesTooDense;
        }

        cv::Mat cleaned = softCleanupMask16(mask, z, sliceCount, confidence);

        totalKeptComponents += (cv::countNonZero(cleaned) > 0) ? 1 : 0;

        totalFilledVoxels += cv::countNonZero(cleaned);
        totalVoxels += cleaned.rows * cleaned.cols;

        resultMasks.append(cleaned);

        if ((z % 8) == 0 || z + 1 == sliceCount) {
            const int pct = 14 + (((z + 1) * 52) / qMax(1, sliceCount));
            reportProgress(pct, QString("Adaptive segmentation %1/%2 slices")
                                 .arg(z + 1)
                                 .arg(sliceCount));
        }

    }

    const double occupancy = (totalVoxels > 0) ? (100.0 * double(totalFilledVoxels) / double(totalVoxels)) : 0.0;

    // ===== Step 5: Safety Correction (one-pass retry if needed) =====
    int finalThreshold = globalThreshold;
    double autoAdjustmentRatio = 1.0;

    // Adjust occupancy bounds if preserving grayscale structures
    double effectiveLowBound = settings.lowOccupancyThreshold;
    double effectiveHighBound = settings.highOccupancyThreshold;
    
    if (settings.preserveGrayscale && settings.highOccupancyThreshold > 90.0) {
        // For grayscale objects, relax the bounds to avoid aggressive pruning
        effectiveLowBound = qMax(0.1, settings.lowOccupancyThreshold * 0.5);  // Allow sparser structures
        effectiveHighBound = qMin(99.9, settings.highOccupancyThreshold * 1.2); // Allow denser structures
        qDebug() << "Grayscale preservation enabled: relaxed occupancy bounds to [" 
                 << effectiveLowBound << "%, " << effectiveHighBound << "%]";
    }

    reportProgress(68, "Validating occupancy and continuity...");

    if (checkOccupancyAndSuggestCorrection(
            resultMasks,
            slices16,
            globalThreshold,
            foregroundIsDark,
            effectiveLowBound,
            effectiveHighBound,
            &finalThreshold)) {

        // Occupancy out of bounds, retry once with adjusted threshold
        qDebug() << "Occupancy out of bounds (" << occupancy << "%). Auto-adjusting threshold.";

        if (finalThreshold != globalThreshold) {
            autoAdjustmentRatio = double(finalThreshold) / double(qMax(1, globalThreshold));

            // Recompute per-slice thresholds with adjusted global
            if (settings.useUniformThreshold) {
                adaptiveThresholds.clear();
                adaptiveThresholds.reserve(slices16.size());
                for (int i = 0; i < slices16.size(); ++i) {
                    adaptiveThresholds.append(finalThreshold);
                }
            } else {
                adaptiveThresholds = computeAdaptiveThresholdsPerSlice(
                    slices16,
                    finalThreshold,
                    settings.globalWeight
                );
            }

            // Re-threshold all slices with new thresholds
            resultMasks.clear();
            totalFilledVoxels = 0;
            totalVoxels = 0;
            slicesTooSparse = 0;
            slicesTooDense = 0;
            totalKeptComponents = 0;

            reportProgress(72, "Applying one-pass safety correction...");
            for (int z = 0; z < sliceCount; ++z) {
                const cv::Mat& slice16 = slices16[z];
                if (slice16.empty()) {
                    resultMasks.append(cv::Mat());
                    continue;
                }

                cv::Mat u16;
                if (slice16.type() == CV_16UC1) {
                    u16 = slice16;
                } else {
                    slice16.convertTo(u16, CV_16UC1);
                }

                const int threshold = adaptiveThresholds[z];
                const quint16 threshVal = static_cast<quint16>(qBound(0, threshold, 65535));

                cv::Mat mask;
                cv::compare(u16, cv::Scalar::all(threshVal), mask,
                            foregroundIsDark ? cv::CMP_LE : cv::CMP_GE);

                const double sliceOccupancy = double(cv::countNonZero(mask)) / double(mask.rows * mask.cols);
                double confidence = 0.8;
                if (sliceOccupancy < 0.02) {
                    confidence = 0.5;
                    ++slicesTooSparse;
                } else if (sliceOccupancy > 0.95) {
                    confidence = 0.6;
                    ++slicesTooDense;
                }

                cv::Mat cleaned = softCleanupMask16(mask, z, sliceCount, confidence);

                totalKeptComponents += (cv::countNonZero(cleaned) > 0) ? 1 : 0;

                totalFilledVoxels += cv::countNonZero(cleaned);
                totalVoxels += cleaned.rows * cleaned.cols;

                resultMasks.append(cleaned);

                if ((z % 10) == 0 || z + 1 == sliceCount) {
                    const int pct = 72 + (((z + 1) * 24) / qMax(1, sliceCount));
                    reportProgress(pct, QString("Safety-corrected segmentation %1/%2 slices")
                                         .arg(z + 1)
                                         .arg(sliceCount));
                }

            }

            qDebug() << "Safety correction applied. New threshold:" << finalThreshold
                     << "adjustment ratio:" << autoAdjustmentRatio;
        }
    }

    const double finalOccupancy = (totalVoxels > 0) ? (100.0 * double(totalFilledVoxels) / double(totalVoxels)) : 0.0;

    // ===== Step 6: Populate Output Metrics =====
    if (outMetrics) {
        outMetrics->globalThreshold = globalThreshold;
        outMetrics->adaptiveThreshold = finalThreshold;
        outMetrics->foregroundIsDark = foregroundIsDark;
        outMetrics->occupancy = finalOccupancy;
        outMetrics->initialOccupancy = occupancy;
        outMetrics->finalOccupancy = finalOccupancy;
        outMetrics->keptComponentCount = totalKeptComponents;
        outMetrics->polarityLocked = polaritySource;
        outMetrics->autoAdjustmentRatio = autoAdjustmentRatio;
        outMetrics->slicesTooSparse = slicesTooSparse;
        outMetrics->slicesTooDense = slicesTooDense;
        outMetrics->perSliceThresholds = adaptiveThresholds;
        outMetrics->safetyCorectionApplied = (finalThreshold != globalThreshold);
    }

    reportProgress(100, "Adaptive segmentation complete.");

    // Detailed diagnostic logging
    qDebug() << "\n=== 16-BIT ADAPTIVE PROCESSING COMPLETE ===";
    qDebug() << "Polarity: " << (foregroundIsDark ? "DARK-FOREGROUND (object <= threshold)" : "BRIGHT-FOREGROUND (object >= threshold)");
    qDebug() << "Global Otsu threshold: " << globalThreshold;
    qDebug() << "Final threshold: " << finalThreshold << " (adjustment ratio: " << autoAdjustmentRatio << ")";
    qDebug() << "INITIAL occupancy: " << occupancy << "%";
    qDebug() << "FINAL occupancy: " << finalOccupancy << "%";
    qDebug() << "Safety correction applied: " << (finalThreshold != globalThreshold ? "YES" : "NO");
    qDebug() << "Kept components: " << totalKeptComponents;
    qDebug() << "Sparse slices (low occupancy): " << slicesTooSparse;
    qDebug() << "Dense slices (high occupancy): " << slicesTooDense;
    qDebug() << "Total slices: " << sliceCount;
    qDebug() << "Total volume voxels: " << totalVoxels;
    qDebug() << "Filled voxels: " << totalFilledVoxels;
    qDebug() << "=== END PROCESSING ===\n";
    
    qDebug() << metricsToString(outMetrics ? *outMetrics : ProcessingMetrics());

    return resultMasks;
}

// Force moc to generate metacall code
#include "moc_ImageProcessor.cpp"

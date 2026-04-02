#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QObject>
#include <QVector>
#include <opencv2/opencv.hpp>
#include <cstdint>
#include <vector>
#include <cmath>

/**
 * @brief Robust 16-bit image processing pipeline with adaptive thresholding,
 *        polarity detection, and occupancy guardrails.
 *
 * Design principles:
 * - End-to-end 16-bit pipeline: no 8-to-16 conversions during processing
 * - Adaptive local thresholding per slice while maintaining global consistency
 * - Polarity detection locked across entire volume (no slice-by-slice flipping)
 * - Automatic safety correction if occupancy falls outside plausible bounds
 * - Warm-start from previous good thresholds to reduce manual tuning
 */

class ImageProcessor : public QObject {
    Q_OBJECT

public:
    explicit ImageProcessor(QObject* parent = nullptr);

    /**
     * @brief Debug metrics for thresholding and segmentation quality
     */
    struct ProcessingMetrics {
        int globalThreshold = 0;           // Initial threshold from Otsu or previous
        int adaptiveThreshold = 0;         // Final threshold after local adaptation
        bool foregroundIsDark = false;     // Polarity: true if object is darker than background
        double occupancy = 0.0;            // Percentage of foreground pixels (0-100)
        int keptComponentCount = 0;        // Number of connected components retained
        QString polarityLocked = "";       // "locked" or "unlocked"
        double autoAdjustmentRatio = 1.0;  // Adjustment factor if safety correction applied
        int slicesTooSparse = 0;           // Number of slices with low occupancy
        int slicesTooDense = 0;            // Number of slices with high occupancy
        QVector<int> perSliceThresholds;   // Adaptive threshold per slice
        double initialOccupancy = 0.0;     // Occupancy BEFORE safety correction
        double finalOccupancy = 0.0;       // Occupancy AFTER safety correction
        bool safetyCorectionApplied = false; // Whether auto-adjustment was triggered
    };

    /**
     * @brief Settings for adaptive thresholding behavior
     */
    struct AdaptiveSettings {
        double globalWeight = 0.65;          // Blend factor: global vs local (0.5-0.9)
        double localPercentile = 0.3;        // Local thresholding percentile (0.1-0.5)
        bool useOtsuLocal = true;            // Use Otsu for local or simple percentile
        int polarityMode = 0;                // 0=auto, 1=force dark-foreground, 2=force bright-foreground
        int roiBorder = 30;                  // Border to exclude from ROI computation
        double lowOccupancyThreshold = 0.5;  // % below which we consider "too sparse"
        double highOccupancyThreshold = 98.0; // % above which we consider "too dense"
        bool preserveGrayscale = true;       // Relax occupancy bounds for grayscale structures
    };

    /**
     * @brief Process 16-bit slices with adaptive thresholding and warm-start
     *
     * @param slices16 Input 16-bit image slices (CV_16UC1)
     * @param previousGoodThreshold Warm-start threshold from prior dataset (0 = compute new)
     * @param settings Adaptive thresholding configuration
     * @param outMetrics Optional output for debug metrics
     * @return Vector of binarized masks (CV_8UC1)
     */
    QVector<cv::Mat> processSlices16Adaptive(
        const QVector<cv::Mat>& slices16,
        int previousGoodThreshold = 0,
        const AdaptiveSettings& settings = AdaptiveSettings(),
        ProcessingMetrics* outMetrics = nullptr
    );

    /**
     * @brief Detect whether foreground is darker or brighter, sampled across volume
     *
     * @param slices16 Input 16-bit slices
     * @return true if dark-foreground (object <= background), false if bright-foreground
     */
    static bool detectForegroundPolarity(const QVector<cv::Mat>& slices16);

    /**
     * @brief Compute Otsu threshold for 16-bit data from ROI across all slices
     *
     * @param slices16 Input 16-bit slices
     * @param borderCrop Percentage of border to exclude (e.g., 0.1 = exclude 10%)
     * @return Optimal threshold value (0-65535)
     */
    static int computeOtsuThreshold16(
        const QVector<cv::Mat>& slices16,
        double borderCrop = 0.1
    );

    /**
     * @brief Quick threshold suggestion for Auto button (no heavy processing)
     *        Just computes global Otsu with warm-start fallback.
     *        Fast: O(width*height*depth) for histogram only.
     *
     * @param slices16 Input 16-bit slices
     * @param previousGoodThreshold Warm-start from last successful threshold (0 = compute new)
     * @return Recommended threshold value
     */
    static int suggestThresholdQuick(
        const QVector<cv::Mat>& slices16,
        int previousGoodThreshold = 0
    );

    /**
     * @brief Compute per-slice adaptive thresholds using local Otsu + global blend
     *
     * @param slices16 Input 16-bit slices
     * @param globalThreshold Global threshold baseline
     * @param globalWeight Blend factor (0.5-0.9 recommended)
     * @param useOtsuLocal Use Otsu (true) or percentile (false) for local step
     * @return Vector of per-slice thresholds
     */
    static QVector<int> computeAdaptiveThresholdsPerSlice(
        const QVector<cv::Mat>& slices16,
        int globalThreshold,
        double globalWeight = 0.65,
        bool useOtsuLocal = true
    );

    /**
     * @brief Check volume occupancy and suggest automatic threshold adjustment if needed
     *
     * @param masks8 Binarized masks (output of thresholding)
     * @param slices16 Original 16-bit slices (for retry with adjusted threshold)
     * @param currentThreshold Current threshold that produced masks
     * @param foregroundIsDark Polarity flag
     * @param lowBound Acceptable occupancy lower bound (%)
     * @param highBound Acceptable occupancy upper bound (%)
     * @param outSuggestedThreshold Output suggested threshold if adjustment needed
     * @return true if adjustment recommended, false if occupancy is acceptable
     */
    static bool checkOccupancyAndSuggestCorrection(
        const QVector<cv::Mat>& masks8,
        const QVector<cv::Mat>& slices16,
        int currentThreshold,
        bool foregroundIsDark,
        double lowBound = 1.0,
        double highBound = 95.0,
        int* outSuggestedThreshold = nullptr
    );

    /**
     * @brief Soften per-slice cleanup: reduce border suppression, keep more components
     *
     * @param maskSlice Single binary mask (CV_8UC1)
     * @param sliceIndex For profile selection (sparse vs dense)
     * @param totalSlices For determining stack profile
     * @param confidenceLevel 0.0-1.0 controlling how aggressively we prune (higher = more permissive)
     * @return Cleaned up mask
     */
    static cv::Mat softCleanupMask16(
        const cv::Mat& maskSlice,
        int sliceIndex,
        int totalSlices,
        double confidenceLevel = 0.8
    );

    /**
     * @brief Get human-readable summary of processing metrics for logging
     */
    static QString metricsToString(const ProcessingMetrics& metrics);

    /**
     * @brief Store successful threshold for warm-start on next dataset
     */
    void saveLastGoodThreshold(int threshold);

    /**
     * @brief Load previously saved good threshold
     */
    int loadLastGoodThreshold() const;

private:
    int lastGoodThreshold = 0;  // Warm-start storage
};

#endif // IMAGEPROCESSOR_H

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

// MainWindow.h
#pragma once
#include <QMainWindow>
#include <QOpenGLWidget>
#include <QFile>
#include <QProgressBar>
#include <QToolBar>
#include <QDockWidget>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QDir>
#include <QProgressBar>
#include <QSlider>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>          // Add this
#include <QThread>
#include <QVector3D>
#include <QAction>
#include <QCheckBox>       // Add this for material colors
#include <QSpinBox>        // Add this for material thresholds
#include <QPushButton>     // Add this for color pickers
#include "MarchingCubes.h"
#include "ImageProcessor.h"
#include <opencv2/opencv.hpp>
#include <QTimer>
#include <QElapsedTimer>
#include <QTabWidget>
#include <QScrollArea>
#include <QGridLayout>
#include <QMessageBox>
#include <QProgressDialog>
#include <QFutureWatcher>
#include <QPointer>
#include <QSpinBox>
#include <QtConcurrent/QtConcurrent>
#include <functional>


class GLWidget;
class QDockWidget;
class QProgressBar;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
   explicit MainWindow(QWidget* parent = nullptr);
   ~MainWindow();
   static cv::Mat QImageToCvMat(const QImage& img);
   static QImage cvMatToQImage(const cv::Mat& mat);

private slots:
    void openDataset();
    void toggleControls(bool visible);
    void updateThreshold(double value);
    void updateIsoLevel(int value);
    void exportSTL();
    void onThreshold16Changed(int value);
    void onThreshold16Auto();
    void onOtsuComputationFinished();
    void handleMeshGenerationStarted();
    void handleMeshComputationFinished();
    void handleMeshRenderingFinished();
    void appendPreviewThumbnails();
    void finalizeImportUi();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;


private:
    using ProgressCallback = std::function<void(int, const QString&)>;
    void saveProcessedImage(const QImage& image, const QString& filePath);
    void finalizeMeshProgressDialog();
    void createMenu();
    void createToolbar();
    void createCentralWidget();
    void createPreviewTab();
    void addMaterialVisualizationControls();
    void showFullSizeImage(const QImage& image);
    void updateImagePreviews();
    void loadMesh();
    void syncThresholdControls();
    QSpinBox* threshold16SpinBox = nullptr;
    QPushButton* threshold16AutoButton = nullptr;
    QLabel* thresholdLabel = nullptr;
    QAction* generate3DAct;
    double currentThreshold = 0.43;
    static constexpr int kDefaultThreshold16 = 28180;
    int currentThreshold16 = 28180;
    QTabWidget* mainTabs;
    QWidget* previewTab;
    QScrollArea* previewScroll;
    QWidget* previewContainer;
    QGridLayout* previewLayout;
    QVector<QImage> previewBuildImages;
    int previewBuildIndex = 0;
    bool previewBuildInProgress = false;
    QTimer previewBuildTimer;
    QPointer<QProgressDialog> activeImportDialog;
    // void createDockPanels();
    void loadImages(const QStringList& filePaths);
    // QImage loadImage(const QString& path);  // Add this line
    // void handleLoadedImages(const QVector<QImage>& images);
    QImage preprocessLoadedImage(const QImage& input, const QString& path, double thresholdOverride = -1.0, int totalSlicesOverride = -1);
    QImage loadMemoryMappedImage(const QString& path); // Add this line
    // Add this typedef
    using VolumeData = QVector<QVector<QVector<float>>>;
    using VolumeData16 = QVector<QVector<QVector<quint16>>>;
    // Declare the function
    VolumeData convertToVolume(const QVector<QImage>& images, const ProgressCallback& progressCallback = ProgressCallback());
    VolumeData convertToVolume16(const QVector<cv::Mat>& slices16, int previousGoodThreshold = 0, float smoothingIntensity = 1.0f, const ProgressCallback& progressCallback = ProgressCallback());
    static quint16 computeOtsuThreshold16(const VolumeData16& volume);
    void updateLoadingDialog(int progress, const QString& message);
    void generateMesh();
    VolumeData currentVolume;
    VolumeData16 currentSegmentationVolume16;
    MarchingCubes::Mesh currentMesh;
    QSlider *isoSlider; // Add this line
    QSlider *threshSlider; // Add this line
    QElapsedTimer isoChangeTimer;
    QElapsedTimer importTimer;
    QElapsedTimer meshTimer;
    QElapsedTimer stlTimer;
    QElapsedTimer meshLoadTimer;
    QVector<QImage> loadedImages; // Store loaded images
    QVector<cv::Mat> segmentationSlices16; // Reserved for 16-bit segmentation path.
    QAction* exportSTLAct;
    GLWidget* glWidget;
    QToolBar* toolBar = nullptr;  // Main toolbar for actions and controls
    QDockWidget* controlsDock;
    QProgressBar *progressBar;
    QLabel* imageLabel;
    bool authenticated = false;
    QTimer *inactivityTimer;
    QVector<QImage> originalImages;  // Add this line
    QStringList currentImagePaths;
    bool hasSegmentation16Data = false;
    QProgressDialog* loadingDialog;
    QFutureWatcher<void> meshGenerationWatcher;
    QFutureWatcher<int> otsuWatcher;
    bool isGeneratingMesh = false;
    bool waitingForMeshUploadCompletion = false;
    bool isOtsuRunning = false;
    bool isExportingStl = false;
    MarchingCubes::Mesh pendingMesh; // Store mesh while waiting for OpenGL

    // Default physical voxel spacing used for geometry scaling.
    // Tune to your scanner metadata when available.
    float voxelSpacingX = 0.30f;
    float voxelSpacingY = 0.30f;
    float voxelSpacingZ = 0.50f;

    // Adaptive image processing
    ImageProcessor imageProcessor;
    ImageProcessor::ProcessingMetrics lastMetrics;

    // Material visualization and segmentation
    bool materialColorsEnabled = false;
    // UI Controls for material visualization
    QCheckBox* materialColorsCheckBox = nullptr;


    // Material segmentation methods
    VolumeData16 buildRawVolume16(const QVector<cv::Mat>& slices16) const;
    void estimateMaterialThresholds(const VolumeData16& volume16, int& ceramicThreshold16, int& boneThreshold16) const;
    MarchingCubes::Mesh classifyMeshByMaterial(const MarchingCubes::Mesh& inputMesh,
                                                const VolumeData16& volume16);
    void onMaterialColorsToggled(bool enabled);

};
#endif // MAINWINDOW_H

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
#include "MarchingCubes.h"
#include <opencv2/opencv.hpp>
#include <QTimer>
#include <QTabWidget>
#include <QScrollArea>
#include <QGridLayout>
#include <QMessageBox>
#include <QProgressDialog>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>


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
    void onThresholdChanged(int value);
    void onThresholdApply();
    void handleMeshGenerationStarted();
    void handleMeshComputationFinished();
    void handleMeshRenderingFinished();

protected:
    bool eventFilter(QObject* watched, QEvent* event) override;


private:
    void saveProcessedImage(const QImage& image, const QString& filePath);
    void createMenu();
    void createToolbar();
    void createCentralWidget();
    void createPreviewTab();
    void showFullSizeImage(const QImage& image);
    void updateImagePreviews();
    void loadMesh();
    QSlider* thresholdSlider;
    QLabel* thresholdLabel;
    QPushButton* thresholdApplyButton;
    double pendingThreshold = 0.35;  // Renamed from currentThreshold
    double currentThreshold = 0.35; // Default threshold
    QTabWidget* mainTabs;
    QWidget* previewTab;
    QScrollArea* previewScroll;
    QWidget* previewContainer;
    QGridLayout* previewLayout;
    // void createDockPanels();
    void loadImages(const QStringList& filePaths);
    // QImage loadImage(const QString& path);  // Add this line
    // void handleLoadedImages(const QVector<QImage>& images);
    QImage loadMemoryMappedImage(const QString& path); // Add this line
    // Add this typedef
    using VolumeData = QVector<QVector<QVector<float>>>;
    // Declare the function
    VolumeData convertToVolume(const QVector<QImage>& images);
    void generateMesh();
    VolumeData currentVolume;
    MarchingCubes::Mesh currentMesh;
    QSlider *isoSlider; // Add this line
    QSlider *threshSlider; // Add this line
    QElapsedTimer isoChangeTimer;
    QVector<QImage> loadedImages; // Store loaded images
    QAction* exportSTLAct;
    GLWidget* glWidget;
    QDockWidget* controlsDock;
    QProgressBar *progressBar;
    QLabel* imageLabel;
    bool authenticated = false;
    QTimer *inactivityTimer;
    QVector<QImage> originalImages;  // Add this line
    QStringList currentImagePaths;
    QProgressDialog* loadingDialog;
    QFutureWatcher<void> meshGenerationWatcher;
    bool isGeneratingMesh = false;
    MarchingCubes::Mesh pendingMesh; // Store mesh while waiting for OpenGL


};
#endif // MAINWINDOW_H

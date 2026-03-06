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
    loadingDialog->setLabelText("Generating 3D mesh...");
    loadingDialog->setCancelButton(nullptr);
    loadingDialog->setRange(0, 0);
    loadingDialog->setMinimumDuration(0);
    loadingDialog->reset();

    connect(&meshGenerationWatcher, &QFutureWatcher<void>::started,
            this, &MainWindow::handleMeshGenerationStarted);
    connect(&meshGenerationWatcher, &QFutureWatcher<void>::finished,
            this, &MainWindow::handleMeshComputationFinished);
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
    QAction* generateAct = createAction("Generate 3D", ":icons/icons/cube.png");
    QAction* resetAct = createAction("Reset View", ":icons/icons/reset.png");
    QAction* loadMeshAct = createAction("Load Mesh", ":icons/icons/up-loading.png");
    exportSTLAct = new QAction(QIcon(":icons/icons/material.png"), "Export STL", this);  // New export

    // Add to toolbar with spacers
    toolBar->addAction(openAct);
    toolBar->addSeparator();
    toolBar->addAction(generateAct);
    toolBar->addSeparator();
    toolBar->addAction(resetAct);
    toolBar->addSeparator();
    toolBar->addAction(loadMeshAct);
    toolBar->addSeparator();
    toolBar->addAction(exportSTLAct);  // Add export button

    // Connect signals
    connect(openAct, &QAction::triggered, this, &MainWindow::openDataset);
    connect(generateAct, &QAction::triggered, this, &MainWindow::generateMesh);
    connect(resetAct, &QAction::triggered, [this]() { glWidget->resetView(); });
    connect(loadMeshAct, &QAction::triggered, this, &MainWindow::loadMesh);
    connect(exportSTLAct, &QAction::triggered, this, &MainWindow::exportSTL);

    // Threshold controls
    thresholdLabel = new QLabel("Threshold: 35%", this);
    thresholdSlider = new QSlider(Qt::Horizontal, this);
    thresholdSlider->setRange(0, 100);
    thresholdSlider->setValue(35);
    thresholdSlider->setFixedWidth(150);

    thresholdApplyButton = new QPushButton("Apply", this);
    thresholdApplyButton->setFixedWidth(80);

    toolBar->addSeparator();
    toolBar->addWidget(thresholdLabel);
    toolBar->addWidget(thresholdSlider);
    toolBar->addWidget(thresholdApplyButton);

    connect(thresholdSlider, &QSlider::valueChanged, this, &MainWindow::onThresholdChanged);
    connect(thresholdApplyButton, &QPushButton::clicked, this, &MainWindow::onThresholdApply);
}


void MainWindow::openDataset() {
    QFileDialog dialog(this);
    dialog.setWindowTitle("Load MRI Dataset");
    dialog.setFileMode(QFileDialog::Directory); // Allow selecting a directory
    dialog.setOption(QFileDialog::ShowDirsOnly, true); // Only show directories

    if (dialog.exec() == QDialog::Accepted) {
        QString folderPath = dialog.selectedFiles().first(); // Get the selected folder path
        QDir dir(folderPath);

        // Filter for image files
        QStringList filters = {"*.bmp", "*.dcm", "*.png", "*.jpg", "*.tif"};
        QStringList filePaths = dir.entryList(filters, QDir::Files, QDir::Name);

        // Prepend folder path to each file name
        for (QString& file : filePaths) {
            file = dir.filePath(file);
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
    loadedImages.clear();
    originalImages.clear();

    // Create a progress dialog
    QProgressDialog progressDialog("Loading images...", "Cancel", 0, filePaths.size(), this);
    progressDialog.setWindowTitle("Loading Progress");
    progressDialog.setWindowModality(Qt::WindowModal); // Block interaction with main window
    progressDialog.setMinimumDuration(0); // Show immediately
    progressDialog.show();

    for (int i = 0; i < filePaths.size(); ++i) {
        // Check if the user canceled the operation
        if (progressDialog.wasCanceled()) {
            loadedImages.clear();
            statusBar()->showMessage("Loading canceled.", 3000);
            return;
        }

        QImage originalImg;
        if (!originalImg.load(filePaths[i])) {
            qWarning() << "Failed to load image:" << filePaths[i];
            continue;
        }
        originalImages.append(originalImg);

        QImage img = loadMemoryMappedImage(filePaths[i]);
        if (!img.isNull()) {
            loadedImages.append(img);
        }

        // Update progress
        progressDialog.setValue(i + 1);
        progressDialog.setLabelText(QString("Loading image %1/%2").arg(i+1).arg(filePaths.size()));
        qApp->processEvents(); // Keep the UI responsive
    }

    progressDialog.close();

    if (!loadedImages.isEmpty()) {
        statusBar()->showMessage("Images loaded successfully.", 3000);
        updateImagePreviews();
        mainTabs->setCurrentIndex(1);  // Switch to preview tab
    } else {
        statusBar()->showMessage("No valid images loaded.", 3000);
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

QImage MainWindow::loadMemoryMappedImage(const QString& path) {
    QImage img;
    if (!img.load(path)) {
        qWarning() << "Failed to load image:" << path;
        return QImage();
    }

    // Step 2: Crop the image
    QRect cropRect(45.5, 48.5, 1417, 537);
    QImage croppedImage = img.copy(cropRect);
    croppedImage.save("qt_cropped_image_001.bmp");

    // Step 3: Edge detection
    cv::Mat cvImg = QImageToCvMat(croppedImage);
    cv::Mat edges;
    cv::Canny(cvImg, edges, 0, 0.001 * 255);
    QImage edgeImage = cvMatToQImage(edges);
    edgeImage.save("qt_edge_image_001.bmp");

    // Step 4: Threshold the grayscale image to create a binary mask
    cv::Mat binaryMask;
    cv::threshold(cvImg, binaryMask, currentThreshold * 255, 255, cv::THRESH_BINARY);
    QImage binaryMaskImage = cvMatToQImage(binaryMask);
    binaryMaskImage.save("qt_binary_mask_001.bmp");

    // Step 5: Overlay edges on the original image
    cv::Mat overlay;
    cv::cvtColor(cvImg, overlay, cv::COLOR_GRAY2BGR);
    overlay.setTo(cv::Scalar(0, 255, 0), edges); // Green edges
    QImage overlayImage = cvMatToQImage(overlay);
    overlayImage.save("qt_overlay_image_001.bmp");

    // Step 6: Apply the binary mask to the overlay to retain only bone regions
    cv::Mat finalOverlay;
    overlay.copyTo(finalOverlay, binaryMask); // Retain only bone regions with green edges
    QImage finalOverlayImage = cvMatToQImage(finalOverlay);
    finalOverlayImage.save("qt_final_overlay_image_001.bmp");

    // Step 7: Combine the thresholded image (black and white) with the overlay
    cv::Mat finalImage;
    cv::cvtColor(binaryMask, finalImage, cv::COLOR_GRAY2BGR); // Convert binary mask to BGR
    finalOverlay.copyTo(finalImage, finalOverlay); // Add green edges to the final image
    QImage processedImage = cvMatToQImage(finalImage);
    processedImage.save("qt_final_image_001.bmp");

    return processedImage;
}

MainWindow::VolumeData MainWindow::convertToVolume(const QVector<QImage>& images) {
    VolumeData volume;

    if (images.isEmpty()) {
        qWarning() << "No images to convert";
        return volume;
    }

    // Get dimensions from first image
    const int width = images.first().width();
    const int height = images.first().height();
    const int depth = images.size();

    qDebug() << "Converting images to volume. Dimensions:"
             << width << "x" << height << "x" << depth;

    // Validate dimensions
    for (const QImage& img : images) {
        if (img.width() != width || img.height() != height) {
            qCritical() << "Image dimensions mismatch! Expected"
                        << width << "x" << height
                        << "got" << img.width() << "x" << img.height();
            return VolumeData();
        }
    }

    // Allocate 3D array [z][y][x]
    volume.resize(depth);
    for (int z = 0; z < depth; z++) {
        volume[z].resize(height);
        const QImage& img = images[z];

        for (int y = 0; y < height; y++) {
            volume[z][y].resize(width);
            for (int x = 0; x < width; x++) {
                // Get pixel value (0 to 255)
                QRgb pixel = img.pixel(x, y);
                float val = qGray(pixel); // Grayscale value (0 to 255)

                // Apply threshold (0.43 * 255)
                if (val >= 0.43 * 255) {
                    volume[z][y][x] = 1.0f; // Set to 1 if above or equal to threshold
                } else {
                    volume[z][y][x] = 0.0f; // Set to 0 if below threshold
                }
            }
        }
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

    // Save the 3D volume to a text file for human-readable comparison
    QFile volumeTextFile("qt_volume.txt");
    if (volumeTextFile.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&volumeTextFile);
        for (int z = 0; z < volume.size(); ++z) {
            for (int y = 0; y < volume[z].size(); ++y) {
                for (int x = 0; x < volume[z][y].size(); ++x) {
                    // Write binary values (0 or 1)
                    out << static_cast<int>(volume[z][y][x]) << " ";
                }
                out << "\n";
            }
            out << "\n";
        }
        volumeTextFile.close();
    }

    qDebug() << "Volume created and saved successfully.";
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
     currentMesh = MarchingCubes::generateMesh(currentVolume, isoLevel);
     qDebug() << "Generated mesh:"
              << currentMesh.vertices.size() << "vertices,"
              << currentMesh.indices.size() / 3 << "triangles";

     qDebug() << "Updating GLWidget...";
     setCentralWidget(glWidget); // Force GLWidget to be visible
     glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
     glWidget->update();

     qDebug() << "Exiting updateIsoLevel";
 }


 // Convert QImage to cv::Mat
 cv::Mat MainWindow::QImageToCvMat(const QImage& img) {
     cv::Mat mat(img.height(), img.width(), CV_8UC1, (void*)img.bits(), img.bytesPerLine());
     return mat.clone(); // Ensure deep copy
 }

 // Convert cv::Mat to QImage
 QImage MainWindow::cvMatToQImage(const cv::Mat& mat) {
     if (mat.type() == CV_8UC1) {
         // Grayscale
         return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8);
     } else if (mat.type() == CV_8UC3) {
         // RGB
         return QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888).rgbSwapped();
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

     isGeneratingMesh = true;
     statusBar()->showMessage("Starting mesh generation...");

     QFuture<void> future = QtConcurrent::run([this]() {
         // 1. CPU-intensive computation in background
         auto volume = convertToVolume(loadedImages);
         if (!volume.isEmpty()) {
             pendingMesh = MarchingCubes::generateMesh(volume, 1.0f);
         }
     });

     meshGenerationWatcher.setFuture(future);
 }

 void MainWindow::handleMeshGenerationStarted() {
     loadingDialog->show();
     QApplication::processEvents();
 }

 void MainWindow::handleMeshComputationFinished() {
     if (!pendingMesh.vertices.isEmpty()) {
         currentMesh = pendingMesh; // Update currentMesh for export
         glWidget->updateMesh(currentMesh.vertices, currentMesh.indices);
     } else {
         handleMeshRenderingFinished();
     }
 }

 void MainWindow::handleMeshRenderingFinished() {
     isGeneratingMesh = false;
     loadingDialog->hide();

     if (pendingMesh.vertices.isEmpty()) {
         QMessageBox::warning(this, "Error", "Failed to generate mesh.");
     } else {
         statusBar()->showMessage("Mesh generation completed", 3000);
     }
     pendingMesh = MarchingCubes::Mesh(); // Clear stored mesh
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


 void MainWindow::onThresholdChanged(int value) {
     pendingThreshold = value / 100.0;
     thresholdLabel->setText(QString("Threshold: %1%").arg(value));
 }

 void MainWindow::onThresholdApply() {
     if (qFuzzyCompare(currentThreshold, pendingThreshold)) {
         return;  // No change needed
     }

     currentThreshold = pendingThreshold;
     statusBar()->showMessage("Applying new threshold...", 3000);

     if (!currentImagePaths.isEmpty()) {
         // Reprocess images with new threshold
         loadedImages.clear();
         QProgressDialog progress("Re-processing images...", "Cancel", 0, currentImagePaths.size(), this);

         for (int i = 0; i < currentImagePaths.size(); ++i) {
             if (progress.wasCanceled()) break;
             loadedImages.append(loadMemoryMappedImage(currentImagePaths[i]));
             progress.setValue(i + 1);
             qApp->processEvents();
         }

         // Regenerate 3D model
         generateMesh();
     }
 }

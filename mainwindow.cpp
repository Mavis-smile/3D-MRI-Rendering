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
};

struct ProcessedSliceResult {
    int index = -1;
    bool valid = false;
    QImage processed;
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
    thresholdLabel = new QLabel("Threshold: 43%", this);
    thresholdSlider = new QSlider(Qt::Horizontal, this);
    thresholdSlider->setRange(0, 100);
    thresholdSlider->setValue(43);
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
    loadedImages.clear();
    originalImages.clear();
    loadedImages.reserve(filePaths.size());
    originalImages.reserve(filePaths.size());

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
    for (const LoadedSliceResult& result : orderedResults) {
        if (!result.valid) {
            ++failedCount;
            continue;
        }
        originalImages.append(result.original);
        loadedImages.append(result.processed);
    }

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
            ? QString("Loaded %1 image(s). %2 file(s) could not be decoded.")
                .arg(loadedImages.size()).arg(failedCount)
            : QString("Images loaded successfully (%1 slices).").arg(loadedImages.size());
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
    const double xySigma = sparseDepth ? 0.65 : (denseDepth ? 2.3 : 1.4);
    const cv::Size xyKernel = sparseDepth ? cv::Size(3, 3) : cv::Size(5, 5);
    qDebug() << "Volume profile" << depthProfile
             << "XY sigma" << xySigma
             << "spacing" << voxelSpacingX << voxelSpacingY << voxelSpacingZ;
    for (int z = 0; z < depth; ++z) {
        cv::Mat sliceMat(height, width, CV_32F);
        for (int y = 0; y < height; ++y)
            std::memcpy(sliceMat.ptr<float>(y), volume[z][y].constData(), width * sizeof(float));
        cv::GaussianBlur(sliceMat, sliceMat, xyKernel, xySigma);
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

     isGeneratingMesh = true;
     statusBar()->showMessage("Starting mesh generation...");

     QFuture<void> future = QtConcurrent::run([this]() {
         const int sliceCount = loadedImages.size();
         const float targetIso = (sliceCount <= 48) ? 0.31f : ((sliceCount >= 200) ? 0.36f : 0.42f);
         const QString profile = (sliceCount <= 48) ? "sparse" : ((sliceCount >= 200) ? "dense" : "balanced");
         qDebug() << "Reconstruction profile:" << profile
                  << "slices:" << sliceCount
                  << "iso:" << targetIso
                  << "spacing:" << voxelSpacingX << voxelSpacingY << voxelSpacingZ;

         auto reportProgress = [this](int progress, const QString& message) {
             QMetaObject::invokeMethod(this, [this, progress, message]() {
                 updateLoadingDialog(progress, message);
             }, Qt::QueuedConnection);
         };

         reportProgress(1, "Preparing volume data...");

         // 1. CPU-intensive computation in background
         auto volume = convertToVolume(loadedImages, reportProgress);
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

     if (!originalImages.isEmpty()) {
         // Reprocess images from the already loaded originals to avoid disk re-reads.
         loadedImages.clear();
         loadedImages.reserve(originalImages.size());
         QVector<int> sliceIndices(originalImages.size());
         std::iota(sliceIndices.begin(), sliceIndices.end(), 0);
         const double threshold = currentThreshold;
         const int totalSlices = originalImages.size();
         QProgressDialog progress("Re-processing images...", "Cancel", 0, originalImages.size(), this);
         QFutureWatcher<ProcessedSliceResult> watcher;
         QEventLoop waitLoop;
         connect(&watcher, &QFutureWatcher<ProcessedSliceResult>::finished, &waitLoop, &QEventLoop::quit);
         connect(&watcher, &QFutureWatcher<ProcessedSliceResult>::progressRangeChanged,
                 &progress, &QProgressDialog::setRange);
         connect(&watcher, &QFutureWatcher<ProcessedSliceResult>::progressValueChanged,
                 &progress, &QProgressDialog::setValue);
         connect(&watcher, &QFutureWatcher<ProcessedSliceResult>::progressValueChanged,
                 this, [this, &progress, totalSlices](int value) {
                     progress.setLabelText(QString("Re-processing image %1/%2").arg(value).arg(totalSlices));
                 });
         connect(&progress, &QProgressDialog::canceled, &watcher, &QFutureWatcher<ProcessedSliceResult>::cancel);

         watcher.setFuture(QtConcurrent::mapped(sliceIndices, [this, threshold, totalSlices](int index) -> ProcessedSliceResult {
             ProcessedSliceResult result;
             result.index = index;
             result.processed = preprocessLoadedImage(originalImages[index], currentImagePaths.value(index), threshold, totalSlices);
             result.valid = !result.processed.isNull();
             return result;
         }));

         waitLoop.exec();

         if (progress.wasCanceled()) {
             loadedImages.clear();
             statusBar()->showMessage("Re-processing canceled.", 3000);
             return;
         }

         const QList<ProcessedSliceResult> results = watcher.future().results();
         QVector<ProcessedSliceResult> orderedResults(originalImages.size());
         for (const ProcessedSliceResult& result : results) {
             if (result.index >= 0 && result.index < orderedResults.size()) {
                 orderedResults[result.index] = result;
             }
         }

         for (const ProcessedSliceResult& result : orderedResults) {
             if (result.valid) {
                 loadedImages.append(result.processed);
             }
         }

         // Regenerate 3D model
         generateMesh();
     }
 }

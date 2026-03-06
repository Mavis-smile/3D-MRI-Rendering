// // ImageProcessor.cpp
// #include <opencv2/opencv.hpp>

// QVector<cv::Mat> ImageProcessor::processStack(
//     const QStringList& filePaths,
//     double cannyThreshold,
//     double binaryThreshold
//     ) {
//     QVector<cv::Mat> processedStack;

//     foreach (const QString& path, filePaths) {
//         cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_GRAYSCALE);
//         cv::Mat edges;
//         cv::Canny(img, edges, cannyThreshold, cannyThreshold*3);
//         cv::threshold(edges, edges, binaryThreshold*255, 255, cv::THRESH_BINARY);
//         processedStack.append(edges);
//     }

//     return processedStack;
// }

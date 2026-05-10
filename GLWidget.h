#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QVector3D>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QFutureWatcher>

inline bool operator<(const QVector3D& a, const QVector3D& b) {
    if (a.x() != b.x()) return a.x() < b.x();
    if (a.y() != b.y()) return a.y() < b.y();
    return a.z() < b.z();
}


class GLWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT

public:
    explicit GLWidget(QWidget *parent = nullptr);
    ~GLWidget();

    void updateMesh(const QVector<QVector3D>& vertices,
                    const QVector<unsigned int>& indices);
    
    // New: Extended mesh update with per-vertex colors
    void updateMeshWithMaterials(const QVector<QVector3D>& vertices,
                                 const QVector<unsigned int>& indices,
                                 const QVector<QColor>& vertexColors);
    
    void loadMeshFromFiles(const QString& verticesPath, const QString& facesPath);
    
    // Material visualization controls
    void setMaterialColorsEnabled(bool enabled);
    void setMaterialColor(int materialType, const QColor& color);
    void setMaterialVisibility(int materialType, bool visible);
    bool isMaterialColorsEnabled() const { return materialColorsEnabled; }
    QColor getMaterialColor(int materialType) const;
    bool isMaterialVisible(int materialType) const;



protected:
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;


    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent*) override;
    void keyPressEvent(QKeyEvent* event) override;

signals:
    void meshUpdateComplete();

public slots:
    void resetView() {
        xRot = 0;
        yRot = 0;
        zoom = 1.0f;
        update();
    }

private:
    void createShaders();
    void createGeometry();

    QOpenGLBuffer vbo;
    QOpenGLBuffer ibo;
    QOpenGLBuffer colorVbo;              // Per-vertex color buffer
    QOpenGLVertexArrayObject vao;

    GLuint shaderProgram;
    GLuint modelLoc, viewLoc, projLoc;
    GLuint colorLoc;                    // Color attribute location
    GLuint materialColorsEnabledLoc;    // Uniform for enabling material colors
    GLuint materialColorLoc[3];         // Uniform for per-material colors (Background, Bone, Ceramic)
    GLuint materialVisibilityLoc[3];    // Uniform for per-material visibility

    QMatrix4x4 projection;
    QMatrix4x4 view;
    QMatrix4x4 model;

    QVector3D meshCenter;
    float meshSize;
    QVector3D cameraPosition;

    QPoint lastMousePos;
    float zoom = 1.0f;
    float xRot = 0.0f;
    float yRot = 0.0f;

    size_t indexCount = 0;
    
    // Material visualization state
    bool materialColorsEnabled = false;
    QColor materialColors[3] = {
        QColor(100, 100, 100),   // Background (gray)
        QColor(255, 255, 255),   // Bone (white)
        QColor(50, 130, 255)     // Ceramic (blue)
    };
    bool materialVisibility[3] = { true, true, true };  // All materials visible by default
    QVector<QColor> currentVertexColors;                // Current per-vertex colors
    bool hasPerVertexColors = false;                    // Flag indicating if mesh has colors
    
    QFutureWatcher<void> meshExportWatcher;
    QVector<unsigned int> queuedExportIndices;
    QVector<QVector3D> queuedExportVertices;
    bool exportQueued = false;

    void createTestCube();
    void createTestTriangle();
    void queueMeshExport(const QVector<unsigned int>& indices, const QVector<QVector3D>& vertices);
    void startQueuedMeshExport();
    static void exportMeshToFiles(const QVector<unsigned int>& indices, const QVector<QVector3D>& vertices);

};

#endif // GLWIDGET_H

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
    void loadMeshFromFiles(const QString& verticesPath, const QString& facesPath);



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
    QOpenGLVertexArrayObject vao;

    GLuint shaderProgram;
    GLuint modelLoc, viewLoc, projLoc;

    QMatrix4x4 projection;
    QMatrix4x4 view;
    QMatrix4x4 model;

    QVector3D meshCenter; // Add this line
    float meshSize;       // Add this line
    QVector3D cameraPosition; // Camera position for panning

    QPoint lastMousePos;
    float zoom = 1.0f;
    float xRot = 0.0f;
    float yRot = 0.0f;

    size_t indexCount = 0;

    void createTestCube();
    void createTestTriangle();
    void exportMeshToFiles(const QVector<unsigned int>& indices, const QVector<QVector3D>& vertices);

};

#endif // GLWIDGET_H

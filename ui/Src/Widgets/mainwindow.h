#ifndef MAINWINDOW_H
#define MAINWINDOW_H
#include <QMainWindow>
#include <QFrame>
#include <QPalette>
#include <QPushButton>
#include <QAction>
#include <QLabel>

namespace Ui {
class MainWindow;
}

class MeshWidget;
class SuggestiveContour;
class SketchWidget;
class TFModel;
class TFModel2;

enum WidgetType {NOWIDGET, MESH, FRONTSKETCH, SIDESKETCH};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();
    TFModel *sketch2model_;

private:
    Ui::MainWindow *ui;

public Q_SLOTS:
    void    activateMeshWidget();
    void    activateFrontSketchWidget();
    void    activateSideSketchWidget();
    void    generateMesh();
    void    undo();
    void    clear();
    void    renderSketch();
    void    save();

private:
    void    initMainWindow();
    void    initConnection();

    void    keyPressEvent(QKeyEvent *e);

private:
//    QPushButton     *button_sketching_module_;
//    QPushButton     *button_mesh_refine_module_;
    QPushButton     *button_undo_;
    QPushButton     *button_clear_;
    QPushButton     *button_prerender_;
    QPushButton     *button_save_;

    QLabel          *label_mesh_widget_;
    QLabel          *label_front_sketch_widget_;
    QLabel          *label_side_sketch_widget_;
    QLabel          *label_front_view_;
    QLabel          *label_side_view_;

private:
    MeshWidget      *mesh_widget_;
    SuggestiveContour *sketch_renderer_;
    SketchWidget    *front_sketch_widget_;
    SketchWidget    *side_sketch_widget_;

    WidgetType      current_widget_;

private:
    bool    b_refine_mode_;
};

#endif // MAINWINDOW_H

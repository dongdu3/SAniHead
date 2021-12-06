#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QDesktopWidget>
#include <QFileDialog>
#include <QTextCodec>
#include <QTextStream>
#include <QMessageBox>

#include <iostream>

#include "MeshGenerator/tensorflowmodel.h"
#include "MeshRenderer/suggestivecontour.h"
#include "Widgets/meshwidget.h"
#include "Widgets/sketchwidget.h"

#define VIEW_SIZE1 900      // for main view widget
#define VIEW_SIZE2 400      // for side view widget

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    mesh_widget_ = nullptr;
    sketch_renderer_ = nullptr;
    front_sketch_widget_ = nullptr;
    side_sketch_widget_ = nullptr;
    sketch2model_ = nullptr;
    current_widget_ = NOWIDGET;
    b_refine_mode_ = false;

    initMainWindow();
    initConnection();
}

MainWindow::~MainWindow()
{
    if (mesh_widget_)
    {
        delete mesh_widget_;
        mesh_widget_ = nullptr;
    }
    if (sketch_renderer_)
    {
        delete sketch_renderer_;
        sketch_renderer_ = nullptr;
    }
    if (front_sketch_widget_)
    {
        delete front_sketch_widget_;
        front_sketch_widget_ = nullptr;
    }
    if (side_sketch_widget_)
    {
        delete side_sketch_widget_;
        side_sketch_widget_ = nullptr;
    }
    if (sketch2model_)
    {
        delete sketch2model_;
        sketch2model_ = nullptr;
    }

    delete ui;
}

void MainWindow::initMainWindow()
{
    // set mainwindow position and size
    this->setWindowTitle(tr("AnimalHeadSketchModeler"));
    this->setFixedSize(1312, 908);
    QDesktopWidget *desktop = QApplication::desktop();
    int screen_width = desktop->width();
    int screen_height = desktop->height();
    this->move((screen_width/2)-(width()/2), (screen_height/2)-(height()/2)-45);


    // create pushbutton
    QFont font;
    font.setPointSize(20);
//    int w_button = this->width()/2-6;
//    int h_button = 60;
    int space_size = 4;

//    button_sketching_module_ = new QPushButton(tr("Sketching"), this);
//    button_sketching_module_->setFont(font);
//    button_sketching_module_->setToolTip(tr("Sketching module"));
//    button_sketching_module_->setStyleSheet("border:2px groove gray;border-radius:10px;border-style:outset");
//    button_sketching_module_->setGeometry(space_size, space_size, w_button, h_button);

//    button_mesh_refine_module_ = new QPushButton(tr("Refinement"), this);
//    button_mesh_refine_module_->setFont(font);
//    button_mesh_refine_module_->setToolTip(tr("Refine mesh module"));
//    button_mesh_refine_module_->setStyleSheet("border:2px groove gray;border-radius:10px;border-style:outset");
//    button_mesh_refine_module_->setGeometry(space_size+w_button+space_size, space_size, w_button, h_button);

    // create label and widgets
    // main view widget is initialized for sketching viewer and can switch to mesh viewer(side view at the beginning);
    label_mesh_widget_ = new QLabel(this);
//    label_mesh_widget_->setGeometry(space_size, space_size+h_button+space_size, VIEW_SIZE1, VIEW_SIZE1);
    label_mesh_widget_->setGeometry(space_size, space_size, VIEW_SIZE1, VIEW_SIZE1);
    mesh_widget_ = new MeshWidget(label_mesh_widget_);
    mesh_widget_->setGeometry(0, 0, VIEW_SIZE1, VIEW_SIZE1);

    label_front_sketch_widget_ = new QLabel(QString("Front View"), this);
//    label_front_sketch_widget_->setGeometry(space_size+VIEW_SIZE1+space_size, space_size+h_button+space_size, VIEW_SIZE2, VIEW_SIZE2);
    label_front_sketch_widget_->setGeometry(space_size+VIEW_SIZE1+space_size, space_size, VIEW_SIZE2, VIEW_SIZE2);
    front_sketch_widget_ = new SketchWidget(label_front_sketch_widget_);
    front_sketch_widget_->setGeometry(0, 0, label_front_sketch_widget_->width(), label_front_sketch_widget_->height());
    label_front_view_ = new QLabel(front_sketch_widget_);
    label_front_view_->setFont(QFont("Arial", 15, QFont::Normal));
    label_front_view_->setText("Front");
    label_front_view_->setGeometry(8, 8, 60, 20);

    label_side_sketch_widget_ = new QLabel(QString("Side View"), this);
//    label_side_sketch_widget_->setGeometry(space_size+VIEW_SIZE1+space_size, space_size+h_button+space_size+VIEW_SIZE2+space_size, VIEW_SIZE2, VIEW_SIZE2);
    label_side_sketch_widget_->setGeometry(space_size+VIEW_SIZE1+space_size, space_size+VIEW_SIZE2+space_size, VIEW_SIZE2, VIEW_SIZE2);
    side_sketch_widget_ = new SketchWidget(label_side_sketch_widget_);
    side_sketch_widget_->setGeometry(0, 0, label_side_sketch_widget_->width(), label_side_sketch_widget_->height());
    label_side_view_ = new QLabel(side_sketch_widget_);
    label_side_view_->setFont(QFont("Arial", 15, QFont::Normal));
    label_side_view_->setText("Side");
    label_side_view_->setGeometry(8, 8, 60, 20);

    font.setPointSize(11);
    const int bw = 60;
    const int bh = 30;
    button_undo_ = new QPushButton(tr("Undo"), this);
    button_undo_->setFont(font);
    button_undo_->setToolTip(tr("Undo last operation"));
    button_undo_->setStyleSheet("border:2px groove gray;border-radius:10px;border-style:outset");
//    button_undo_->setGeometry(space_size*2+VIEW_SIZE1, space_size*5+h_button+VIEW_SIZE2*2, 50, 25);
    button_undo_->setGeometry(space_size*2+VIEW_SIZE1, space_size*4+VIEW_SIZE2*2, bw, bh);

    button_clear_ = new QPushButton(tr("Clear"), this);
    button_clear_->setFont(font);
    button_clear_->setToolTip(tr("Clear all widgets"));
    button_clear_->setStyleSheet("border:2px groove gray;border-radius:10px;border-style:outset");
//    button_prerender_->setGeometry(space_size*3+VIEW_SIZE1 + 50, space_size*5+h_button+VIEW_SIZE2*2, 50, 25);
    button_clear_->setGeometry(space_size*3+VIEW_SIZE1 + bw, space_size*4+VIEW_SIZE2*2, bw, bh);

    button_prerender_ = new QPushButton(tr("Render"), this);
    button_prerender_->setFont(font);
    button_prerender_->setToolTip(tr("Pre-render the other view sketch with the current generated mesh"));
    button_prerender_->setStyleSheet("border:2px groove gray;border-radius:10px;border-style:outset");
//    button_prerender_->setGeometry(space_size*3+VIEW_SIZE1 + 50, space_size*5+h_button+VIEW_SIZE2*2, 50, 25);
    button_prerender_->setGeometry(space_size*4+VIEW_SIZE1 + bw*2, space_size*4+VIEW_SIZE2*2, bw, bh);

    button_save_ = new QPushButton(tr("Save"), this);
    button_save_->setFont(font);
    button_save_->setToolTip(tr("Save results(mesh and sketch)"));
    button_save_->setStyleSheet("border:2px groove gray;border-radius:10px;border-style:outset");
//    button_save_->setGeometry(space_size*4+VIEW_SIZE1 + 50*2, space_size*5+h_button+VIEW_SIZE2*2, 50, 25);
    button_save_->setGeometry(space_size*5+VIEW_SIZE1 + bw*3, space_size*4+VIEW_SIZE2*2, bw, bh);


    // create tensorflow model
    sketch2model_ = new TFModel();
}

void MainWindow::initConnection()
{
    connect(mesh_widget_, SIGNAL(activateMeshWidgetStatus()), this, SLOT(activateMeshWidget()));
    connect(front_sketch_widget_, SIGNAL(activateSketchWidgetStatus()), this, SLOT(activateFrontSketchWidget()));
    connect(side_sketch_widget_, SIGNAL(activateSketchWidgetStatus()), this, SLOT(activateSideSketchWidget()));
    connect(front_sketch_widget_, SIGNAL(generateFromSketch()), this, SLOT(generateMesh()));
    connect(side_sketch_widget_, SIGNAL(generateFromSketch()), this, SLOT(generateMesh()));
    connect(button_undo_, SIGNAL(clicked()), this, SLOT(undo()));
    connect(button_clear_, SIGNAL(clicked()), this, SLOT(clear()));
    connect(button_prerender_, SIGNAL(clicked()), this, SLOT(renderSketch()));
    connect(button_save_, SIGNAL(clicked()), this, SLOT(save()));
}

void MainWindow::keyPressEvent(QKeyEvent *e)
{
    if (!e->modifiers() && e->key() == Qt::Key_1)
    {
        b_refine_mode_ = !b_refine_mode_;
        if(b_refine_mode_)
        {
            std::cout<<"change to refined mode..."<<std::endl;
        }
        else
        {
            std::cout<<"change to basic mode..."<<std::endl;
        }
    }
    else if (!e->modifiers() && e->key() == Qt::Key_2)
    {
        mesh_widget_->b_render_color_ = !mesh_widget_->b_render_color_;
        mesh_widget_->updateModelView();
    }
//    else if (!e->modifiers() && e->key() == Qt::Key_3)
//    {
//        mesh_widget_->autoModifyPredefinedContours();
//        mesh_widget_->updateModelView();
//    }
    else if (!e->modifiers() && e->key() == Qt::Key_4)
    {
        mesh_widget_->updateMesh(trimesh::TriMesh::read("predict.obj"));
        sketch2model_->generateVertLabelProb("predict.obj");
//        mesh_widget_->b_render_predefined_contour_ = true;
        mesh_widget_->loadMeshFeatureProbability();
    }
    else if (!e->modifiers() && e->key() == Qt::Key_S)
    {
        mesh_widget_->changeToMode(Sketching);
        mesh_widget_->updateModelView();
    }
    else if (!e->modifiers() && e->key() == Qt::Key_W)
    {
        mesh_widget_->changeToMode(ContourModify);
        mesh_widget_->smoothPredefinedContours(4);
        mesh_widget_->updateModelView();
    }
    else if (!e->modifiers() && e->key() == Qt::Key_E)
    {
        mesh_widget_->changeToMode(ContourDraging);
        mesh_widget_->smoothPredefinedContours(4);
        mesh_widget_->updateModelView();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_E)
    {
        mesh_widget_->changeToMode(ContourEditing);
        mesh_widget_->smoothPredefinedContours(4);
        mesh_widget_->updateModelView();
    }
    else if (!e->modifiers() && e->key() == Qt::Key_R)
    {
        mesh_widget_->changeToMode(CurveDraging);
        mesh_widget_->updateModelView();
    }
    else if (!e->modifiers() && e->key() == Qt::Key_C)
    {
        mesh_widget_->resetInteraction();
        mesh_widget_->updateModelView();
    }
    else if (e->key() == Qt::Key_Plus)
    {
        mesh_widget_->sculpture_scale_ += 0.01;
        std::cout<<mesh_widget_->sculpture_scale_<<std::endl;
    }
    else if (!e->modifiers() && e->key() == Qt::Key_Minus)
    {
        if (mesh_widget_->sculpture_scale_ > 0.01)
        {
            mesh_widget_->sculpture_scale_ -= 0.01;
            std::cout<<mesh_widget_->sculpture_scale_<<std::endl;
        }
        else
        {
            mesh_widget_->sculpture_scale_ *= 0.5;
            std::cout<<mesh_widget_->sculpture_scale_<<std::endl;
        }
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_Z)
    {
        undo();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_A)
    {
        mesh_widget_->sculpture_mode_ = Subdivision;
        mesh_widget_->changeToMode(Sculpting);
        mesh_widget_->updateModelView();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_S)
    {
        mesh_widget_->sculpture_mode_ = Smooth;
        mesh_widget_->changeToMode(Sculpting);
        mesh_widget_->updateModelView();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_D)
    {
        mesh_widget_->sculpture_mode_ = Downward;
        mesh_widget_->changeToMode(Sculpting);
        mesh_widget_->updateModelView();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_F)
    {
        mesh_widget_->sculpture_mode_ = Flatten;
        mesh_widget_->changeToMode(Sculpting);
        mesh_widget_->updateModelView();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_G)
    {
        mesh_widget_->sculpture_mode_ = Grab;
        mesh_widget_->changeToMode(Sculpting);
        mesh_widget_->updateModelView();
    }
    else if (e->modifiers() == Qt::ControlModifier && e->key() == Qt::Key_U)
    {
        mesh_widget_->sculpture_mode_ = Upward;
        mesh_widget_->changeToMode(Sculpting);
        mesh_widget_->updateModelView();
    }

    mesh_widget_->updateGL();
}

void MainWindow::activateMeshWidget()
{
    current_widget_ = MESH;
}

void MainWindow::activateFrontSketchWidget()
{
    current_widget_ = FRONTSKETCH;
}

void MainWindow::activateSideSketchWidget()
{
    current_widget_ = SIDESKETCH;
}

void MainWindow::generateMesh()
{
    front_sketch_widget_->checkActivation();
    side_sketch_widget_->checkActivation();

    if (!front_sketch_widget_->is_activated_ && !side_sketch_widget_->is_activated_)
    {
        mesh_widget_->releaseMesh();
        return;
    }

    mesh_widget_->process_mode_ = Sketching;
    mesh_widget_->b_render_color_ = false;
    mesh_widget_->b_render_predefined_contour_ = false;

//    std::cout<<front_sketch_widget_->is_activated_<<std::endl;
//    std::cout<<side_sketch_widget_->is_activated_<<std::endl;

    if (front_sketch_widget_->is_activated_ && !side_sketch_widget_->is_activated_)
    {
        front_sketch_widget_->saveSketchImage("sketch_front.png");
        sketch2model_->generateMeshSingleView("sketch_front.png");
    }
    else if (!front_sketch_widget_->is_activated_ && side_sketch_widget_->is_activated_)
    {
        side_sketch_widget_->saveSketchImage("sketch_left.png");
        sketch2model_->generateMeshSingleView("sketch_left.png");
//        mesh_widget_->updateMesh(trimesh::TriMesh::read("predict.obj"));
    }
    else if (front_sketch_widget_->is_activated_ && side_sketch_widget_->is_activated_)
    {
        front_sketch_widget_->saveSketchImage("sketch_front.png");
        side_sketch_widget_->saveSketchImage("sketch_left.png");
        if (b_refine_mode_)
        {
            sketch2model_->generateMesh("sketch");
        }
        else
        {
            sketch2model_->generateMeshDualView("sketch");
        }
    }

    mesh_widget_->updateMesh(trimesh::TriMesh::read("predict.obj"));

    sketch2model_->generateVertLabelProb("predict.obj");
    mesh_widget_->loadMeshFeatureProbability();
}

void MainWindow::undo()
{
    if (current_widget_ == FRONTSKETCH)
    {
        front_sketch_widget_->undoSketch();
    }
    else if (current_widget_ == SIDESKETCH)
    {
        side_sketch_widget_->undoSketch();
    }
    else if (current_widget_ == MESH)
    {
        mesh_widget_->undo();
    }

    update();
}

void MainWindow::clear()
{
    front_sketch_widget_->clear();
    side_sketch_widget_->clear();
    mesh_widget_->clear();
}

void MainWindow::renderSketch()
{
    if(front_sketch_widget_->is_activated_ && !side_sketch_widget_->is_activated_)
    {
//        QImage img(VIEW_SIZE2, VIEW_SIZE2, QImage::Format_Grayscale8);
//        sketch_renderer_->renderMeshSuggestiveContour(img, mesh_widget_->getMesh(), 0.5*PI);
        std::vector<std::vector<QPoint>> pts;
        sketch_renderer_->getMeshSuggestiveContour(pts, mesh_widget_->getMesh(), 0.5*PI, VIEW_SIZE2, VIEW_SIZE2, true);
        side_sketch_widget_->setSketchPts(pts);
    }
    if(!front_sketch_widget_->is_activated_ && side_sketch_widget_->is_activated_)
    {
//        QImage img(VIEW_SIZE2, VIEW_SIZE2, QImage::Format_Grayscale8);
//        sketch_renderer_->renderMeshSuggestiveContour(img, mesh_widget_->getMesh(), 0);
        std::vector<std::vector<QPoint>> pts;
        sketch_renderer_->getMeshSuggestiveContour(pts, mesh_widget_->getMesh(), 0, VIEW_SIZE2, VIEW_SIZE2);
        front_sketch_widget_->setSketchPts(pts);
    }
}

void MainWindow::save()
{
    // save sketch image
    QString filename = QFileDialog::getSaveFileName(this,tr("Save Sketch Image"),"../Results/","(*.png)");
    if(filename.isEmpty())
    {
        QMessageBox::information(NULL,tr("Fail"),tr("Create file fail! Please try again!"),QMessageBox::Yes|QMessageBox::No,QMessageBox::Yes);
        return;
    }
    if (front_sketch_widget_->is_activated_)
    {
        front_sketch_widget_->saveSketchImage(filename+QString("_front.png"));
    }
    if (side_sketch_widget_->is_activated_)
    {
        side_sketch_widget_->saveSketchImage(filename+QString("_side.png"));
    }

    // save mesh image
    filename = QFileDialog::getSaveFileName(this,tr("Save Mesh"),"../Results/","(*.ply)");
    if(filename.isEmpty())
    {
        QMessageBox::information(NULL,tr("Fail"),tr("Create file fail! Please try again!"),QMessageBox::Yes|QMessageBox::No,QMessageBox::Yes);
        return;
    }
    mesh_widget_->saveMesh(std::string(filename.toUtf8().constData())+".ply");
}


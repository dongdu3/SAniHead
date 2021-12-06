#ifndef OPENGL_H
#define OPENGL_H

#include <QGLWidget>
#include <QEvent>

enum ProcessMode {Sketching, CurveDraging, ContourDraging, ContourEditing, ContourModify, Sculpting};    // ContourDraging for the pre-defined contours
enum SculptureMode {None, Subdivision, Smooth, Downward, Flatten, Grab, Upward};

class CArcBall;

class COpenGL : public QGLWidget
{
	Q_OBJECT

public:
	COpenGL(QWidget *parent);
	~COpenGL();
	GLfloat *getBallMatrix();

protected:
	 void initializeGL();
	 void resizeGL(int w, int h);
	 void paintGL();
     void drawAxis();
	 void setLight();
	 void mousePressEvent(QMouseEvent *e);
	 void mouseMoveEvent(QMouseEvent *e);
	 void mouseReleaseEvent(QMouseEvent *e);
	 void mouseDoubleClickEvent(QMouseEvent *e);
     void wheelEvent(QWheelEvent *e);
	 virtual void Render();

private:
	// Arc Ball Thing
	CArcBall *myArcBall;
	GLdouble scal;
	GLdouble cx;
	GLdouble cy;
	// Mesh Thing

    // for interface
public:
    QPoint p_anchor_;

    QPoint      mouse_pos_;
    GLfloat     mouse_size_;

    ProcessMode process_mode_;
    SculptureMode sculpture_mode_;
    std::vector<QPoint> sculpture_screen_pos_;
    std::vector<QPoint> tar_curve_pos_;
    bool    b_render_color_;
    bool    b_render_predefined_contour_;
    bool    is_picking_screen_pts_;
    bool    b_curve_selected_;
    bool    b_curve_dragging_;
};

#endif // OPENGL_H

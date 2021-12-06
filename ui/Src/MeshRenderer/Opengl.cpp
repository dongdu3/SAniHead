#include <GL/glut.h>
#include "Opengl.h"
#include "ArcBall.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QColorDialog>
#include <QInputDialog>
#include <iostream>

const double coef = atan(1.0)*4.0*5.0;
const GLfloat pi = 3.1415926536f;

COpenGL::COpenGL(QWidget *parent)
    : QGLWidget(parent), scal(0.0)
{
	cx = 0.0;
	cy = 0.0;
	myArcBall = new CArcBall(width(), height());

    p_anchor_ = QPoint(-100, -100);
    mouse_pos_ = QPoint(-100, -100);
    mouse_size_ = 18;
    process_mode_ = Sketching;
    sculpture_mode_ = None;
    b_render_color_ = false;
    b_render_predefined_contour_ = false;
    is_picking_screen_pts_ = false;
    b_curve_selected_ = false;
    b_curve_dragging_ = false;

    this->setMouseTracking(true);
}

COpenGL::~COpenGL()
{
	delete myArcBall;
}

void COpenGL::initializeGL()
{
	glClearColor(1.0f,1.0f,1.0f,1.0);
	glShadeModel(GL_SMOOTH);
	glClearDepth(1.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
}
void COpenGL::resizeGL(int w, int h)
{
	if ( h ==0 )
		h = 1;

	myArcBall->reSetBound(w, h);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0, (GLdouble)w/(GLdouble)h, 0.001, 1000.0);
	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

void COpenGL::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	setLight();
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glTranslatef(-0.01*cx, -0.01*cy, -3.0+coef*atan(scal));
	glTranslatef(0 + myArcBall->glTransX + myArcBall->glCurTransX,0 + myArcBall->glTransY + myArcBall->glCurTransY,
		0 + myArcBall->glTransZ - myArcBall->glCurTransZ);
	glPushMatrix();
	glMultMatrixf(myArcBall->GetBallMatrix());
	/*glEnable(GL_LIGHTING);*/
	Render();
	//glDisable(GL_LIGHTING);
	glPopMatrix();
}

void COpenGL::drawAxis()
{
    float x_origin = -1.0;
        float y_origin = -1.0;
        float z_origin = 0.5;
        glEnable(GL_BLEND);
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_FASTEST);	// Antialias the lines;
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_LINE_STIPPLE);
        float wid=3.0;
        glLineWidth(wid);
        /*	glLineStipple(1,0x24FF);*/
        glEnable(GL_COLOR_MATERIAL);
        glBegin(GL_LINES);
        glColor3f(1.0,0.0,0.0);
        glVertex3f(x_origin,y_origin,z_origin);
        glVertex3f(x_origin+0.5,y_origin,z_origin);
        glVertex3f(x_origin+0.42,y_origin,z_origin-0.04);
        glVertex3f(x_origin+0.5,y_origin,z_origin);
        glVertex3f(x_origin+0.42,y_origin,z_origin+0.04);
        glVertex3f(x_origin+0.5,y_origin,z_origin);
        glVertex3f(x_origin+0.58,y_origin+0.04,z_origin);
        glVertex3f(x_origin+0.62,y_origin-0.04,z_origin);
        glVertex3f(x_origin+0.58,y_origin-0.04,z_origin);
        glVertex3f(x_origin+0.62,y_origin+0.04,z_origin);

        glColor3f(0,1.0,0);
        glVertex3f(x_origin,y_origin,z_origin);
        glVertex3f(x_origin,y_origin+0.5,z_origin);
        glVertex3f(x_origin-0.04,y_origin+0.42,z_origin);
        glVertex3f(x_origin,y_origin+0.5,z_origin);
        glVertex3f(x_origin+0.04,y_origin+0.42,z_origin);
        glVertex3f(x_origin,y_origin+0.5,z_origin);
        glVertex3f(x_origin-0.03,y_origin+0.64,z_origin);
        glVertex3f(x_origin,y_origin+0.6,z_origin);
        glVertex3f(x_origin+0.03,y_origin+0.64,z_origin);
        glVertex3f(x_origin,y_origin+0.6,z_origin);
        glVertex3f(x_origin,y_origin+0.56,z_origin);
        glVertex3f(x_origin,y_origin+0.6,z_origin);

        glColor3f(0,0,1.0);
        glVertex3f(x_origin,y_origin,z_origin);
        glVertex3f(x_origin,y_origin,z_origin+0.5);
        glVertex3f(x_origin-0.04,y_origin,z_origin+0.42);
        glVertex3f(x_origin,y_origin,z_origin+0.5);
        glVertex3f(x_origin+0.04,y_origin,z_origin+0.42);
        glVertex3f(x_origin,y_origin,z_origin+0.5);

        glVertex3f(x_origin-0.03,y_origin+0.04,z_origin+0.6);
        glVertex3f(x_origin+0.03,y_origin+0.04,z_origin+0.6);
        glVertex3f(x_origin+0.03,y_origin+0.04,z_origin+0.6);
        glVertex3f(x_origin-0.03,y_origin-0.04,z_origin+0.6);
        glVertex3f(x_origin-0.03,y_origin-0.04,z_origin+0.6);
        glVertex3f(x_origin+0.03,y_origin-0.04,z_origin+0.6);

        glEnd();
        glDisable(GL_COLOR_MATERIAL);
        glDisable(GL_LINE_STIPPLE);
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_BLEND);
}

void COpenGL::Render()
{
    if(process_mode_==CurveDraging || process_mode_==ContourEditing)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, this->width(), this->height(),0,-1,1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_BLEND);

        glLineWidth(2);

        // render curve line for curve selection for draging
        if (sculpture_screen_pos_.size() > 1)
        {
            glColor3f(51./255., 167./255., 255.);
            glBegin(GL_LINES);
            for (int i=0; i<sculpture_screen_pos_.size()-1; ++i)
            {
                glVertex2f(sculpture_screen_pos_[i].x(), sculpture_screen_pos_[i].y());
                glVertex2f(sculpture_screen_pos_[i+1].x(), sculpture_screen_pos_[i+1].y());
            }
            glEnd();
        }

        if (tar_curve_pos_.size() > 1)
        {
            glColor3f(1., 51./255., 0.);
            glBegin(GL_LINES);
            for (int i=0; i<tar_curve_pos_.size()-1; ++i)
            {
                glVertex2f(tar_curve_pos_[i].x(), tar_curve_pos_[i].y());
                glVertex2f(tar_curve_pos_[i+1].x(), tar_curve_pos_[i+1].y());
            }
            glEnd();
        }

        glPopAttrib();
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

    // paint mouse position
    if (process_mode_==Sculpting && mouse_pos_.rx()>0 && mouse_pos_.rx()<this->width() && mouse_pos_.ry()>0 && mouse_pos_.ry()<this->height())
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0, this->width(), this->height(), 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glPushAttrib(GL_ENABLE_BIT);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_BLEND);

        glLineWidth(1.f);

        glColor4f(70.f/255, 181.f/255, 209.f/255, 0.75f);
        glBegin(GL_LINE_LOOP);
        float R = mouse_size_/2.f;
        for (int i=0; i<100; i++)
        {
            glVertex2f(R*std::cos(2*pi/100*i)+mouse_pos_.rx(), R*std::sin(2*pi/100*i)+mouse_pos_.ry());
        }
        glEnd();

        glPopAttrib();
        glPopMatrix();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
    }

//    drawAxis();
}

void COpenGL::setLight()
{
	static GLfloat mat_ambient[] = {0.13, 0.13, 0.13, 1};
	static GLfloat mat_specular[] = {1.0, 1.0, 1.0, 1.0};
	static GLfloat white_light[] = {0.8, 0.8, 0.8, 1.0};

	static GLfloat light_position_one[] = {0.0, 0.0, 1.0, 0.0};
	static GLfloat light_position_two[] = {0.0, 0.0, -1.0, 0.0};

	static GLfloat mat_specular_1[] = {0.2, 0.2, 0.2, 1.0};
	static GLfloat mat_shininess[] = {5.0};
	static GLfloat lmodel_ambient[] = {0.6, 0.6, 0.6, 1.0};

	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular_1);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, mat_shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, lmodel_ambient); 


	glLightfv(GL_LIGHT0, GL_AMBIENT, mat_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, white_light);
	glLightfv(GL_LIGHT0, GL_SPECULAR, mat_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position_one);

	glLightfv(GL_LIGHT1, GL_AMBIENT, mat_ambient);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, white_light);
	glLightfv(GL_LIGHT1, GL_SPECULAR, mat_specular);
	glLightfv(GL_LIGHT1, GL_POSITION, light_position_two);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);

}

void COpenGL::mousePressEvent(QMouseEvent *e)
{
	if (e->button() == Qt::LeftButton)
	{
		myArcBall->MouseDown(e->pos());
		setCursor(Qt::OpenHandCursor);
	}
	else if (e->button() == Qt::RightButton)
	{
        if (process_mode_ == CurveDraging)
        {
            if (b_curve_selected_)
            {
                b_curve_dragging_ = true;
                p_anchor_ = e->pos();
                mouse_pos_ = e->pos();
            }
            else
            {
                sculpture_screen_pos_.push_back(e->pos());
                updateGL();
            }
        }
        else if (process_mode_ == ContourDraging || process_mode_ == ContourModify)
        {
            p_anchor_ = e->pos();
        }
        else if (process_mode_ == ContourEditing)
        {
            tar_curve_pos_.push_back(e->pos());
        }
        else if (process_mode_ == Sculpting)
        {
            if (sculpture_mode_ == Grab)
            {
                p_anchor_ = e->pos();
                mouse_pos_ = e->pos();
            }
            else
            {
                is_picking_screen_pts_ = true;
                sculpture_screen_pos_.push_back(e->pos());
            }
        }
        else
        {
            myArcBall->InitBall();
        }
	}
	else if (e->button() == Qt::MiddleButton)
	{
        myArcBall->TransMouseDown(e->pos());
        setCursor(Qt::OpenHandCursor);
	}
}

void COpenGL::mouseMoveEvent(QMouseEvent *e)
{
	if (e->buttons() == Qt::LeftButton)
	{
		myArcBall->MouseMove(e->pos());
        setCursor(Qt::ClosedHandCursor);
    }
    if (e->buttons() == Qt::RightButton)
    {
        if (process_mode_ == CurveDraging)
        {
            if (!b_curve_selected_)
            {
                sculpture_screen_pos_.push_back(e->pos());
            }
        }
        else if (process_mode_ == ContourEditing)
        {
            tar_curve_pos_.push_back(e->pos());
        }
        else if (process_mode_ == Sculpting)
        {
            if (sculpture_mode_ != Grab && is_picking_screen_pts_)
            {
                sculpture_screen_pos_.push_back(e->pos());
            }
        }
    }
	if (e->buttons() == Qt::MiddleButton)
	{
        myArcBall->TransMouseMove(e->pos());
        setCursor(Qt::ClosedHandCursor);
	}

    mouse_pos_ = e->pos();

    updateGL();
}

void COpenGL::mouseReleaseEvent(QMouseEvent *e)
{
	if (e->button() == Qt::LeftButton)
	{
		myArcBall->MouseUp(e->pos());
		setCursor(Qt::ArrowCursor);
		updateGL();
	}
    else if (e->button() == Qt::RightButton)
    {
        if (process_mode_ == CurveDraging)
        {
            sculpture_screen_pos_.clear();
            if (b_curve_dragging_)
            {
                b_curve_dragging_ = false;
                p_anchor_ = QPoint(-100, -100);
            }
            updateGL();
        }
        else if (process_mode_ == ContourDraging || process_mode_ == ContourModify)
        {
            p_anchor_ = QPoint(-100, -100);
        }
        else if (process_mode_ == ContourEditing)
        {
            tar_curve_pos_.clear();
        }
        else if (process_mode_ == Sculpting)
        {
            if (sculpture_mode_ == Grab)
            {
                p_anchor_ = QPoint(-100, -100);
                mouse_pos_ = QPoint(-100, -100);
            }
            else
            {
                is_picking_screen_pts_ = false;
                sculpture_screen_pos_.clear();
            }
        }
    }
    else if (e->button() == Qt::MiddleButton)
	{
        myArcBall->TransMouseUp(e->pos());
        setCursor(Qt::ArrowCursor);
        updateGL();
	}
}

void COpenGL::wheelEvent(QWheelEvent *e)
{
    if (!e->modifiers())
    {
        if (e->delta() > 0)
        {
            scal += 0.01;
        }
        else
        {
            scal -= 0.01;
        }
    }
    else if (e->modifiers() == Qt::ControlModifier)
    {
        if (process_mode_ == Sculpting)
        {
            if (e->delta() > 0)
            {
                mouse_size_ += 2;
            }
            else
            {
                if (mouse_size_ > 2)
                {
                    mouse_size_ -= 2;
                }
            }
        }
    }

	updateGL();
}

void COpenGL::mouseDoubleClickEvent(QMouseEvent *e)
{
/*	if (e->button() == Qt::LeftButton)
	{
		double MV[16], PR[16];
		int    VP[4];
		glGetDoublev(GL_MODELVIEW_MATRIX, MV);
		glGetDoublev(GL_PROJECTION_MATRIX, PR);
		glGetIntegerv(GL_VIEWPORT, VP);
		GLdouble tmp;
		int x = e->pos().x();
		int y = height() - e->pos().y();

		gluUnProject(1.0*x, 1.0*y, 1.0,
					MV, PR, VP,
					&cx, &cy, &tmp);

		updateGL();
	}*/
}

GLfloat *COpenGL::getBallMatrix()
{
	if ( myArcBall )
	{
		return myArcBall->GetBallMatrix();
	}
	else
	{
		return NULL;
	}
}

#include "Widgets/sketchwidget.h"

#include <QFileDialog>
#include <QMessageBox>
#include <assert.h>
#include <cmath>
#include <string>
#include <iostream>

#include "pthread.h"

SketchWidget::SketchWidget(QWidget *parent) : QWidget(parent)
{
    img_w_ = parent->width();
    img_h_ = parent->height();
    img_ = QImage(img_w_, img_h_, QImage::Format_Grayscale8);
    img_.fill(255);

    is_draw_ = false;
    is_erase_ = false;
    is_activated_ = false;

    setMouseTracking(true);			//	enable mouse tracking mode
}

SketchWidget::~SketchWidget()
{

}

void SketchWidget::resizePaintArea(int w, int h)
{
    img_w_ = w;
    img_h_ = h;
    img_ = img_.scaled(QSize(w, h), Qt::KeepAspectRatio);
}

void SketchWidget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);

    QRect back_rect(0, 0, img_w_, img_h_);
    painter.drawRect(back_rect);
    // Draw image
    QRectF rect(0, 0, img_w_, img_h_);
    img_.fill(255);
    painter.drawImage(rect, img_);
//    painter.save();
    // draw sketch/stroke line
    QPen pen_black(Qt::black, 2);
    if (sketch_pts_.size() > 0)
    {
        painter.setPen(pen_black);
        for(size_t i=0; i<sketch_pts_.size(); i++)
        {
            for (size_t j=0; j<sketch_pts_[i].size()-1; j++)
            {
                if(sketch_pts_status_[i][j] && sketch_pts_status_[i][j+1])
                {
                    painter.drawLine(sketch_pts_[i][j], sketch_pts_[i][j+1]);
                }
            }
        }
    }
//    painter.restore();

    if (is_draw_ && current_stroke_pts_.size()>0)
    {
        painter.setPen(pen_black);
        for (size_t i=0; i<current_stroke_pts_.size()-1; ++i)
        {
            painter.drawLine(current_stroke_pts_[i], current_stroke_pts_[i+1]);
        }
    }
}

void SketchWidget::mousePressEvent(QMouseEvent *event)
{
    if (Qt::LeftButton == event->button())
    {
        is_draw_ = true;
        current_stroke_pts_.push_back(event->pos());
    }
    else if (Qt::RightButton == event->button())
    {
        is_erase_ = true;
        setNearPointInvisible(event->pos());
    }
    else if (Qt::MidButton == event->button())
    {

    }

    Q_EMIT activateSketchWidgetStatus();

    update();
}

void SketchWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (is_draw_)
    {
        current_stroke_pts_.push_back(event->pos());
    }
    else if (is_erase_)
    {
        setNearPointInvisible(event->pos());
    }

    update();
}

void SketchWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (is_draw_)
    {
        // smooth the strokes
        if (current_stroke_pts_.size() > 2)
        {
            const int sm_iter = 1;
            for (int k=0; k<sm_iter; ++k)
            {
                for (int i=1; i<current_stroke_pts_.size()-1; ++i)
                {
                    current_stroke_pts_[i] = (current_stroke_pts_[i-1] + current_stroke_pts_[i+1])/2.f;
                }
            }
        }
        sketch_pts_.push_back(current_stroke_pts_);
        sketch_pts_status_.push_back(std::vector<bool>(current_stroke_pts_.size(), true));
        current_stroke_pts_.clear();
        is_draw_ = false;
    }
    else if (is_erase_)
    {
        is_erase_ = false;
    }

    update();

    Q_EMIT  generateFromSketch();
}

void SketchWidget::setNearPointInvisible(const QPoint p)
{
    const float max_d = 7;
    for (size_t i=0; i<sketch_pts_status_.size(); ++i)
    {
        for (size_t j=0; j<sketch_pts_status_[i].size(); ++j)
        {
            if (sketch_pts_status_[i][j])
            {
                QPoint tp = p - sketch_pts_[i][j];
                float d = std::sqrt(tp.rx()*tp.rx() + tp.ry()*tp.ry());
                if (d < max_d)
                {
                    sketch_pts_status_[i][j] = false;
                }
            }
        }
    }
}

void SketchWidget::checkActivation()
{
    is_activated_ = false;
    for (int i=0; i<sketch_pts_status_.size(); ++i)
    {
        for (int j=0; j<sketch_pts_status_[i].size(); ++j)
        {
            if (sketch_pts_status_[i][j])
            {
                is_activated_ = true;
                break;
            }
        }

        if (is_activated_)
        {
            break;
        }
    }

//    std::cout<<is_activated_<<std::endl;
}

void SketchWidget::setSketchPts(std::vector<std::vector<QPoint> > pts)
{
    sketch_pts_.clear();
    sketch_pts_status_.clear();

    sketch_pts_ = pts;
    sketch_pts_status_.resize(sketch_pts_.size());
    for (int i=0; i<sketch_pts_.size(); ++i)
    {
        sketch_pts_status_[i].resize(sketch_pts_[i].size(), true);
    }
    update();
}

void SketchWidget::saveSketchImage(const QString &path)
{
    img_.fill(255);
    QPainter painter(&img_);
    QPen pen_black(Qt::black, 2);
    painter.setPen(pen_black);
    if (sketch_pts_.size() > 0)
    {
        for(size_t i=0; i<sketch_pts_.size(); i++)
        {
            for (size_t j=0; j<sketch_pts_[i].size()-1; j++)
            {
                if(sketch_pts_status_[i][j] && sketch_pts_status_[i][j+1])
                {
                    painter.drawLine(sketch_pts_[i][j], sketch_pts_[i][j+1]);
                }
            }
        }
    }
    painter.end();
    img_.save(path);
}

void SketchWidget::undoSketch()
{
    if(sketch_pts_.size() > 0)
    {
        sketch_pts_.pop_back();
        sketch_pts_status_.pop_back();

        update();
        Q_EMIT generateFromSketch();
    }
}

void SketchWidget::clear()
{
    is_draw_ = false;
    is_erase_ = false;
    is_activated_ = false;

    current_stroke_pts_.clear();
    sketch_pts_.clear();
    sketch_pts_status_.clear();

    update();
}

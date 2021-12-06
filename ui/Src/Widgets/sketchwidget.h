#ifndef SketchWidget_H
#define SketchWidget_H
#include <QWidget>
#include <QPoint>
#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <vector>

enum ViewType {Front, Side};

class SketchWidget : public QWidget
{
    Q_OBJECT
public:
    explicit SketchWidget(QWidget *parent = nullptr);
    ~SketchWidget();

public:
  void resizePaintArea(int w, int h);
  void paintEvent(QPaintEvent *);
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);

protected:
  void setNearPointInvisible(const QPoint p);

public:
  void checkActivation();
  void setSketchPts(std::vector<std::vector<QPoint>> pts);
  void saveSketchImage(const QString &path);
  void undoSketch();
  void clear();

private:
    int img_h_;
    int img_w_;
    QImage img_;

private:
    bool is_draw_;
    bool is_erase_;

    std::vector<QPoint> current_stroke_pts_;
    std::vector<std::vector<QPoint>> sketch_pts_;
    std::vector<std::vector<bool>> sketch_pts_status_;      // the size is as the same as the sketch_pts, and true for visible, false for invisible

public:
    bool is_activated_;

Q_SIGNALS:
    void activateSketchWidgetStatus();
    void generateFromSketch();
};

#endif // SketchWidget_H

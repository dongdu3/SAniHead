#ifndef SUGGESTIVECONTOUR_H
#define SUGGESTIVECONTOUR_H

#include <TriMesh.h>
#include <QImage>

#define PI 3.141592653

using trimesh::point;
using std::vector;

struct Ray
{
    point origin, direction;
};

struct Camera
{
    point pos, dir;
    point u, v;
};

class SuggestiveContour
{
public:
    void renderMeshSuggestiveContour(QImage &img, trimesh::TriMesh *mesh, const float &ang);
    void getMeshSuggestiveContour(std::vector<std::vector<QPoint>> &pts, trimesh::TriMesh *mesh, const float &ang, const int img_w=400, const int img_h=400, bool b_render_half=false);

protected:
    void initRenderSuggesContour();
    void calcSuggestiveContours(trimesh::vec viewpos, vector<float> &ndotv, vector<float> &kr, vector<float> &sctest_num,
                                vector<float> &sctest_den, bool b_render_half=false);
    void drawIsolines(trimesh::vec viewpos, vector<float> &val, vector<float> &test_num, vector<float> &test_den,
                      vector<float> &ndotv, bool do_bfcull, bool do_hermite, bool do_test, bool b_render_half=false);

private:
    trimesh::vec calcGradKr(int i, trimesh::vec viewpos);
    void computePerview(trimesh::vec viewpos, vector<float> &ndotv, vector<float> &kr, vector<float> &sctest_num,
                        vector<float> &sctest_den);
    void drawMeshIsoline(trimesh::vec viewpos, int v0, int v1, int v2, vector<float> &val, vector<float> &test_num,
                         vector<float> &test_den, vector<float> &ndotv, bool do_bfcull, bool do_hermite, bool do_test);
    void drawMeshIsoline2(trimesh::vec viewpos, int v0, int v1, int v2, vector<float> &val, vector<float> &test_num,
                          vector<float> &test_den, bool do_hermite, bool do_test);
    float findZeroLinear(float val0, float val1);
    float findZeroHermite(int v0, int v1, float val0, float val1, trimesh::vec &grad0, trimesh::vec &grad1);

private:
    Camera calcCamPose(float alpha, float fi, float d);
    void rotateAroundVec(point p, point n, point& q, float theta);
    void rotateMesh(float ang);

public:
    SuggestiveContour();
    ~SuggestiveContour();

//private:
//    trimesh::TriMesh mesh_;
//    float sug_thresh_;
//    std::vector<trimesh::point> sketch_pts_;
};

#endif // SUGGESTIVECONTOUR_H

#ifndef MeshWidget_H
#define MeshWidget_H

#include "MeshRenderer/ViewObj.h"
#include "MeshRenderer/Opengl.h"
#include "TriMesh.h"

#include <QWidget>

class LaplacianDeformation;
class BaseProcessor;
class CDijkstraShortestPath;

class MeshWidget : public COpenGL
{
    Q_OBJECT
public:
    MeshWidget(QWidget *parent = 0);
    ~MeshWidget();

public:
    void    setMesh(trimesh::TriMesh *mesh);
    void    updateMesh(trimesh::TriMesh *mesh);
    void    updateVerts(std::vector<trimesh::point> pts);
    trimesh::TriMesh* getMesh();
    void    saveMesh(const std::string &path);
    void    releaseMesh();

    void    changeToMode(ProcessMode mode);
    void    loadFeaturedMesh(const std::string mesh_name);
    void    loadMeshFeatureProbability(const std::string label_name="predict_label.txt");
    void    undo();
    void    clear();
    void    resetInteraction();

public:
    void    updateModelView();      // update mesh information: vertex, normal, color, ...

protected:
    void    Render();
    void    unifyMesh();
    void    drawTstrips();

    void	mousePressEvent(QMouseEvent *e);
    void	mouseMoveEvent(QMouseEvent *e);
    void	mouseReleaseEvent(QMouseEvent *e);

protected:
    trimesh::Color jetColor(float t);

protected:
//    void	getFaceCenter(trimesh::point &fcenter, const trimesh::TriMesh::Face &face);
//    int     pickOnePointOnMesh(float &err, const QPoint &p);

    void    calcIntersectPtsWithLineDrawn();
//    void    calcFeatureAreaNearLineDrawn(bool with_feature_check=true, float feat_val_thres=0.45);

    void    initLaplacianDeformer(const double lambda=1.);
    void    doSurfaceCurveDragging();
    void    doFeatureCurveDragging();
    void    doContourCurveDragging();
    void    doContourCurveEditing();
//    void    doFeatureCurveEditing();
    void    doSculpture();

private:
    // for grabing
    void    pickAreaFacesNearScreenPoint(QPoint &p, std::vector<trimesh::point> &proj_vts, std::vector<int> &fidx, const float &d_thres);
    void    pickAreaVertexNearScreenPoint(QPoint p, const float &d_thres, float fd_thres_time=2);
    void    pickPredefinedContourVertexNearScreenPoint(QPoint p, const float &d_thres);

    // for curve deformation with line drawn on the surface
    void    pickSurfaceCurveWithLineDrawn(std::vector<QPoint> screen_line_pts, const float d_thres);
    int     pickSurfaceVertexWithScreenPoint(const std::vector<trimesh::point> &proj_vts, const std::vector<int> &vidx_sel, const QPoint &p);

    // for feature curve deformation with predicted feature probablity
    void    pickAreaFacesWithScreenPoints(std::vector<QPoint> &screen_pts, std::vector<trimesh::vec2> &proj_vts, std::vector<int> &fidx, const float &d_thres);
    void    pickAreaVertexWithLineDrawn(std::vector<QPoint> &screen_pts, std::vector<trimesh::vec2> &proj_vts, std::vector<int> &vidx, const float &d_thres, float fd_thres_time=3.5);
    void    calcFeatureCurveNearLineDrawn(std::vector<QPoint> screen_pts, const float feat_val_thres=0.45);
    std::vector<QPoint>    calcScreenCurvePtsForEditing(std::vector<QPoint> screen_pts, bool b_trans=true);

    bool    pickTwoTerminalFeaturePtsOfLineDrawn(int &p_id_start, int &p_id_end, const QPoint &start_point, const QPoint &end_point, const std::vector<trimesh::vec2> &proj_vts, const std::vector<int> &vidx_sel, const float feat_val_thres=0.45, const int n_iter_neighbor_search=2);

    // for sculpture
    void    pickVisibleFacesWithScreenPoints(std::vector<QPoint> &screen_pts, std::vector<trimesh::point> &proj_vts, std::vector<int> &fidx, const float &d_thres);
    void    pickVisibleVertexWithScreenPoints(std::vector<QPoint> &screen_pts, const float &d_thres, float fd_thres_time=2);

    // others
    void    interpolateScreenPts(std::vector<QPoint> &screen_pts);
    void    pickMeshFacesWithLineDrawn(std::vector<QPoint> &screen_pts, std::vector<trimesh::point> &proj_vts, std::vector<int> &fidx);
    void    subdivideMeshPart(std::vector<QPoint> &screen_pts, const float &d_thres);
    void    loadPredefinedContourVertexIndex();
    std::vector<int>    findCommonNeighbor(const int &id1, const int &id2);

    int     pickPointOnMesh(QPoint p, const float &d_thres);
    int     pickPointOnContour(QPoint p, const float &d_thres);
    void    modifyContour(const int influence_step=5);
    void    smoothModifiedContour(const int sm_iter=3);
    void    updatePredefinedContour(const int sm_iter=3);   // if sm_iter > 0, then do smoothing
    void    updatePredefinedContour(const std::vector<int> contour_idx, const int sm_iter);     // if sm_iter > 0, then do smoothing

    int     searchBestPointWithFeatureProbability(const int id);    // in 2-ring neighborhoods

public:
    void    autoModifyPredefinedContours(const int sm_iter=3);     // only for the contours of ears; if sm_iter > 0, then do smoothing

public:
    void    smoothPredefinedContours(int sm_iter=3);

Q_SIGNALS:
    void    activateMeshWidgetStatus();

public Q_SLOTS:


private:
    ViewObj model_view_;
    trimesh::TriMesh *mesh_;
    BaseProcessor *base_processor_;
    LaplacianDeformation *lap_deformer_;

    std::vector<std::vector<trimesh::point>> intersect_line_pts_;

public:
    std::vector<trimesh::TriMesh> meshes_bk_;      // for backup in the period of the refinement/sculpture module
    CDijkstraShortestPath *dijk_short_path_;

    // for interaction
public:
    std::vector<int> v_pick_idx_;
    std::vector<bool> b_v_fixed_;
    std::vector<bool> b_face_sub_;

    float sculpture_scale_;
    std::vector<float> w_pick_sculpture_;    // for the weighted scale when doing sculpture

    //  pre-difined contours
    int n_predefined_contour_;
//    bool b_render_predefined_contour_;
    std::vector<std::vector<int>> predefined_contour_idx_;
    std::vector<std::vector<std::vector<int>>> predefined_contour_idx_bk_;
    std::vector<int> id_contour_;
    std::vector<bool> b_contour_;
    bool b_contour_changed_;
    LaplacianDeformation *contour_deformer_;

    // tmp projected vertices and normals
    int contour_pick_id_;
    int src_pick_id_;
    int tar_pick_id_;
    std::vector<trimesh::vec2> proj_vec2_;
    std::vector<trimesh::vec3> proj_norms_;
};

#endif // MeshWidget_H

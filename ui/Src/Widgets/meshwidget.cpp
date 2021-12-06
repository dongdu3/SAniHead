#include "Widgets/meshwidget.h"
#include "MeshProcessor/BaseProcessor.h"
#include "MeshProcessor/LineIntersection/raytriangleintersection.h"
#include "MeshProcessor/LineIntersection/LineSegmentIntersection.h"
#include "MeshProcessor/DijkstraShortestPath.h"
#include "MeshProcessor/Deformation/LaplacianDeformation.h"

#include <assert.h>
#include <Eigen/Dense>
#include <QPainter>

using namespace Eigen;

MeshWidget::MeshWidget(QWidget *parent) : COpenGL(parent)
{
    mesh_ = nullptr;
    base_processor_ = new BaseProcessor();
    dijk_short_path_ = new CDijkstraShortestPath();
    lap_deformer_ = nullptr;
    contour_deformer_ = nullptr;
    b_contour_changed_ = false;
    sculpture_scale_ = 0.08;
    n_predefined_contour_ = 4;
    src_pick_id_ = -1;
    tar_pick_id_ = -1;
    contour_pick_id_ = -1;
    loadPredefinedContourVertexIndex();

    this->setMouseTracking(true);
}

MeshWidget::~MeshWidget()
{
    model_view_.releaseVBO();

    this->clear();
    if (base_processor_)
    {
        delete base_processor_;
        base_processor_ = nullptr;
    }
    if (dijk_short_path_)
    {
        delete dijk_short_path_;
        dijk_short_path_ = nullptr;
    }
}

bool inline igtPointInTriangle2D(trimesh::point p, trimesh::point pt[3])
{	// can have a reference in BuildParent program
    trimesh::point v1 = pt[0] - p;
    trimesh::point v2 = pt[1] - p;
    trimesh::point v3 = pt[2] - p;

    //	Clear the value on z
    v1[2]=0;

    v2[2]=0;
    v3[2]=0;

    //the direction of the triangle is anti-clockwise
    float fS1 = v1.cross(v2)[2];
    float fS2 = v2.cross(v3)[2];
    float fS3 = v3.cross(v1)[2];

    if (( fS1*fS2 > 0 ) && ( fS3*fS2> 0 ))
        return true;
    else
        return false;
}

void MeshWidget::Render()
{
    if(mesh_ && !mesh_->vertices.empty())
    {
        model_view_.renderVBO();
    }

    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);

    COpenGL::Render();

//    // draw intersection lines
//    if(intersect_line_pts_.size() > 0)
//    {
//        glLineWidth(3);
//        glColor3f(1., 51./255., 0);
//        glBegin(GL_LINES);
//        for (int i=0; i<intersect_line_pts_.size(); ++i)
//        {
//            for (int j=0; j<intersect_line_pts_[i].size()-1; ++j )
//            {
//                glVertex3fv(intersect_line_pts_[i][j]);
//                glVertex3fv(intersect_line_pts_[i][j+1]);
//            }
//        }
//        glEnd();
//    }

    // draw selected lines
    if(process_mode_==CurveDraging && v_pick_idx_.size()>0)
    {
        const int nv_pick = v_pick_idx_.size();
        std::vector<trimesh::point> vts_render(nv_pick, trimesh::point(0, 0, 0));
        for (int i=0; i<nv_pick; ++i)
        {
            vts_render[i] = mesh_->vertices[v_pick_idx_[i]];
        }

        // smooth the pts_pick
        const int sm_iter = 3;
        for (int k=0; k<sm_iter; ++k)
        {
            for (int i=1; i<nv_pick-1; ++i)
            {
                vts_render[i] = (vts_render[i-1] + vts_render[i+1])/2.f;
            }
        }

        glLineWidth(5);
        glColor3f(1., 51./255., 0);
        glBegin(GL_LINES);
        for (int i=0; i<v_pick_idx_.size()-1; ++i)
        {
            glVertex3fv(vts_render[i]);
            glVertex3fv(vts_render[i+1]);
        }
        glEnd();
    }

    // draw pre-defined contours
    if (mesh_ && b_render_predefined_contour_)
    {
        glLineWidth(4);
        glColor3f(1., 51./255., 0);
//        glColor3f(51./255., 167./255., 255.);
        glBegin(GL_LINES);
        for (int i=0; i<n_predefined_contour_; ++i)
        {
            const int n = predefined_contour_idx_[i].size();
            for (int j=0; j<n-1; ++j)
            {
                glVertex3fv(mesh_->vertices[predefined_contour_idx_[i][j]]);
                glVertex3fv(mesh_->vertices[predefined_contour_idx_[i][j+1]]);
            }

            if (i < 2)
            {
                glVertex3fv(mesh_->vertices[predefined_contour_idx_[i][n-1]]);
                glVertex3fv(mesh_->vertices[predefined_contour_idx_[i][0]]);
            }
        }
        glEnd();
    }

    if (tar_pick_id_ >= 0)
    {
        glPointSize(9);
        glColor3f(1., 51./255., 0);
        glBegin(GL_POINTS);
        glVertex3fv(mesh_->vertices[tar_pick_id_]);
        glEnd();
    }
    else if (src_pick_id_ >= 0)
    {
        glPointSize(9);
        glColor3f(1., 51./255., 0);
        glBegin(GL_POINTS);
        glVertex3fv(mesh_->vertices[src_pick_id_]);
        glEnd();
    }

//    // draw testing
//    if(mesh_ && v_pick_idx_.size() > 0)
//    {
//        glPointSize(5);
//        glColor3f(1., 0, 0);
//        glBegin(GL_POINTS);
//        for (int i=0; i<v_pick_idx_.size(); ++i)
//        {
//            glVertex3fv(mesh_->vertices[v_pick_idx_[i]]);
//        }
//        glEnd();
//    }

    // drawing mesh faces when do subdivision
    if (mesh_ && sculpture_mode_==Subdivision)
    {
        glLineWidth(1);
        glPolygonMode(GL_FRONT, GL_LINE);
        glColor3f(0.5, 1.0, 1.0);
        glBegin(GL_LINES);
        for (int i=0; i<mesh_->faces.size(); ++i)
        {
            const trimesh::TriMesh::Face &f = mesh_->faces[i];
            for (int j=0; j<3; ++j)
            {
                glVertex3fv(mesh_->vertices[f[j]]);
                glVertex3fv(mesh_->vertices[f[(j+1)%3]]);
            }
        }
        glEnd();
//        drawTstrips();
        glPolygonMode(GL_FRONT, GL_FILL);
    }

    glDisable(GL_POINT_SMOOTH);
    glDisable(GL_LINE_SMOOTH);
}

void MeshWidget::updateModelView()
{
    if(mesh_ && !mesh_->vertices.empty())
    {
        mesh_->clear_normals();
        mesh_->need_normals();
        if(mesh_->faces.empty())
        {
            model_view_.updateVBO(GL_STATIC_DRAW,GL_POINTS, mesh_->vertices.size(), (float *)mesh_->vertices.data(), (float *)mesh_->normals.data());
        }
        else
        {
            if (b_render_color_ && mesh_->colors.size()>0)
            {
                int face_size = mesh_->faces.size();
                std::vector <trimesh::point> vertices(face_size*3), normals(face_size*3), colors(face_size*3);
                for(int i=0; i<face_size; ++i)
                {
                    for(int j=0; j<3; ++j)
                    {
                        vertices[i*3+j] = mesh_->vertices[mesh_->faces[i][j]];
                        normals[i*3+j] = mesh_->normals[mesh_->faces[i][j]];
                        colors[i*3+j] = mesh_->colors[mesh_->faces[i][j]];
                    }
                }
                model_view_.updateVBO(GL_STATIC_DRAW,GL_TRIANGLES,3 * mesh_->faces.size(), (float *)vertices.data(), (float *)normals.data(), (float *)colors.data());
            }
            else
            {
                int face_size = mesh_->faces.size();
                std::vector <trimesh::point> vertices(face_size*3),normals(face_size*3);
                for(int i=0; i<face_size; ++i)
                {
                    for(int j=0; j<3; ++j)
                    {
                        vertices[i*3+j] = mesh_->vertices[mesh_->faces[i][j]];
                        normals[i*3+j] = mesh_->normals[mesh_->faces[i][j]];
                    }
                }
                model_view_.updateVBO(GL_STATIC_DRAW,GL_TRIANGLES,3 * mesh_->faces.size(), (float *)vertices.data(), (float *)normals.data());
            }
        }
    }

    updateGL();
}

void MeshWidget::mousePressEvent(QMouseEvent *e)
{
    COpenGL::mousePressEvent(e);

    if (sculpture_mode_==Grab && p_anchor_.rx()>=0 && mouse_pos_.rx()>=0)
    {
        pickAreaVertexNearScreenPoint(p_anchor_, mouse_size_, 2.);
        initLaplacianDeformer();
        meshes_bk_.push_back(*mesh_);
    }
    else if (b_curve_dragging_ && p_anchor_.rx()>=0)
    {
        meshes_bk_.push_back(*mesh_);
    }
    else if (process_mode_==ContourDraging && b_render_predefined_contour_ && p_anchor_.rx()>=0)
    {
        pickPredefinedContourVertexNearScreenPoint(p_anchor_, 30);
        if (v_pick_idx_.size() > 0)
        {
            meshes_bk_.push_back(*mesh_);
        }
    }
    else if (process_mode_==ContourModify && p_anchor_.rx()>=0)
    {
        src_pick_id_ = pickPointOnContour(p_anchor_, 10);
        updateGL();
    }

    Q_EMIT activateMeshWidgetStatus();
}

void MeshWidget::mouseMoveEvent(QMouseEvent *e)
{
    COpenGL::mouseMoveEvent(e);

    if (sculpture_mode_==Grab && p_anchor_.rx()>=0 && mouse_pos_.rx()>=0)
    {
        doSculpture();
    }
    else if (b_curve_dragging_ && p_anchor_.rx()>=0 && mouse_pos_.rx()>=0)
    {
        doFeatureCurveDragging();
    }
    else if (process_mode_==ContourDraging && v_pick_idx_.size()>0 && p_anchor_.rx()>=0 && mouse_pos_.rx()>=0)
    {
        doContourCurveDragging();
    }
    else if (process_mode_==ContourModify && src_pick_id_>=0)
    {
        tar_pick_id_ = pickPointOnMesh(mouse_pos_, 10);
        modifyContour();
    }
}

void MeshWidget::mouseReleaseEvent(QMouseEvent *e)
{
    if (process_mode_ == CurveDraging)
    {
        if (!b_curve_selected_ && sculpture_screen_pos_.size() > 0)
        {
//            interpolateScreenPts(sculpture_screen_pos_);
//            calcFeatureCurveNearLineDrawn(sculpture_screen_pos_);
            pickSurfaceCurveWithLineDrawn(sculpture_screen_pos_, mouse_size_);
            b_curve_selected_ = true;
        }
    }
    else if (process_mode_==ContourEditing)
    {
        doContourCurveEditing();
    }
    else if (process_mode_==ContourModify && src_pick_id_>=0)
    {
        updatePredefinedContour(3);
        src_pick_id_ = -1;
        tar_pick_id_ = -1;
        contour_pick_id_ = -1;
    }
    else if (process_mode_==Sculpting)
    {
        if (sculpture_mode_ == Grab)
        {
            if (p_anchor_.rx() > 0)
            {
                v_pick_idx_.clear();
                b_v_fixed_.clear();
            }
        }
        else
        {
            if (sculpture_screen_pos_.size() > 0)
            {
                interpolateScreenPts(sculpture_screen_pos_);

                if (sculpture_mode_ == Subdivision)
                {
                    subdivideMeshPart(sculpture_screen_pos_, mouse_size_/2.f);
                }
                else
                {
                    pickVisibleVertexWithScreenPoints(sculpture_screen_pos_, mouse_size_/2.f, 1.5);
                    initLaplacianDeformer();
                    doSculpture();
                }
            }
        }
    }

    COpenGL::mouseReleaseEvent(e);

//    calcIntersectPtsFromLineDrawn();
}

trimesh::Color MeshWidget::jetColor(float t)
{
    assert(t>=0);

    if(t>=1)
    {
        return trimesh::Color(1.f, 0.f, 0.f);
    }

    float t0, t1; float l = 2.0+sqrtf(2.0);
    t0 = 1.0f/l;
    t1 = (1.0f+sqrtf(2.0))/l;

    // actually not jet, but from (0, 0, 1)-->(0, 1, 1)-->(1, 1, 0)-->(1, 0, 0)
    // blue-->cyan-->yello-->red

    trimesh::Color b; b[0] = 0.f; b[1] = 0.f;   b[2] = 1.f;       // blue
    trimesh::Color cy; cy[0] = 0.f; cy[1] = 1.f; cy[2] = 1.f;     // cyan
    trimesh::Color y; y[0] = 1.f;   y[1] = 1.f; y[2] = 0.f;       // yellow
    trimesh::Color r; r[0] = 1.f;   r[1] = 0.f;   r[2] = 0.f;     // red

    trimesh::Color rt;
    if(t<=t0)
    {
        float s = 1-t/t0;
        rt = s*b + (1-s)*cy;
    }
    else if(t<=t1)
    {
        float s = 1-(t-t0)/(t1-t0);
        rt = s*cy + (1-s)*y;
    }
    else
    {
        float s = 1-(t-t1)/(1.0f-t1);
        rt = s*y + (1-s)*r;
    }

    return trimesh::Color(rt[0], rt[1], rt[2]);
}

void MeshWidget::unifyMesh()
{
    if(mesh_ && !mesh_->vertices.empty())
    {
        mesh_->need_bsphere();
        for(int i=0; i<mesh_->vertices.size(); ++i)
        {
            mesh_->vertices[i] = (mesh_->vertices[i]-mesh_->bsphere.center)/mesh_->bsphere.r;
        }
    }
}

void MeshWidget::drawTstrips()
{
    mesh_->clear_tstrips();
    mesh_->need_tstrips();

    const int *t = &mesh_->tstrips[0];
    const int *end = t + mesh_->tstrips.size();
    while (likely(t < end))
    {
        int striplen = *t++;
        glDrawElements(GL_TRIANGLE_STRIP, striplen, GL_UNSIGNED_INT, t);
        t += striplen;
    }
}

void MeshWidget::setMesh(trimesh::TriMesh *mesh)
{
    if(mesh_)
    {
        delete mesh_;
        mesh_ = nullptr;
    }

    mesh_ = mesh;
    b_face_sub_.resize(mesh_->faces.size(), false);
    unifyMesh();
    updateModelView();
}

void MeshWidget::updateMesh(trimesh::TriMesh *mesh)
{
    resetInteraction();

    if (mesh_)
    {
        delete mesh_;
        mesh_ = nullptr;
    }

    if (contour_deformer_)
    {
        delete contour_deformer_;
        contour_deformer_ = nullptr;
    }

    mesh_ = mesh;
    b_face_sub_.resize(mesh_->faces.size(), false);
    if (b_contour_changed_)
    {
        loadPredefinedContourVertexIndex();
    }

    updateModelView();
}

void MeshWidget::updateVerts(std::vector<trimesh::point> pts)
{
    assert(mesh_);
    mesh_->vertices = pts;
    b_face_sub_.resize(mesh_->faces.size(), false);
    updateModelView();
}

trimesh::TriMesh* MeshWidget::getMesh()
{
    return mesh_;
}

void MeshWidget::saveMesh(const std::string &path)
{
    if(mesh_)
    {
        mesh_->write(path);
    }
}

void MeshWidget::releaseMesh()
{
    if(mesh_)
    {
        delete mesh_;
        mesh_ = nullptr;

        b_face_sub_.clear();
    }

    updateModelView();
}

void MeshWidget::changeToMode(ProcessMode mode)
{
    if (process_mode_ != mode)
    {
        resetInteraction();
    }

    process_mode_ = mode;
    if (process_mode_ != Sculpting)
    {
        sculpture_mode_ = None;
    }

//    if (mode==CurveEditing || mode==CurveDraging)
//    {
//        b_render_color_ = true;
//    }
//    else
//    {
//        b_render_color_ = false;
//    }

    if (mode == ContourModify || mode == ContourDraging || mode == ContourEditing)
    {
        b_render_predefined_contour_ = true;
    }

    if (sculpture_mode_ == Subdivision)
    {
        mouse_size_ = 40;
    }
    else
    {
        mouse_size_ = 18;
    }

    updateGL();
}

void MeshWidget::undo()
{
    if (meshes_bk_.size() > 0)
    {
        if (process_mode_==ContourDraging || process_mode_ == ContourEditing || process_mode_==CurveDraging || sculpture_mode_!=None)
        {
            *mesh_ = meshes_bk_[meshes_bk_.size()-1];
            meshes_bk_.pop_back();

            updateModelView();
        }
    }

    int n = predefined_contour_idx_bk_.size();
    if (n > 1)
    {
        predefined_contour_idx_ = predefined_contour_idx_bk_[n-2];
        predefined_contour_idx_bk_.pop_back();
    }
}

void MeshWidget::clear()
{
    if (mesh_)
    {
        if (lap_deformer_)
        {
            delete lap_deformer_;
            lap_deformer_ = nullptr;
        }

        if (contour_deformer_)
        {
            delete contour_deformer_;
            contour_deformer_ = nullptr;
        }

        delete mesh_;
        mesh_ = nullptr;

        sculpture_scale_ = 0.08;
        w_pick_sculpture_.clear();

        // for OpenGL base class
        mouse_size_ = 20;
        p_anchor_ = QPoint(-100, -100);
        mouse_pos_ = QPoint(-100, -100);
        b_render_color_ = false;
        b_curve_selected_ = false;
        b_curve_dragging_ = false;
        b_render_predefined_contour_ = false;
        is_picking_screen_pts_ = false;
        process_mode_ = Sketching;
        sculpture_mode_ = None;

        intersect_line_pts_.clear();
        sculpture_screen_pos_.clear();
        meshes_bk_.clear();
        v_pick_idx_.clear();
        b_v_fixed_.clear();
        b_face_sub_.clear();
        tar_curve_pos_.clear();

        src_pick_id_ = -1;
        tar_pick_id_ = -1;
        contour_pick_id_ = -1;

        predefined_contour_idx_ = predefined_contour_idx_bk_[0];
        predefined_contour_idx_bk_.clear();

        loadPredefinedContourVertexIndex();

//        update();
        updateModelView();
    }
}

void MeshWidget::resetInteraction()
{
    if (lap_deformer_)
    {
        delete lap_deformer_;
        lap_deformer_ = nullptr;
    }

    sculpture_scale_ = 0.08;
    w_pick_sculpture_.clear();

    // for OpenGL base class
    mouse_size_ = 18;
//    p_anchor_ = QPoint(-100, -100);
//    mouse_pos_ = QPoint(-100, -100);
    b_curve_selected_ = false;
    b_curve_dragging_ = false;
    is_picking_screen_pts_ = false;
    b_render_predefined_contour_ = false;
    intersect_line_pts_.clear();
    sculpture_screen_pos_.clear();
    v_pick_idx_.clear();
    b_v_fixed_.clear();
    tar_curve_pos_.clear();

    src_pick_id_ = -1;
    tar_pick_id_ = -1;
    contour_pick_id_ = -1;
}

void MeshWidget::loadFeaturedMesh(const std::string mesh_name)
{
    const int nv = 4962;
    const int nf = 9920;

    if (mesh_)
    {
        mesh_->colors.resize(nv);
    }
    else
    {
        mesh_ = new trimesh::TriMesh;
        mesh_->vertices.resize(nv);
        mesh_->colors.resize(nv);
    }

    std::ifstream mesh_reader(mesh_name);

    // skip first 12 lines
    std::string s;
    for (int i=0; i<12; ++i)
    {
        std::getline(mesh_reader, s);
    }

    // read mesh vertices and colors
    float x, y, z;
    int r, g, b;
    for (int i=0; i<2466; ++i)
    {
        mesh_reader>>x>>y>>z>>r>>g>>b;
        mesh_->vertices[i] = trimesh::point(x, y, z);
        mesh_->colors[i] = trimesh::vec3(r/255.f, g/255.f, b/255.f);
    }

    if (mesh_->faces.size()==0)
    {
        mesh_->faces.resize(nf);
        // read mesh faces
        int n, id1, id2, id3;
        for (int i=0; i<4928; ++i)
        {
            mesh_reader>>n>>id1>>id2>>id3;

        }
    }

    updateModelView();
}

void MeshWidget::loadMeshFeatureProbability(const std::string label_name)
{
    if (mesh_)
    {
        const int nv = mesh_->vertices.size();
        mesh_->colors.resize(nv);

        std::string lab_val;
        std::ifstream label_pre(label_name);
        for (int i=0; i<nv; ++i)
        {
            assert(std::getline(label_pre, lab_val));
            mesh_->colors[i] = jetColor(std::stof(lab_val));
        }
        label_pre.close();
    }
}

//void MeshWidget::getFaceCenter(trimesh::point &fcenter, const trimesh::TriMesh::Face &face)
//{
//    if (mesh_)
//    {
//        fcenter = mesh_->vertices[face[0]] + mesh_->vertices[face[1]] + mesh_->vertices[face[2]];
//        fcenter /= 3.0;
//    }
//}

//int MeshWidget::pickOnePointOnMesh(float &err, const QPoint &p)
//{
//    if (mesh_)
//    {
//        trimesh::point vDstPos(p.x(), p.y(), 0);

//        ////////////////////////////////////////////////////////////////////
//        //	Step.1	Get Modelview/Projection Matrices
//        double	matMV[16] , matProj[16];
//        int		ViewPort[4];

//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glMultMatrixf(getBallMatrix ());
//        glGetDoublev  (GL_MODELVIEW_MATRIX, matMV);
//        glGetDoublev  (GL_PROJECTION_MATRIX, matProj);
//        glGetIntegerv (GL_VIEWPORT, ViewPort);
//        glPopMatrix();

//        ////////////////////////////////////////////////////////////////////
//        //	Step.2	Searching the nearest vertex

//        //	2.1	Store the value
//        int		nNearestVrtIdx = -1;
//        int		nNearestTriIdx	= -1;
//        double	dNearestTriDist	= 10000;

//        //	2.2	Search along the triangle
//        int m_nTriangles = mesh_->faces.size();
//        for( int iTri=0; iTri<m_nTriangles; iTri++ )
//        {
//            // get the triangle
//            trimesh::TriMesh::Face pTri = mesh_->faces[iTri];

//            // Get triangel center distance;
//            double dTriDist = 0;
//            {
//                trimesh::point vTriCent;
//                getFaceCenter(vTriCent,pTri);

//                double dScrX,dScrY,dScrZ;
//                gluProject(vTriCent[0], vTriCent[1], vTriCent[2],
//                    matMV, matProj, ViewPort,
//                    &dScrX,&dScrY,&dScrZ);

//                dTriDist = dScrZ;
//            }

//            //	Check whether the nearest
//            if (dTriDist < dNearestTriDist)
//            {
//                // Get Vertex Screen Positions
//                trimesh::point pvSrcPos[3];
//                {	// Project them
//                    for( int nVrt=0; nVrt<3; nVrt++)
//                    {
//                        trimesh::point vPos = mesh_->vertices[pTri[nVrt]];

//                        double dScrX,dScrY,dScrZ;
//                        gluProject (vPos[0], vPos[1], vPos[2],
//                            matMV, matProj, ViewPort,
//                            &dScrX,&dScrY,&dScrZ);

//                        pvSrcPos[nVrt][0]	= float(dScrX);
//                        pvSrcPos[nVrt][1]	= float(dScrY);
//                        pvSrcPos[nVrt][2]	= 0;
//                    }
//                }

//                // if the pixel is not in the area of the triangle, skip this triangle
//                if(igtPointInTriangle2D(vDstPos, pvSrcPos))
//                {
//                    //	Keep the best triangle info
//                    nNearestTriIdx	= iTri;
//                    dNearestTriDist	= dTriDist;

//                    // Get the nearest vertices
//                    float fNearestVrtDist  = 1000;
//                    for( int nVrt=0; nVrt<3; nVrt++)
//                    {
//                        // get the vertex
//                        int vIdx = pTri[nVrt];

//                        // the distance between the cursor and the projected vertex
//                        float fLen = (len(vDstPos - pvSrcPos[nVrt]));

//                        if ( fLen < fNearestVrtDist )
//                        {
//                            fNearestVrtDist	= fLen;
//                            nNearestVrtIdx	= vIdx;
//                        }
//                    }
//                    err = fNearestVrtDist;
//                }
//            }
//        }

//        return nNearestVrtIdx;
//    }
//}

void MeshWidget::interpolateScreenPts(std::vector<QPoint> &screen_pts)
{
    int np = screen_pts.size();
    if (np > 1)
    {
        std::vector<QPoint> pts_inter;
        for (int i=0; i<np-1; ++i)
        {
            QPoint p1 = screen_pts[i];
            QPoint p2 = screen_pts[i+1];
            float d = trimesh::len(trimesh::vec2(p1.x(), p1.y())-trimesh::vec2(p2.x(), p2.y()));

            pts_inter.push_back(p1);

            if (d > mouse_size_)
            {
                int n_iter = int(d/mouse_size_);
                float ratio = 1.f/(n_iter+1);
                for (int j=1; j<=n_iter; ++j)
                {
                    pts_inter.push_back(p1*(j*ratio) + p2*(1.f-j*ratio));
                }
            }
        }
        pts_inter.push_back(screen_pts[np-1]);

        screen_pts = pts_inter;
    }
}

int MeshWidget::pickPointOnMesh(QPoint p, const float &d_thres)
{
    assert (proj_vec2_.size() == proj_norms_.size());

    mesh_->need_normals();
    trimesh::vec3 dir(0.f, 0.f, 1.f);
    trimesh::vec2 p_screen(p.rx(), this->height()-p.ry());
    int nv = mesh_->vertices.size();
    float min_d = 1e8;
    int min_id = -1;

    if (src_pick_id_ < 0)  // has picked the point on the contour
    {
        // project mesh vertices
        double mat_mv[16], mat_proj[16];
        int view_port[4];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(getBallMatrix());
        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
        glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
        glGetIntegerv(GL_VIEWPORT, view_port);

        Eigen::MatrixXf m(4, 4);
        int k = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m(i, j) = mat_mv[k];
                k++;
    //                std::cout<<m(i, j)<<' ';
            }
    //            std::cout<<std::endl;
        }
        m = m.inverse();

        for (int i=0; i<nv; ++i)
        {
            GLdouble x=0, y=0, z=0;
            gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
            proj_vec2_.push_back(trimesh::vec2(x, y));

            Eigen::Vector4f n(mesh_->normals[i][0], mesh_->normals[i][1], mesh_->normals[i][2], 1.f);
            n = m*n;
            proj_norms_.push_back(trimesh::point(n[0], n[1], n[2]));

            if (trimesh::dot(proj_norms_[i], dir) > 0.f)
            {
                float d = trimesh::len(proj_vec2_[i]-p_screen);
                if (d < min_d)
                {
                    min_d = d;
                    min_id = i;
                }
            }
        }
        glPopMatrix();
    }
    else
    {
        for (int i=0; i<nv; ++i)
        {
            if (trimesh::dot(proj_norms_[i], dir) > 0.f)
            {
                float d = trimesh::len(proj_vec2_[i]-p_screen);
                if (d < min_d)
                {
                    min_d = d;
                    min_id = i;
                }
            }
        }
    }

    if (min_d <= d_thres)
    {
        return min_id;
    }
    else
    {
        return -1;
    }
}

int MeshWidget::pickPointOnContour(QPoint p, const float &d_thres)
{
    int nv = mesh_->vertices.size();

    // project mesh vertices
    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    Eigen::MatrixXf m(4, 4);
    int k = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m(i, j) = mat_mv[k];
            k++;
//                std::cout<<m(i, j)<<' ';
        }
//            std::cout<<std::endl;
    }
    m = m.inverse();

    proj_vec2_.clear();
    proj_norms_.clear();
    for (int i=0; i<nv; ++i)
    {
        GLdouble x=0, y=0, z=0;
        gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
        proj_vec2_.push_back(trimesh::vec2(x, y));

        Eigen::Vector4f n(mesh_->normals[i][0], mesh_->normals[i][1], mesh_->normals[i][2], 1.f);
        n = m*n;
        proj_norms_.push_back(trimesh::point(n[0], n[1], n[2]));
    }
    glPopMatrix();

    mesh_->need_normals();
    trimesh::vec3 dir(0.f, 0.f, 1.f);
    trimesh::vec2 p_screen(p.rx(), this->height()-p.ry());
    float min_d = 1e8;
    int min_id = -1;
    for (int i=0; i<n_predefined_contour_; ++i)
    {
        for (int j=0; j<predefined_contour_idx_[i].size(); ++j)
        {
            const int id = predefined_contour_idx_[i][j];
            if (trimesh::dot(proj_norms_[id], dir) > 0.f)
            {
                float d = trimesh::len(proj_vec2_[id]-p_screen);
                if (d < min_d)
                {
                    min_d = d;
                    min_id = id;
                    contour_pick_id_ = i;
                }
            }
        }
    }

    if (min_d <= d_thres)
    {
        return min_id;
    }
    else
    {
        contour_pick_id_ = -1;
        return -1;
    }
}

void MeshWidget::modifyContour(const int influence_step)
{
    if (src_pick_id_>=0 && tar_pick_id_>=0 && src_pick_id_!=tar_pick_id_ && contour_pick_id_>=0)
    {
        b_contour_changed_ = true;

        const std::vector<int> &idx = predefined_contour_idx_bk_[predefined_contour_idx_bk_.size()-1][contour_pick_id_];
        int landmark_id = idx.size()/2;
        for (int i=0; i<idx.size(); ++i)
        {
            if (idx[i] == src_pick_id_)
            {
                landmark_id = i;
            }
        }

        std::vector<int> picked_idx;
        std::vector<int> new_contour_idx;
        if (landmark_id > influence_step)
        {
            for (int i=0; i<landmark_id-influence_step; ++i)
            {
                new_contour_idx.push_back(idx[i]);
            }

            picked_idx.push_back(idx[landmark_id-influence_step]);
            picked_idx.push_back(tar_pick_id_);
        }
        else
        {
            if (src_pick_id_ != idx[0])
            {
                picked_idx.push_back(idx[0]);
                picked_idx.push_back(tar_pick_id_);
            }
            else
            {
                picked_idx.push_back(tar_pick_id_);
            }
        }

        if (idx.size()-1-landmark_id > influence_step)
        {
            picked_idx.push_back(idx[landmark_id+influence_step]);
        }
        else
        {
            if (src_pick_id_ != idx[idx.size()-1])
            {
                picked_idx.push_back(idx[idx.size()-1]);
            }
            else
            {
                picked_idx.push_back(tar_pick_id_);
            }
        }

        for (int i=0; i<picked_idx.size()-1; ++i)
        {
            const int id1 = picked_idx[i];
            const int id2 = picked_idx[i+1];
            if (id1 != id2)
            {
                std::vector<int> route_p_id;
                dijk_short_path_->getDijkstraPath(route_p_id, id1, id2, mesh_, true);
                if (route_p_id.size() > 0)
                {
                    if (i == 0)
                    {
                        new_contour_idx.push_back(route_p_id[0]);
                    }
                    for (int j=1; j<route_p_id.size(); ++j)
                    {
                        new_contour_idx.push_back(route_p_id[j]);
                    }
                }
            }
        }

        if (idx.size()-1-landmark_id > influence_step)
        {
            for(int i=landmark_id+influence_step+1; i<idx.size(); ++i)
            {
                new_contour_idx.push_back(idx[i]);
            }
        }

        predefined_contour_idx_[contour_pick_id_] = new_contour_idx;
        predefined_contour_idx_bk_.push_back(predefined_contour_idx_);

        updateGL();
    }
}

void MeshWidget::pickMeshFacesWithLineDrawn(std::vector<QPoint> &screen_pts, std::vector<trimesh::point> &proj_vts, std::vector<int> &fidx)
{
    if (screen_pts.size()>0 && mesh_)
    {
        // calculate bounding box of line drawn
        int npts = screen_pts.size();
        QPoint p_max(0, 0), p_min(10000, 10000);
        for (int i=0; i<npts; ++i)
        {
            screen_pts[i].ry() = this->height()-screen_pts[i].ry();

            if (screen_pts[i].rx() > p_max.rx())
            {
                p_max.rx() = screen_pts[i].rx();
            }
            if (screen_pts[i].rx() < p_min.rx())
            {
                p_min.rx() = screen_pts[i].rx();
            }

            if (screen_pts[i].ry() > p_max.ry())
            {
                p_max.ry() = screen_pts[i].ry();
            }
            if (screen_pts[i].ry() < p_min.ry())
            {
                p_min.ry() = screen_pts[i].ry();
            }
        }
//        std::cout<<"p_max: "<<p_max.rx()<<" "<<p_max.ry()<<std::endl;
//        std::cout<<"p_min: "<<p_min.rx()<<" "<<p_min.ry()<<std::endl;

        // project mesh vertices
        double matMV[16], matProj[16];
        int viewPort[4];
//        GLdouble camera_pos[3];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(getBallMatrix ());
        glGetDoublev(GL_MODELVIEW_MATRIX, matMV);
        glGetDoublev(GL_PROJECTION_MATRIX, matProj);
        glGetIntegerv(GL_VIEWPORT, viewPort);

//        assert(gluUnProject((viewPort[2]-viewPort[0])/2 , (viewPort[3]-viewPort[1])/2, 0.f, matMV, matProj, viewPort, &camera_pos[0],&camera_pos[1],&camera_pos[2])==true);

        Eigen::MatrixXf m(4, 4);
        int k = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m(i, j) = matMV[k];
                k++;
//                std::cout<<m(i, j)<<' ';
            }
//            std::cout<<std::endl;
        }
        m = m.inverse();

        int nv = mesh_->vertices.size();
        for (int i=0; i<nv; ++i)
        {
            GLdouble x=0, y=0, z=0;
            gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], matMV, matProj, viewPort, &x, &y, &z);
            proj_vts.push_back(trimesh::point(x, y, z));

            Eigen::Vector4f v(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], 1.f);
            v = m*v;

            proj_vts[i][2] = v[2]*400;
        }

//        trimesh::vec3 DOF(matMV[2], matMV[6], matMV[10]);
//        std::cout<<"DOF: "<<DOF<<std::endl;

        glPopMatrix();

        // choose candidate triangle faces of the mesh
        trimesh::vec3 view_dir(0.f, 0.f, -1.f);
        int nf = mesh_->faces.size();
        std::vector<int> tmp_f_idx;
        for (int i=0; i<nf; ++i)
        {
            const trimesh::TriMesh::Face &f = mesh_->faces[i];

            bool is_pick = false;
            for (int j=0; j<3; ++j)
            {
                const trimesh::point &p = proj_vts[f[j]];
                if (p[0]<p_max.rx() && p[0]>p_min.rx() && p[1]<p_max.ry() && p[1]>p_min.ry())
                {
                    is_pick = true;
                    break;
                }
            }
            if (is_pick)
            {
                const trimesh::point &v0 = proj_vts[f[0]];
                const trimesh::point &v1 = proj_vts[f[1]];
                const trimesh::point &v2 = proj_vts[f[2]];
                trimesh::vec3 v1v0 = v1 - v0;
                trimesh::vec3 v2v0 = v2 - v0;
                if (trimesh::cross(v1v0, v2v0).dot(-view_dir) > 0)
                {
                    tmp_f_idx.push_back(i);
                }
            }
        }
        // double check using ray tracing intersection
        int nf_pick = tmp_f_idx.size();
        std::vector<bool> is_picked(nf_pick, true);
        for (int i=0; i<nf_pick; ++i)
        {
            const trimesh::TriMesh::Face &f1 = mesh_->faces[tmp_f_idx[i]];
            bool is_insert = false;
            for (int k=0; k<3; ++k)
            {
                trimesh::point ray_ori = proj_vts[f1[k]];
                for (int j=0; j<nf_pick; ++j)
                {
                    if (i != j)
                    {
                        float d = 0;
                        const trimesh::TriMesh::Face &f2 = mesh_->faces[tmp_f_idx[j]];
                        if (checkRayTriangleIntersected2(ray_ori, -1.f*view_dir, proj_vts[f2[0]], proj_vts[f2[1]], proj_vts[f2[2]], d))
                        {
                            if (d > 1e-3)
                            {
                                is_insert = true;
                                break;
                            }
                        }
                    }

                    if (is_insert)
                    {
                        break;
                    }
                }

                if (is_insert)
                {
                    break;
                }
            }

            if (is_insert)
            {
                is_picked[i] = false;
            }
        }

        fidx.clear();
        for (int i=0; i<nf_pick; ++i)
        {
            if (is_picked[i])
            {
                fidx.push_back(tmp_f_idx[i]);
            }
        }

        tmp_f_idx.clear();

//        // testing picked faces
//        trimesh::TriMesh mesh_area;
//        for (int i=0; i<fidx.size(); ++i)
//        {
//            const trimesh::TriMesh::Face &f = mesh_->faces[fidx[i]];
//            mesh_area.vertices.push_back(mesh_->vertices[f[0]]);
//            mesh_area.vertices.push_back(mesh_->vertices[f[1]]);
//            mesh_area.vertices.push_back(mesh_->vertices[f[2]]);
//            mesh_area.faces.push_back(trimesh::TriMesh::Face(3*i, 3*i+1, 3*i+2));
//        }
//        mesh_area.write("mesh_area.obj");
    }
}

void MeshWidget::subdivideMeshPart(std::vector<QPoint> &screen_pts, const float &d_thres)
{
    std::vector<int> fidx;
    std::vector<trimesh::point> proj_vts;
    float d_thres_face = d_thres+1;
    pickVisibleFacesWithScreenPoints(screen_pts, proj_vts, fidx, d_thres_face);  // screen_pts.ry() has been set to be in OpenGL

    base_processor_->subdivideMeshPartMPS(mesh_, fidx);

    updateModelView();
}

void MeshWidget::loadPredefinedContourVertexIndex()
{
    predefined_contour_idx_.clear();
    predefined_contour_idx_bk_.clear();
    b_contour_changed_ = false;

    std::string file_path = "./utils/";
    id_contour_.clear();
    b_contour_.resize(4962, false);
    for (int i=1; i<=n_predefined_contour_; ++i)
    {
        std::vector<int> v_idx;
        std::ifstream fr(file_path+"curve"+std::to_string(i)+"_idx.txt");
        int idx = 0;
        while(fr >> idx)
        {
            v_idx.push_back(idx);

            if (!b_contour_[idx])
            {
                id_contour_.push_back(idx);
                b_contour_[idx] = true;
            }
        }
        fr.close();

        predefined_contour_idx_.push_back(v_idx);
        std::cout<<"load pre-defined contour "<<i<<"; vertex number: "<<v_idx.size()<<std::endl;
    }
}

void MeshWidget::updatePredefinedContour(const int sm_iter)
{
    predefined_contour_idx_bk_.push_back(predefined_contour_idx_);

    id_contour_.clear();
    b_contour_.resize(mesh_->vertices.size(), false);
    for (int i=0; i<n_predefined_contour_; ++i)
    {
        for (int j=0; j<predefined_contour_idx_[i].size(); ++j)
        {
            const int id = predefined_contour_idx_[i][j];
            if (!b_contour_[id])
            {
                b_contour_[id] = true;
                id_contour_.push_back(id);
            }
        }
    }

    if (contour_deformer_)
    {
        delete contour_deformer_;
        contour_deformer_ = nullptr;
    }

    contour_deformer_ = new LaplacianDeformation(mesh_, 0.2);
    contour_deformer_->preCalcDeformationSolver(b_contour_);

    if (sm_iter > 0)
    {
        const std::vector<int> &v_idx = predefined_contour_idx_[contour_pick_id_];
        const int nv_idx = v_idx.size();
        for (int k=0; k<sm_iter; ++k)
        {
            for (int i=1; i<nv_idx-1; ++i)
            {
                mesh_->vertices[v_idx[i]] = (mesh_->vertices[v_idx[i-1]] + mesh_->vertices[v_idx[i+1]])/2.f;
            }
        }

        contour_deformer_->doLaplacianDeformation(id_contour_);

        updateModelView();
    }
    else
    {
        updateGL();
    }
}

void MeshWidget::updatePredefinedContour(const std::vector<int> contour_idx, const int sm_iter)
{
    id_contour_.clear();
    b_contour_.clear();
    if (contour_deformer_)
    {
        delete contour_deformer_;
        contour_deformer_ = nullptr;
    }

    b_contour_.resize(mesh_->vertices.size(), false);
    for (int i=0; i<n_predefined_contour_; ++i)
    {
        for (int j=0; j<predefined_contour_idx_[i].size(); ++j)
        {
            const int id = predefined_contour_idx_[i][j];
            if (!b_contour_[id])
            {
                b_contour_[id] = true;
                id_contour_.push_back(id);
            }
        }
    }

    contour_deformer_ = new LaplacianDeformation(mesh_, 0.2);
    contour_deformer_->preCalcDeformationSolver(b_contour_);

    if (sm_iter > 0)
    {
        for (int i=0; i<contour_idx.size(); ++i)
        {
            const std::vector<int> &v_idx = predefined_contour_idx_[contour_idx[i]];
            const int nv_idx = v_idx.size();
            for (int k=0; k<sm_iter; ++k)
            {
                for (int i=1; i<nv_idx-1; ++i)
                {
                    mesh_->vertices[v_idx[i]] = (mesh_->vertices[v_idx[i-1]] + mesh_->vertices[v_idx[i+1]])/2.f;
                }
            }
        }

        contour_deformer_->doLaplacianDeformation(id_contour_);

        updateModelView();
    }
    else
    {
        updateGL();
    }
}

void MeshWidget::smoothModifiedContour(const int sm_iter)
{
    if (contour_pick_id_ >= 0)
    {
        const std::vector<int> &v_idx = predefined_contour_idx_[contour_pick_id_];
        const int nv_idx = v_idx.size();
        for (int k=0; k<sm_iter; ++k)
        {
            for (int i=1; i<nv_idx-1; ++i)
            {
                mesh_->vertices[v_idx[i]] = (mesh_->vertices[v_idx[i-1]] + mesh_->vertices[v_idx[i+1]])/2.f;
            }
        }

        contour_deformer_->doLaplacianDeformation(id_contour_);

        updateModelView();
    }
}

std::vector<int> MeshWidget::findCommonNeighbor(const int &id1, const int &id2)
{
    mesh_->need_neighbors();

    std::vector<int> commom_idx;
    const std::vector<int> &nei1 = mesh_->neighbors[id1];
    const std::vector<int> &nei2 = mesh_->neighbors[id2];
    for (int i=0; i<nei1.size(); ++i)
    {
        for (int j=0; j<nei2.size(); ++j)
        {
            if (nei2[j] == nei1[i])
            {
                commom_idx.push_back(nei1[i]);
                break;
            }
        }
    }

    return commom_idx;
}

int MeshWidget::searchBestPointWithFeatureProbability(const int id)
{
    assert(id >= 0);

    mesh_->need_neighbors();

    float max_val = mesh_->colors[id][0];
    float max_id = id;
    const std::vector<int> &nei_idx1 = mesh_->neighbors[id];
    for (int i=0; i<nei_idx1.size(); ++i)
    {
        if (mesh_->colors[nei_idx1[i]][0] > max_val)
        {
            max_val = mesh_->colors[nei_idx1[i]][0];
            max_id = nei_idx1[i];
        }

        const std::vector<int> &nei_idx2 = mesh_->neighbors[nei_idx1[i]];
        for (int j=0; j<nei_idx2.size(); ++j)
        {
            if (mesh_->colors[nei_idx2[j]][0] > max_val)
            {
                max_val = mesh_->colors[nei_idx2[j]][0];
                max_id = nei_idx2[j];
            }
        }
    }

    return max_id;
}

void MeshWidget::autoModifyPredefinedContours(const int sm_iter)
{
    assert(mesh_->vertices.size() == mesh_->colors.size());

    b_contour_changed_ = true;

    const int n_iter = 3;
    mesh_->need_neighbors();
    for (int m=0; m<n_iter; ++m)
    {
        for (int i=2; i<n_predefined_contour_; ++i)
        {
            const std::vector<int> &old_contour_idx = predefined_contour_idx_[i];
            const int n = old_contour_idx.size();
            const int n_sample = 7; // 7 nodes
            int step = (n+1)/(n_sample-1);
            if (step == 0)
            {
                step = 1;
            }

            std::vector<int> idx_sample;
            std::vector<int> idx_id;
            for (int j=0; j<n; j+=step)
            {
                idx_sample.push_back(old_contour_idx[j]);
                idx_id.push_back(j);
            }
            if (idx_sample.size() < n_sample)
            {
                idx_sample.push_back(old_contour_idx[n-1]);
                idx_id.push_back(n-1);
            }
            assert(idx_sample.size() == n_sample);

            std::vector<int> new_contour_idx;
            for (int j=0; j<n_sample-1; ++j)
            {
                // look for better point in 2-ring neighborhoods
                int id1 = searchBestPointWithFeatureProbability(idx_sample[j]);
                int id2 = searchBestPointWithFeatureProbability(idx_sample[j+1]);
                if ((id1 != id2) && (id1 != idx_sample[j] || id2 != idx_sample[j+1]))
                {
                    std::vector<int> route_p_id;
                    dijk_short_path_->getDijkstraPath(route_p_id, id1, id2, mesh_, true);
                    if (route_p_id.size() > 0)
                    {
                        if (j == 0)
                        {
                            new_contour_idx.push_back(route_p_id[0]);
                        }
                        for (int j=1; j<route_p_id.size(); ++j)
                        {
                            new_contour_idx.push_back(route_p_id[j]);
                        }
                    }
                }
                else
                {
                    if (j == 0)
                    {
                        new_contour_idx.push_back(old_contour_idx[idx_id[0]]);
                    }
                    for (int k=idx_id[j]+1; k<=idx_id[j+1]; ++k)
                    {
                        new_contour_idx.push_back(old_contour_idx[k]);
                    }
                }
            }

            predefined_contour_idx_[i] = new_contour_idx;
        }
    }

    for (int i=0; i<n_predefined_contour_; ++i)
    {
        std::cout<<"pre-defined contour id num: "<<predefined_contour_idx_[i].size()<<std::endl;
    }

    if (sm_iter > 0)
    {
        std::vector<int> contour_idx;
        contour_idx.push_back(2);   // for the specified contours (ears)
        contour_idx.push_back(3);
        updatePredefinedContour(contour_idx, sm_iter);
    }
}

void MeshWidget::smoothPredefinedContours(int sm_iter)
{
    if (!contour_deformer_ && mesh_)    // only do once
    {
//        autoModifyPredefinedContours(0);    // only do once

        id_contour_.clear();
        b_contour_.clear();
        b_contour_.resize(mesh_->vertices.size(), false);
        for (int i=0; i<n_predefined_contour_; ++i)
        {
            for (int j=0; j<predefined_contour_idx_[i].size(); ++j)
            {
                const int id = predefined_contour_idx_[i][j];
                if (!b_contour_[id])
                {
                    b_contour_[id] = true;
                    id_contour_.push_back(id);
                }
            }
        }

        contour_deformer_ = new LaplacianDeformation(mesh_, 0.2);
        contour_deformer_->preCalcDeformationSolver(b_contour_);

        for (int k=0; k<sm_iter; ++k)
        {
            for (int j=0; j<n_predefined_contour_; ++j)
            {
                const int n = predefined_contour_idx_[j].size();
//                std::vector<trimesh::point> sm_contour;
//                sm_contour.push_back(mesh_->vertices[predefined_contour_idx_[j][0]]);
//                for (int i=1; i<n-1; ++i)
//                {
//                    sm_contour.push_back((mesh_->vertices[predefined_contour_idx_[j][i-1]] + mesh_->vertices[predefined_contour_idx_[j][i+1]])/2.f);
//                }
//                sm_contour.push_back(mesh_->vertices[predefined_contour_idx_[j][n-1]]);

//                if (j < 2)
//                {
//                    sm_contour[0] = (mesh_->vertices[predefined_contour_idx_[j][n-1]] + mesh_->vertices[predefined_contour_idx_[j][1]])/2.f;
//                    sm_contour[n-1] = (mesh_->vertices[predefined_contour_idx_[j][n-2]] + mesh_->vertices[predefined_contour_idx_[j][0]])/2.f;
//                }

//                for (int i=0; i<n; ++i)
//                {
//                    mesh_->vertices[predefined_contour_idx_[j][i]] = sm_contour[i];
//                }

                if (j < 2)
                {
                    mesh_->vertices[predefined_contour_idx_[j][0]] = (mesh_->vertices[predefined_contour_idx_[j][n-1]] + mesh_->vertices[predefined_contour_idx_[j][1]])/2.f;
                    mesh_->vertices[predefined_contour_idx_[j][n-1]] = (mesh_->vertices[predefined_contour_idx_[j][n-2]] + mesh_->vertices[predefined_contour_idx_[j][0]])/2.f;
                }
                for (int i=1; i<n-1; ++i)
                {
                    mesh_->vertices[predefined_contour_idx_[j][i]] = (mesh_->vertices[predefined_contour_idx_[j][i-1]] + mesh_->vertices[predefined_contour_idx_[j][i+1]])/2.f;
                }
            }
        }

        contour_deformer_->doLaplacianDeformation(id_contour_);
        std::cout<<"smoothed contours"<<std::endl;
    }
}

int MeshWidget::pickSurfaceVertexWithScreenPoint(const std::vector<trimesh::point> &proj_vts, const std::vector<int> &vidx_sel, const QPoint &p)
{
    const int nv_sel = vidx_sel.size();
    float min_d = 100000;
    int min_id = -1;
    for (int i=0; i<nv_sel; ++i)
    {
        float d = trimesh::len(trimesh::vec2(proj_vts[vidx_sel[i]][0], proj_vts[vidx_sel[i]][1])-trimesh::vec2(p.x(), p.y()));
        if (d < min_d)
        {
            min_d = d;
            min_id = vidx_sel[i];
        }
    }

    return min_id;
}

void MeshWidget::pickSurfaceCurveWithLineDrawn(std::vector<QPoint> screen_line_pts, const float d_thres)
{
    int nv_line = screen_line_pts.size();
    if (nv_line > 0)
    {
        const int nv_ori = mesh_->vertices.size();

        std::vector<int> fidx;
        std::vector<trimesh::point> proj_vts;
        pickVisibleFacesWithScreenPoints(screen_line_pts, proj_vts, fidx, d_thres);  // screen_pts.ry() has been set to be in OpenGL
        fidx = base_processor_->subdivideMeshPartMPS(mesh_, fidx, b_face_sub_);

        const int nv = mesh_->vertices.size();

        // project mesh vertices
        double mat_mv[16], mat_proj[16];
        int view_port[4];
    //        GLdouble camera_pos[3];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(getBallMatrix());
        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
        glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
        glGetIntegerv(GL_VIEWPORT, view_port);
        for (int i=nv_ori; i<nv; ++i)
        {
            GLdouble x=0, y=0, z=0;
            gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
            proj_vts.push_back(trimesh::point(x, y, z));    // please note that the z value is not corresponding to the former ones (not *height), here we just use x, y data
        }
        glPopMatrix();

        b_v_fixed_.clear();
        b_v_fixed_.resize(nv, true);
        std::vector<int> vidx_sel;      // id_map of part2ori_sel
        int nv_sel = 0;
        std::vector<int> id_map_ori2part(nv, -1);
        trimesh::TriMesh part_mesh;
        for (int i=0; i<fidx.size(); ++i)
        {
            const trimesh::TriMesh::Face &f = mesh_->faces[fidx[i]];
            for (int j=0; j<3; ++j)
            {
                if (b_v_fixed_[f[j]])
                {
                    vidx_sel.push_back(f[j]);
                    id_map_ori2part[f[j]] = nv_sel;
                    part_mesh.vertices.push_back(mesh_->vertices[f[j]]);

                    b_v_fixed_[f[j]] = false;
                    nv_sel++;
                }
            }

            part_mesh.faces.push_back(trimesh::TriMesh::Face(id_map_ori2part[f[0]], id_map_ori2part[f[1]], id_map_ori2part[f[2]]));
        }
        part_mesh.need_neighbors();
//        part_mesh.write("part_mesh.ply");   // for testing

        v_pick_idx_.clear();
        if (nv_line > 1)
        {
            int step = nv_line/8;
            if (step == 0)
            {
                step = 1;
            }
            int start_id = 0;
            int end_id = step;
            int id1 = -1, id2 = -1;
            do {
                if (id1 < 0)
                {
                    id1 = pickSurfaceVertexWithScreenPoint(proj_vts, vidx_sel, screen_line_pts[start_id]);
                }
                id2 = pickSurfaceVertexWithScreenPoint(proj_vts, vidx_sel, screen_line_pts[end_id]);

                if (id1 != id2)
                {
                    std::vector<int> route_p_id;
                    dijk_short_path_->getDijkstraPath(route_p_id, id_map_ori2part[id1], id_map_ori2part[id2], &part_mesh, false);
                    if (route_p_id.size() > 0)
                    {
                        for (int i=0; i<route_p_id.size(); ++i)
                        {
                            const int id = vidx_sel[route_p_id[i]];
                            if (!b_v_fixed_[id])
                            {
                                b_v_fixed_[id] = true;
                                v_pick_idx_.push_back(id);
                            }
                        }
                    }

                    start_id = end_id + 1;
                    end_id = start_id + step;
                    if (end_id >= nv_line)
                    {
                        end_id = nv_line-1;
                    }

                    id1 = -1;
                    id2 = -1;
                }
                else
                {
                    if (end_id == nv_line-1)
                    {
                        if (!b_v_fixed_[id1])
                        {
                            b_v_fixed_[id1] = true;
                            v_pick_idx_.push_back(id1);
                        }

                        break;
                    }
                    else
                    {
                        end_id = end_id + step;
                        if (end_id >= nv_line)
                        {
                            end_id = nv_line-1;
                        }
                    }
                }
                std::cout<<start_id<<", "<<end_id<<std::endl;
            }while ((start_id<end_id) && (start_id<nv_line));
        }
        else
        {
            const int id = pickSurfaceVertexWithScreenPoint(proj_vts, vidx_sel, screen_line_pts[0]);
            b_v_fixed_[id] = true;
            v_pick_idx_.push_back(id);
        }
    }

//    // smooth the picked vertices
//    for (int k=0; k<3; ++k)
//    {
//        for (int i=1; i<v_pick_idx_.size()-1; ++i)
//        {
//            mesh_->vertices[v_pick_idx_[i]] = (mesh_->vertices[v_pick_idx_[i-1]] + mesh_->vertices[v_pick_idx_[i+1]])/2.f;
//        }
//    }

    initLaplacianDeformer(0.2);

    updateGL();
}

void MeshWidget::pickAreaFacesWithScreenPoints(std::vector<QPoint> &screen_pts, std::vector<trimesh::vec2> &proj_vts, std::vector<int> &fidx, const float &d_thres)
{
    if (screen_pts.size()>0 && mesh_)
    {
        int npts = screen_pts.size();
        for (int i=0; i<npts; ++i)
        {
            screen_pts[i].ry() = this->height()-screen_pts[i].ry();
        }

        // project mesh vertices
        double mat_mv[16], mat_proj[16];
        int view_port[4];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(getBallMatrix());
        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
        glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
        glGetIntegerv(GL_VIEWPORT, view_port);

        proj_vts.clear();
        int nv = mesh_->vertices.size();
        for (int i=0; i<nv; ++i)
        {
            GLdouble x=0, y=0, z=0;
            gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
            proj_vts.push_back(trimesh::vec2(x, y));
        }

        glPopMatrix();

        // choose triangle faces of the mesh
        int nf = mesh_->faces.size();
        fidx.clear();
        for (int i=0; i<nf; ++i)
        {
            const trimesh::TriMesh::Face &f = mesh_->faces[i];

            bool is_pick = false;
            for (int j=0; j<3; ++j)
            {
                for (int k=0; k<npts; ++k)
                {
                    if (trimesh::len(trimesh::vec2(screen_pts[k].rx(), screen_pts[k].ry()) - trimesh::vec2(proj_vts[f[j]][0], proj_vts[f[j]][1])) <= d_thres)
                    {
                        is_pick = true;
                        break;
                    }
                }

                if (is_pick)
                {
                    break;
                }
            }

            if (is_pick)
            {
                fidx.push_back(i);
            }
        }
    }
}

void MeshWidget::pickAreaVertexWithLineDrawn(std::vector<QPoint> &screen_pts, std::vector<trimesh::vec2> &proj_vts, std::vector<int> &vidx, const float &d_thres, float fd_thres_time)
{
    std::vector<int> fidx;
    pickAreaFacesWithScreenPoints(screen_pts, proj_vts, fidx, d_thres*fd_thres_time);

    // add selected vertices
    int nv_ori = mesh_->vertices.size();
    std::vector<bool> b_v_selected(nv_ori, false);
    for (int i=0; i<fidx.size(); ++i)
    {
        const trimesh::TriMesh::Face &f = mesh_->faces[i];
        for (int j=0; j<3; ++j)
        {
            if (!b_v_selected[f[j]])
            {
                vidx.push_back(f[j]);
                b_v_selected[f[j]] = true;
            }
        }
    }

    fidx = base_processor_->subdivideMeshPartMPS(mesh_, fidx, b_face_sub_);

    // project mesh vertices
    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    int nv = mesh_->vertices.size();
    for (int i=nv_ori; i<nv; ++i)
    {
        GLdouble x=0, y=0, z=0;
        gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
        proj_vts.push_back(trimesh::vec2(x, y));

        vidx.push_back(i);
    }

    glPopMatrix();
}

void MeshWidget::pickVisibleFacesWithScreenPoints(std::vector<QPoint> &screen_pts, std::vector<trimesh::point> &proj_vts, std::vector<int> &fidx, const float &d_thres)
{
    if (screen_pts.size()>0 && mesh_)
    {
        int npts = screen_pts.size();
        for (int i=0; i<npts; ++i)
        {
            screen_pts[i].ry() = this->height()-screen_pts[i].ry();
        }

        // project mesh vertices
        double mat_mv[16], mat_proj[16];
        int view_port[4];
//        GLdouble camera_pos[3];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(getBallMatrix());
        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
        glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
        glGetIntegerv(GL_VIEWPORT, view_port);

        Eigen::MatrixXf m(4, 4);
        int k = 0;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m(i, j) = mat_mv[k];
                k++;
//                std::cout<<m(i, j)<<' ';
            }
//            std::cout<<std::endl;
        }
        m = m.inverse();

        proj_vts.clear();
        int nv = mesh_->vertices.size();
        for (int i=0; i<nv; ++i)
        {
            GLdouble x=0, y=0, z=0;
            gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
            proj_vts.push_back(trimesh::point(x, y, z));

            Eigen::Vector4f v(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], 1.f);
            v = m*v;

            proj_vts[i][2] = v[2]*this->width();
        }

        glPopMatrix();

        // choose candidate triangle faces of the mesh
        trimesh::vec3 view_dir(0.f, 0.f, -1.f);
        int nf = mesh_->faces.size();
        std::vector<int> tmp_f_idx;
        for (int i=0; i<nf; ++i)
        {
            const trimesh::TriMesh::Face &f = mesh_->faces[i];

            bool is_pick = false;
            for (int j=0; j<3; ++j)
            {
                for (int k=0; k<npts; ++k)
                {
                    if (trimesh::len(trimesh::vec2(screen_pts[k].rx(), screen_pts[k].ry()) - trimesh::vec2(proj_vts[f[j]][0], proj_vts[f[j]][1])) <= d_thres)
                    {
                        is_pick = true;
                        break;
                    }
                }

                if (is_pick)
                {
                    break;
                }
            }
            if (is_pick)
            {
                const trimesh::point &v0 = proj_vts[f[0]];
                const trimesh::point &v1 = proj_vts[f[1]];
                const trimesh::point &v2 = proj_vts[f[2]];
                trimesh::vec3 v1v0 = v1 - v0;
                trimesh::vec3 v2v0 = v2 - v0;
                if (trimesh::cross(v1v0, v2v0).dot(-view_dir) > 0)
                {
                    tmp_f_idx.push_back(i);
                }
            }
        }
        // double check using ray tracing intersection
        int nf_pick = tmp_f_idx.size();
        std::vector<bool> is_picked(nf_pick, true);
        for (int i=0; i<nf_pick; ++i)
        {
            const trimesh::TriMesh::Face &f1 = mesh_->faces[tmp_f_idx[i]];
            bool is_insert = false;
            for (int k=0; k<3; ++k)
            {
                trimesh::point ray_ori = proj_vts[f1[k]];
                for (int j=0; j<nf_pick; ++j)
                {
                    if (i != j)
                    {
                        float d = 0;
                        const trimesh::TriMesh::Face &f2 = mesh_->faces[tmp_f_idx[j]];
                        if (checkRayTriangleIntersected2(ray_ori, -1.f*view_dir, proj_vts[f2[0]], proj_vts[f2[1]], proj_vts[f2[2]], d))
                        {
                            if (d > 1e-4)
                            {
                                is_insert = true;
                                break;
                            }
                        }
                    }

                    if (is_insert)
                    {
                        break;
                    }
                }

                if (is_insert)
                {
                    break;
                }
            }

            if (is_insert)
            {
                is_picked[i] = false;
            }
        }

        fidx.clear();
        for (int i=0; i<nf_pick; ++i)
        {
            if (is_picked[i])
            {
                fidx.push_back(tmp_f_idx[i]);
            }
        }
    }
}

void MeshWidget::pickVisibleVertexWithScreenPoints(std::vector<QPoint> &screen_pts, const float &d_thres, float fd_thres_time)
{
    std::vector<int> fidx;
    std::vector<trimesh::point> proj_vts;
    float d_thres_face = d_thres*fd_thres_time;
    pickVisibleFacesWithScreenPoints(screen_pts, proj_vts, fidx, d_thres_face);  // screen_pts.ry() has been set to be in OpenGL

    int nv_ori = mesh_->vertices.size();
    if (mouse_size_<=8)
    {
        fidx = base_processor_->subdivideMeshPartMPS(mesh_, fidx, b_face_sub_, 2);
    }
    else
    {
        fidx = base_processor_->subdivideMeshPartMPS(mesh_, fidx, b_face_sub_);
    }

    // project mesh vertices
    double mat_mv[16], mat_proj[16];
    int view_port[4];
//        GLdouble camera_pos[3];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    Eigen::MatrixXf m(4, 4);
    int k = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m(i, j) = mat_mv[k];
            k++;
//                std::cout<<m(i, j)<<' ';
        }
//            std::cout<<std::endl;
    }
    m = m.inverse();

    int nv = mesh_->vertices.size();
    for (int i=nv_ori; i<nv; ++i)
    {
        GLdouble x=0, y=0, z=0;
        gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
        proj_vts.push_back(trimesh::point(x, y, z));

        Eigen::Vector4f v(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], 1.f);
        v = m*v;

        proj_vts[i][2] = v[2]*this->width();
    }

    glPopMatrix();

    v_pick_idx_.clear();
    w_pick_sculpture_.clear();
    b_v_fixed_.clear();
    b_v_fixed_.resize(mesh_->vertices.size(), true);
    std::vector<int> tmp_mark(mesh_->vertices.size(), false);
    for (int i=0; i<fidx.size(); ++i)
    {
        const trimesh::TriMesh::Face &f = mesh_->faces[fidx[i]];
        for (int j=0; j<3; ++j)
        {
            int vid = f[j];
            if (!tmp_mark[vid])
            {
                tmp_mark[vid] = true;

                float min_d = 10000;
                for (int k=0; k<screen_pts.size(); ++k)
                {
                    float d = trimesh::len(trimesh::vec2(screen_pts[k].rx(), screen_pts[k].ry()) - trimesh::vec2(proj_vts[vid][0], proj_vts[vid][1]));
                    if (d < min_d)
                    {
                        min_d = d;
                    }
                }

                if (min_d < d_thres)
                {
                    v_pick_idx_.push_back(vid);
                    w_pick_sculpture_.push_back(std::exp(-2.f*min_d/d_thres));
                }
                else
                {
                    b_v_fixed_[vid] = false;
                }
            }
        }
    }

    updateGL();
}

//void MeshWidget::pickVisibleVertex(QPoint p, float d, std::vector<int> &vid)
//{
//    double mat_mv[16], mat_proj[16];
//    int view_port[4];
//    glMatrixMode(GL_MODELVIEW);
//    glPushMatrix();
//    glMultMatrixf(getBallMatrix ());
//    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
//    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
//    glGetIntegerv(GL_VIEWPORT, view_port);

//    int screen_w = int(view_port[2] - view_port[0]);
//    int screen_h = int(view_port[3] - view_port[1]);

//    GLfloat *d_buffer = new GLfloat[screen_w*screen_h];
//    glReadPixels(view_port[0], view_port[1], screen_w, screen_h, GL_DEPTH_COMPONENT, GL_FLOAT, d_buffer);

//    int nv = mesh_->vertices.size();
////    std::vector<trimesh::point> vts_proj(nv, trimesh::point(-1, -1, -1));
//    GLfloat local_epsilon = 1e-4;
//    vid.clear();
//    for (int i=0; i<nv; ++i)
//    {
//        GLdouble x=0, y=0, z=0;
//        gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
////        vts_proj[i] = trimesh::point(x, y, z);
//        if (trimesh::len(trimesh::vec2(p.rx(), this->height()-p.ry())-trimesh::vec2(x, y)) <= d)
//        {
//            if (x>=0 && x<screen_w && y>=0 && y<screen_h)
//            {
//                GLfloat z_buffer = d_buffer[int(x) + int(y)*screen_w];
//                if (z_buffer+local_epsilon >= GLfloat((z+1.0)/2.0))
//                {
//                    vid.push_back(i);
//                }
//            }
//        }
//    }

//    delete [] d_buffer;
//}

void MeshWidget::calcIntersectPtsWithLineDrawn()
{
    std::vector<trimesh::point> proj_vts;
    std::vector<int> fidx;
    pickMeshFacesWithLineDrawn(sculpture_screen_pos_, proj_vts, fidx);
//    base_processor_->subdivideMesh(mesh_, fidx);

    if (fidx.size() > 0)
    {
        // calculate points of intersection on mesh surface
        CLineSegmentIntersection line_inters;

        int npts = sculpture_screen_pos_.size();
        std::vector<trimesh::point> inters_pts;
        std::vector<std::vector<trimesh::point>> seg_pts(npts-1);
        std::vector<std::vector<float>> seg_pts_dis(npts-1);
        int f_num = fidx.size();
        for (int j=0; j<f_num; ++j)
        {
            const trimesh::TriMesh::Face &f = mesh_->faces[fidx[j]];
            for (int k=0; k<3; ++k)
            {
                const int &id1 = f[k];
                const int &id2 = f[(k+1)%3];
                const trimesh::point &v1 = mesh_->vertices[id1];
                const trimesh::point &v2 = mesh_->vertices[id2];
                const trimesh::point &proj_v1 = proj_vts[id1];
                const trimesh::point &proj_v2 = proj_vts[id2];
                trimesh::vec2 line_p1(proj_v1[0], proj_v1[1]);
                trimesh::vec2 line_p2(proj_v2[0], proj_v2[1]);

                for (int i=0; i<npts-1; ++i)
                {
                    line_p1 = trimesh::vec2(proj_v1[0], proj_v1[1]);
                    line_p2 = trimesh::vec2(proj_v2[0], proj_v2[1]);

                    trimesh::vec2 p1 = trimesh::vec2(sculpture_screen_pos_[i].rx(), sculpture_screen_pos_[i].ry());
                    trimesh::vec2 p2 = trimesh::vec2(sculpture_screen_pos_[i+1].rx(), sculpture_screen_pos_[i+1].ry());
                    trimesh::vec2 p = trimesh::vec2(0, 0);

                    int intersect_result = line_inters.intersectTwoLineSegment(p, line_p1, line_p2, p1, p2);
                    if (intersect_result == 6)
                    {
//                        inters_pts.push_back(v1);
//                        inters_pts.push_back(v2);
                        seg_pts[i].push_back(v1);
                        seg_pts[i].push_back(v2);
                    }
                    else if (intersect_result > 0)
                    {
                        float ratio = 0;
                        if ( proj_v1[0] != proj_v2[0] )
                        {
                            ratio = (p[0]-proj_v1[0])/(proj_v2[0]-proj_v1[0]);
                        }
                        else
                        {
                            ratio = (p[1]-proj_v1[1])/(proj_v2[1]-proj_v1[1]);
                        }

                        trimesh::point ins_p(ratio*(v2[0]-v1[0])+v1[0], ratio*(v2[1]-v1[1])+v1[1], ratio*(v2[2]-v1[2])+v1[2]);
                        seg_pts[i].push_back(ins_p);
                        seg_pts_dis[i].push_back(trimesh::len(p - trimesh::vec2(sculpture_screen_pos_[i].rx(), sculpture_screen_pos_[i].ry())));
                    }
                }
            }
        }

//        std::cout<<"tmp intersect point num: "<<seg_pts.size()<<std::endl;
        for (int i=0; i<seg_pts.size(); ++i)
        {
            int seg_pts_num = seg_pts[i].size();    /* std::cout<<i<<": "<<seg_pts_num<<std::endl;*/
            if (seg_pts_num == 1)
            {
                inters_pts.push_back(seg_pts[i][0]);
            }
            else if (seg_pts_num > 1)
            {
                while (seg_pts[i].size() > 0)
                {
                    if (seg_pts[i].size() == 1)
                    {
                        inters_pts.push_back(seg_pts[i][0]);
                        break;
                    }

                    int id = 0;
                    float min_d = seg_pts_dis[i][0];
                    for (int j=1; j<seg_pts_dis[i].size(); ++j)
                    {
                        if (seg_pts_dis[i][j] < min_d)
                        {
                            min_d = seg_pts_dis[i][j];
                            id = j;
                        }
                    }

                    inters_pts.push_back(seg_pts[i][id]);
                    seg_pts[i].erase(seg_pts[i].begin() + id);
                    seg_pts_dis[i].erase(seg_pts_dis[i].begin() + id);
                }
            }
        }

//        std::cout<<"intersect point num: "<<inters_pts.size()<<std::endl;
        if (inters_pts.size() > 0)
        {
            intersect_line_pts_.push_back(inters_pts);
        }

        updateGL();
    }
}

void MeshWidget::initLaplacianDeformer(const double lambda)
{
    if (lap_deformer_)
    {
        delete lap_deformer_;
        lap_deformer_ = nullptr;
    }

    lap_deformer_ = new LaplacianDeformation(mesh_, lambda);
    lap_deformer_->preCalcDeformationSolver(b_v_fixed_);
}

void MeshWidget::pickAreaFacesNearScreenPoint(QPoint &p, std::vector<trimesh::point> &proj_vts, std::vector<int> &fidx, const float &d_thres)
{
    p.ry() = this->height()-p.ry();

    // project mesh vertices
    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix ());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    proj_vts.clear();
    int nv = mesh_->vertices.size();
    for (int i=0; i<nv; ++i)
    {
        GLdouble x=0, y=0, z=0;
        gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
        proj_vts.push_back(trimesh::point(x, y, z));
    }
    glPopMatrix();

    fidx.clear();
    int nf = mesh_->faces.size();
    for (int i=0; i<nf; ++i)
    {
        const trimesh::TriMesh::Face &f = mesh_->faces[i];

        bool is_pick = false;
        for (int j=0; j<3; ++j)
        {
            if (trimesh::len(trimesh::vec2(p.rx(), p.ry()) - trimesh::vec2(proj_vts[f[j]][0], proj_vts[f[j]][1])) <= d_thres)
            {
                is_pick = true;
                fidx.push_back(i);

                break;
            }
        }
    }
}

void MeshWidget::pickAreaVertexNearScreenPoint(QPoint p, const float &d_thres, float fd_thres_time)
{
    resetInteraction();

    std::vector<int> fidx;
    std::vector<trimesh::point> proj_vts;
    float d_thres_face = d_thres*fd_thres_time;
    pickAreaFacesNearScreenPoint(p, proj_vts, fidx, d_thres_face);  // screen_pts.ry() has been set to be in OpenGL

    int nv_ori = mesh_->vertices.size();
    fidx = base_processor_->subdivideMeshPartMPS(mesh_, fidx, b_face_sub_);

    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    int nv = mesh_->vertices.size();
    for (int i=nv_ori; i<nv; ++i)
    {
        GLdouble x=0, y=0, z=0;
        gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], mat_mv, mat_proj, view_port, &x, &y, &z);
        proj_vts.push_back(trimesh::point(x, y, z));
    }

    glPopMatrix();

    v_pick_idx_.clear();
    w_pick_sculpture_.clear();
    b_v_fixed_.clear();
    b_v_fixed_.resize(mesh_->vertices.size(), true);
    std::vector<int> tmp_mark(mesh_->vertices.size(), false);
    for (int i=0; i<fidx.size(); ++i)
    {
        const trimesh::TriMesh::Face &f = mesh_->faces[fidx[i]];
        for (int j=0; j<3; ++j)
        {
            int vid = f[j];
            if (!tmp_mark[vid])
            {
                tmp_mark[vid] = true;

                float d = trimesh::len(trimesh::vec2(p.rx(), p.ry()) - trimesh::vec2(proj_vts[vid][0], proj_vts[vid][1]));
                if ( d < d_thres)
                {
                    v_pick_idx_.push_back(vid);
                    w_pick_sculpture_.push_back(std::exp(-d/d_thres));
                }
                else
                {
                    b_v_fixed_[vid] = false;
                }
            }
        }
    }
}

void MeshWidget::pickPredefinedContourVertexNearScreenPoint(QPoint p, const float &d_thres)
{
    p.ry() = this->height()-p.ry();

    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    std::vector<std::vector<trimesh::vec2>> proj_vec2(n_predefined_contour_);
    for (int i=0; i<n_predefined_contour_; ++i)
    {
        const std::vector<int> &tmp_idx = predefined_contour_idx_[i];
        for (int j=0; j<tmp_idx.size(); ++j)
        {
            GLdouble x=0, y=0, z=0;
            gluProject(mesh_->vertices[tmp_idx[j]][0], mesh_->vertices[tmp_idx[j]][1], mesh_->vertices[tmp_idx[j]][2], mat_mv, mat_proj, view_port, &x, &y, &z);
            proj_vec2[i].push_back(trimesh::vec2(x, y));
        }
    }
    glPopMatrix();

    float min_d = 100000;
    int min_n = -1;
    std::vector<std::vector<float>> d_vec(n_predefined_contour_);
    for (int i=0; i<n_predefined_contour_; ++i)
    {
        for (int j=0; j<proj_vec2[i].size(); ++j)
        {
            float d = trimesh::len(trimesh::vec2(p.rx(), p.ry()) - proj_vec2[i][j]);
            d_vec[i].push_back(d);

            if (d < min_d)
            {
                min_d = d;
                min_n = i;
            }
        }
    }

    assert(min_n >= 0);
    v_pick_idx_.clear();
    for (int i=0; i<predefined_contour_idx_[min_n].size(); ++i)
    {
        if (d_vec[min_n][i] <= d_thres)
        {
            v_pick_idx_.push_back(predefined_contour_idx_[min_n][i]);
        }
    }
}

//void MeshWidget::calcFeatureAreaNearLineDrawn(bool with_feature_check, float feat_val_thres)
//{
//    v_pick_idx_.clear();
//    b_v_fixed_.clear();

//    if (src_line_.size()>0 && mesh_)
//    {
//        // calculate bounding box of line drawn
//        int nlp = src_line_.size();         /*std::cout<<"src point num: "<<nlp<<std::endl;*/
//        std::vector<QPoint> src_line_trans = src_line_;
//        QPoint p_max(0, 0), p_min(10000, 10000);
//        for (int i=0; i<nlp; ++i)
//        {
//            src_line_trans[i].ry() = this->height()-src_line_[i].ry();

//            if (src_line_trans[i].rx() > p_max.rx())
//            {
//                p_max.rx() = src_line_trans[i].rx();
//            }
//            if (src_line_trans[i].rx() < p_min.rx())
//            {
//                p_min.rx() = src_line_trans[i].rx();
//            }

//            if (src_line_trans[i].ry() > p_max.ry())
//            {
//                p_max.ry() = src_line_trans[i].ry();
//            }
//            if (src_line_trans[i].ry() < p_min.ry())
//            {
//                p_min.ry() = src_line_trans[i].ry();
//            }
//        }
////        std::cout<<"p_max: "<<p_max.rx()<<" "<<p_max.ry()<<std::endl;
////        std::cout<<"p_min: "<<p_min.rx()<<" "<<p_min.ry()<<std::endl;

//        p_max.rx() += 60;
//        p_max.ry() += 60;
//        p_min.rx() -= 60;
//        p_min.ry() -= 60;

//        // project mesh vertices
//        double matMV[16], matProj[16];
//        int viewPort[4];
//        GLdouble camera_pos[3];
//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glMultMatrixf(getBallMatrix ());
//        glGetDoublev(GL_MODELVIEW_MATRIX, matMV);
//        glGetDoublev(GL_PROJECTION_MATRIX, matProj);
//        glGetIntegerv(GL_VIEWPORT, viewPort);
//        gluUnProject((viewPort[2]-viewPort[0])/2 , (viewPort[3]-viewPort[1])/2, 0.0, matMV, matProj, viewPort, &camera_pos[0],&camera_pos[1],&camera_pos[2]);
//        glPopMatrix();

//        int nv = mesh_->vertices.size();
//        std::vector<std::vector<GLdouble>> proj_vts(nv, std::vector<GLdouble>(3, 0));
//        std::vector<int> vid_area;
//        b_v_fixed_.resize(nv, true);
//        for (int i=0; i<nv; ++i)
//        {
//            gluProject(mesh_->vertices[i][0], mesh_->vertices[i][1], mesh_->vertices[i][2], matMV, matProj, viewPort, &proj_vts[i][0], &proj_vts[i][1], &proj_vts[i][2]);

//            if (proj_vts[i][0]<p_max.rx() && proj_vts[i][0]>p_min.rx() && proj_vts[i][1]<p_max.ry() && proj_vts[i][1]>p_min.ry())
//            {
//                vid_area.push_back(i);
//                b_v_fixed_[i] = false;
//            }
//        }
////        std::cout<<"vid in area num: "<<vid_area.size()<<std::endl;

//        int n_vid_area = vid_area.size();
//        std::vector<bool> vid_status(n_vid_area, false);
//        int step_src = int(src_line_.size()/10.f);
//        if (step_src == 0)
//        {
//            step_src = 1;
//        }
//        assert(mesh_->colors.size() == mesh_->vertices.size());

//        src_line_pts_.clear();
//        for (int i=0; i<src_line_trans.size(); i+=step_src)
//        {
//            float min_d = 1000000;
//            int min_id = -1;
//            for (int j=0; j<n_vid_area; ++j)
//            {
//                if (with_feature_check)
//                {
//                    if(mesh_->colors[vid_area[j]][0] >= feat_val_thres)
//                    {
//                        const int &vid = vid_area[j];
//                        float d = trimesh::len(trimesh::vec2(proj_vts[vid][0], proj_vts[vid][1]) - trimesh::vec2(src_line_trans[i].rx(), src_line_trans[i].ry()));
//                        if (d<min_d && vid_status[j]==false)
//                        {
//                            min_d = d;
//                            min_id = j;
//                        }
//                    }
//                }
//                else
//                {
//                    const int &vid = vid_area[j];
//                    float d = trimesh::len(trimesh::vec2(proj_vts[vid][0], proj_vts[vid][1]) - trimesh::vec2(src_line_trans[i].rx(), src_line_trans[i].ry()));
//                    if (d<min_d && vid_status[j]==false)
//                    {
//                        min_d = d;
//                        min_id = j;
//                    }
//                }
//            }

//            if (min_id >= 0)
//            {
//                v_pick_idx_.push_back(vid_area[min_id]);
//                src_line_pts_.push_back(src_line_trans[i]);
//                vid_status[min_id] = true;
//                b_v_fixed_[vid_area[min_id]] = true;
//            }
//        }
////        std::cout<<"pick vid num: "<<v_pick_idx_.size()<<std::endl;

//        initLaplacianDeformer();
//    }
//}



bool MeshWidget::pickTwoTerminalFeaturePtsOfLineDrawn(int &p_id_start, int &p_id_end, const QPoint &start_point, const QPoint &end_point, const std::vector<trimesh::vec2> &proj_vts, const std::vector<int> &vidx_sel, const float feat_val_thres, const int n_iter_neighbor_search)
{
    p_id_start = -1;
    p_id_end = -1;
    if (vidx_sel.size()>0 && mesh_)
    {
        float min_d_start = 1e5;
        float min_d_end = 1e5;
        float max_val_start = 0;
        float max_val_end = 0;
        float d_start = 0;
        float d_end = 0;
        trimesh::vec2 p_start(start_point.x(), start_point.y());
        trimesh::vec2 p_end(end_point.x(), end_point.y());
        for (int i=0; i<vidx_sel.size(); ++i)
        {
            const int &id = vidx_sel[i];
            if (mesh_->colors[id][0] > feat_val_thres)
            {
                d_start = trimesh::len(proj_vts[id]-p_start);
                d_end = trimesh::len(proj_vts[id]-p_end);
                if (d_start < min_d_start)
                {
                    p_id_start = id;
                    min_d_start = d_start;
                    max_val_start = mesh_->colors[id][0];
                }
                if (d_end < min_d_end)
                {
                    p_id_end = id;
                    min_d_end = d_end;
                    max_val_end = mesh_->colors[id][0];
                }
            }
        }

        mesh_->clear_neighbors();
        mesh_->need_neighbors();
        for (int k=0; k<n_iter_neighbor_search; ++k)
        {
            const std::vector<int> &nei_start = mesh_->neighbors[p_id_start];
            const std::vector<int> &nei_end = mesh_->neighbors[p_id_end];
            for (int i=0; i<nei_start.size(); ++i)
            {
                if (mesh_->colors[nei_start[i]][0] > max_val_start && b_v_fixed_[nei_start[i]]==false)
                {
                    p_id_start = nei_start[i];
                    max_val_start = mesh_->colors[nei_start[i]][0];
                }
            }
            for (int i=0; i<nei_end.size(); ++i)
            {
                if (mesh_->colors[nei_end[i]][0] > max_val_end && b_v_fixed_[nei_start[i]]==false)
                {
                    p_id_end = nei_end[i];
                    max_val_end = mesh_->colors[nei_end[i]][0];
                }
            }
        }

        if (p_id_start<0 || p_id_end<0)
        {
            return false;
        }
        else
        {
            return true;
        }
    }
}

void MeshWidget::calcFeatureCurveNearLineDrawn(std::vector<QPoint> screen_pts, const float feat_val_thres)
{
    if(screen_pts.size() > 1)
    {
        std::vector<int> vidx_sel;
        std::vector<trimesh::vec2> proj_vts;
        pickAreaVertexWithLineDrawn(screen_pts, proj_vts, vidx_sel, mouse_size_);

        int nv = mesh_->vertices.size();
        b_v_fixed_.clear();
        b_v_fixed_.resize(nv, true);
        for (int i=0; i<vidx_sel.size(); ++i)
        {
            b_v_fixed_[vidx_sel[i]] = false;
        }

        int p_id_start = -1;
        int p_id_end = -1;
        v_pick_idx_.clear();
        if (pickTwoTerminalFeaturePtsOfLineDrawn(p_id_start, p_id_end, screen_pts[0], screen_pts[screen_pts.size()-1], proj_vts, vidx_sel, feat_val_thres))
        {
            std::vector<int> route_p_id;
            dijk_short_path_->getDijkstraPath(route_p_id, p_id_start, p_id_end, mesh_, true);

            if (route_p_id.size() > 0)
            {
                for (int i=0; i<route_p_id.size(); ++i)
                {
                    const int &id = route_p_id[i];
                    b_v_fixed_[id] = true;
                    v_pick_idx_.push_back(id);
                }
                route_p_id.clear();

                initLaplacianDeformer(12.);
                b_curve_selected_ = true;
            }
        }
    }
}

std::vector<QPoint>  MeshWidget::calcScreenCurvePtsForEditing(std::vector<QPoint> screen_pts, bool b_trans)
{
    std::vector<QPoint> curve_pts;
    if (screen_pts.size()>0 && v_pick_idx_.size()>0)
    {
        int n_curve = screen_pts.size();
        int n_pick = v_pick_idx_.size();
        if (b_trans)
        {
            for (int i=0; i<n_curve; ++i)
            {
                screen_pts[i].ry() = this->height() - screen_pts[i].ry();
            }
        }

        std::vector<QPoint> screen_pts_inter;
        if (n_curve < n_pick)
        {
            if (n_curve > 1)
            {
                int n_inter_per = int((n_pick-n_curve)/(n_curve-1) + 0.5f);
                float ratio = 1.f/(n_inter_per+1);
                for (int i=0; i<n_curve-1; ++i)
                {
                   screen_pts_inter.push_back(screen_pts[i]);
                    for (int j=1; j<=n_inter_per; ++j)
                    {
                        screen_pts_inter.push_back(j*ratio*screen_pts[i] + (1.f-j*ratio)*screen_pts[i+1]);
                    }
                }
                screen_pts_inter.push_back(QPoint(screen_pts[n_curve-1]));
            }
            else
            {
                for (int i=0; i<n_pick; ++i)
                {
                    curve_pts.push_back(screen_pts[0]);
                }

                return curve_pts;
            }
        }
        else
        {
            screen_pts_inter = screen_pts;
        }

        n_curve = screen_pts_inter.size();
        if (n_pick > 1)
        {
            int n_gap = int((n_curve-1)/(n_pick-1));
            curve_pts.push_back(screen_pts_inter[0]);
            for (int i=1; i<n_pick-1; ++i)
            {
                curve_pts.push_back(screen_pts_inter[i*n_gap]);
            }
            curve_pts.push_back(screen_pts_inter[n_curve-1]);
        }
        else
        {
            curve_pts.push_back(screen_pts_inter[n_curve/2]);
        }
    }

    assert (curve_pts.size() == v_pick_idx_.size());

    return curve_pts;
}

void MeshWidget::doSurfaceCurveDragging()
{
    if (!meshes_bk_.size())
    {
        meshes_bk_.push_back(*mesh_);
    }

    const std::vector<trimesh::point> &vts_bk = meshes_bk_[meshes_bk_.size()-1].vertices;

    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    trimesh::vec2 trans_v(trimesh::vec2(mouse_pos_.rx(), this->height()-mouse_pos_.ry()) - trimesh::vec2(p_anchor_.rx(), this->height()-p_anchor_.ry()));
    std::vector<trimesh::point> handle_pos;
    for (int i=0; i<v_pick_idx_.size(); ++i)
    {
        const int &id = v_pick_idx_[i];
        GLdouble winx=0, winy=0, winz=0;
        gluProject(vts_bk[id][0], vts_bk[id][1], vts_bk[id][2], mat_mv, mat_proj, view_port, &winx, &winy, &winz);
        GLdouble x=0, y=0, z=0;
        gluUnProject(winx + trans_v[0], winy + trans_v[1], winz, mat_mv, mat_proj, view_port, &x, &y, &z);

        handle_pos.push_back(trimesh::point(x, y, z));
    }
    glPopMatrix();

    // v_pick_idx and handle_pos are continuous on the contour, smooth it
    for (int j=0; j<3; ++j)
    {
        handle_pos[0] = (handle_pos[0] + vts_bk[v_pick_idx_[0]])/2.f;
        handle_pos[handle_pos.size()-1] = (handle_pos[handle_pos.size()-1] + vts_bk[v_pick_idx_[handle_pos.size()-1]])/2.f;

        for (int i=1; i<handle_pos.size()-1; ++i)
        {
            handle_pos[i] = (handle_pos[i-1] + handle_pos[i+1])/2.f;
        }
    }

    lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

//    mesh_->write("mesh_deform.ply");

    updateModelView();
}

void MeshWidget::doFeatureCurveDragging()
{
    if (!meshes_bk_.size())
    {
        meshes_bk_.push_back(*mesh_);
    }

    const std::vector<trimesh::point> &vts_bk = meshes_bk_[meshes_bk_.size()-1].vertices;

    double mat_mv[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);

    Eigen::MatrixXf m(4, 4);
    int k = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            m(i, j) = mat_mv[k];
            k++;
        }
    }
    Eigen::MatrixXf m_inv = m.inverse();
    glPopMatrix();

    trimesh::vec2 trans_v(trimesh::vec2(mouse_pos_.rx(), this->height()-mouse_pos_.ry()) - trimesh::vec2(p_anchor_.rx(), this->height()-p_anchor_.ry()));
    float x_delta = trans_v[0]/this->height();
    float y_delta = trans_v[1]/this->height();
    std::vector<trimesh::point> handle_pos;
    for (int i=0; i<v_pick_idx_.size(); ++i)
    {
        const int &id = v_pick_idx_[i];
        Eigen::Vector4f v(vts_bk[id][0], vts_bk[id][1], vts_bk[id][2], 1.f);
        v = m_inv*v;

        v[0] += x_delta;
        v[1] += y_delta;

        v = m*v;
        handle_pos.push_back(trimesh::point(v[0], v[1], v[2]));
    }

    lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

    base_processor_->doLaplacianSmooth(mesh_, v_pick_idx_);
//    mesh_->write("mesh_deform.ply");

    updateModelView();
}

void MeshWidget::doContourCurveDragging()
{
    if (!meshes_bk_.size())
    {
        meshes_bk_.push_back(*mesh_);
    }

    const std::vector<trimesh::point> &vts_bk = meshes_bk_[meshes_bk_.size()-1].vertices;

    double mat_mv[16], mat_proj[16];
    int view_port[4];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMultMatrixf(getBallMatrix());
    glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
    glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
    glGetIntegerv(GL_VIEWPORT, view_port);

    trimesh::vec2 trans_v(trimesh::vec2(mouse_pos_.rx(), this->height()-mouse_pos_.ry()) - trimesh::vec2(p_anchor_.rx(), this->height()-p_anchor_.ry()));
    std::vector<trimesh::point> handle_pos;
    for (int i=0; i<v_pick_idx_.size(); ++i)
    {
        const int &id = v_pick_idx_[i];
        GLdouble winx=0, winy=0, winz=0;
        gluProject(vts_bk[id][0], vts_bk[id][1], vts_bk[id][2], mat_mv, mat_proj, view_port, &winx, &winy, &winz);
        GLdouble x=0, y=0, z=0;
        gluUnProject(winx + trans_v[0], winy + trans_v[1], winz, mat_mv, mat_proj, view_port, &x, &y, &z);

        handle_pos.push_back(trimesh::point(x, y, z));
    }
    glPopMatrix();

    // v_pick_idx and handle_pos are continuous on the contour, smooth it
    for (int j=0; j<3; ++j)
    {
        handle_pos[0] = (handle_pos[0] + vts_bk[v_pick_idx_[0]])/2.f;
        handle_pos[handle_pos.size()-1] = (handle_pos[handle_pos.size()-1] + vts_bk[v_pick_idx_[handle_pos.size()-1]])/2.f;

        for (int i=1; i<handle_pos.size()-1; ++i)
        {
            handle_pos[i] = (handle_pos[i-1] + handle_pos[i+1])/2.f;
        }
    }

    if (b_contour_.size() != mesh_->vertices.size())    // the mesh has been subdivided
    {
        for (int i=b_contour_.size(); i<mesh_->vertices.size(); ++i)
        {
            b_contour_.push_back(false);
        }

        delete contour_deformer_;
        contour_deformer_ = nullptr;

        contour_deformer_ = new LaplacianDeformation(mesh_, 0.2);
        contour_deformer_->preCalcDeformationSolver(b_contour_);
    }

    contour_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

//    mesh_->write("mesh_deform.ply");

    updateModelView();
}

void MeshWidget::doContourCurveEditing()
{
    const int n_curve_pts = tar_curve_pos_.size();
    if (n_curve_pts > 0)
    {
        std::vector<QPoint> tar_curve_pos_trans = tar_curve_pos_;
        for (int i=0; i<n_curve_pts; ++i)
        {
            tar_curve_pos_trans[i].ry() = this->height()-tar_curve_pos_[i].ry();
        }

        double mat_mv[16], mat_proj[16];
        int view_port[4];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glMultMatrixf(getBallMatrix());
        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
        glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
        glGetIntegerv(GL_VIEWPORT, view_port);

        std::vector<std::vector<trimesh::vec2>> proj_vec2(n_predefined_contour_);
        for (int i=0; i<n_predefined_contour_; ++i)
        {
            const std::vector<int> &tmp_idx = predefined_contour_idx_[i];
            for (int j=0; j<tmp_idx.size(); ++j)
            {
                GLdouble x=0, y=0, z=0;
                gluProject(mesh_->vertices[tmp_idx[j]][0], mesh_->vertices[tmp_idx[j]][1], mesh_->vertices[tmp_idx[j]][2], mat_mv, mat_proj, view_port, &x, &y, &z);
                proj_vec2[i].push_back(trimesh::vec2(x, y));
            }
        }

        float min_d = 100000;
        int min_n = -1;
        int start_id = -1;
        for (int i=0; i<n_predefined_contour_; ++i)
        {
            for (int j=0; j<proj_vec2[i].size(); ++j)
            {
                float d = trimesh::len(trimesh::vec2(tar_curve_pos_trans[0].rx(), tar_curve_pos_trans[0].ry()) - proj_vec2[i][j]);
                if (d < min_d)
                {
                    min_d = d;
                    min_n = i;
                    start_id = j;
                }
            }
        }

        int end_id = -1;
        min_d = 100000;
        for (int i=0; i<proj_vec2[min_n].size(); ++i)
        {
            float d = trimesh::len(trimesh::vec2(tar_curve_pos_trans[n_curve_pts-1].rx(), tar_curve_pos_trans[n_curve_pts-1].ry()) - proj_vec2[min_n][i]);
            if (d < min_d)
            {
                min_d = d;
                end_id = i;
            }
        }

        int mid_id = -1;
        int half_size = int(n_curve_pts/2.f);
        min_d = 100000;
        for (int i=0; i<proj_vec2[min_n].size(); ++i)
        {
            float d = trimesh::len(trimesh::vec2(tar_curve_pos_trans[half_size].rx(), tar_curve_pos_trans[half_size].ry()) - proj_vec2[min_n][i]);
            if (d < min_d)
            {
                min_d = d;
                mid_id = i;
            }
        }

//        meshes_bk_.push_back(*mesh_);

        v_pick_idx_.clear();
        if (start_id <= mid_id)
        {
            for (int i=start_id; i<=end_id; ++i)
            {
                v_pick_idx_.push_back(predefined_contour_idx_[min_n][i]);
            }
        }
        else
        {
            for (int i=start_id; i>=end_id; --i)
            {
                v_pick_idx_.push_back(predefined_contour_idx_[min_n][i]);
            }
        }

        tar_curve_pos_trans = calcScreenCurvePtsForEditing(tar_curve_pos_trans, false);

        // smooth the curve_pos
        const int sm_iter = 2;
        for (int i=0; i<sm_iter; ++i)
        {
            for (int j=1; j<tar_curve_pos_trans.size()-1; ++j)
            {
                tar_curve_pos_trans[j] = (tar_curve_pos_trans[j-1] + tar_curve_pos_trans[j+1])/2.f;
            }
        }

        std::vector<trimesh::point> handle_pos;
        for (int i=0; i<v_pick_idx_.size(); ++i)
        {
            const int &id = v_pick_idx_[i];
            GLdouble winx=0, winy=0, winz=0;
            gluProject(mesh_->vertices[id][0], mesh_->vertices[id][1], mesh_->vertices[id][2], mat_mv, mat_proj, view_port, &winx, &winy, &winz);
            GLdouble x=0, y=0, z=0;
            gluUnProject(tar_curve_pos_trans[i].rx(), tar_curve_pos_trans[i].ry(), winz, mat_mv, mat_proj, view_port, &x, &y, &z);

            handle_pos.push_back(trimesh::point(x, y, z));
        }
        glPopMatrix();

        if (b_contour_.size() != mesh_->vertices.size())    // the mesh has been subdivided
        {
            for (int i=b_contour_.size(); i<mesh_->vertices.size(); ++i)
            {
                b_contour_.push_back(false);
            }

            delete contour_deformer_;
            contour_deformer_ = nullptr;

            contour_deformer_ = new LaplacianDeformation(mesh_, 0.2);
            contour_deformer_->preCalcDeformationSolver(b_contour_);
        }

        contour_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

    //    mesh_->write("mesh_deform.ply");
        meshes_bk_.push_back(*mesh_);

        v_pick_idx_.clear();
        updateModelView();
    }
}

//void MeshWidget::doFeatureCurveEditing()
//{
//    int nv_pick = v_pick_idx_.size();
//    if (nv_pick > 0)
//    {
//        meshes_bk_.push_back(*mesh_);

//        double mat_mv[16];
//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glMultMatrixf(getBallMatrix());
//        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
//        Eigen::MatrixXf m(4, 4);
//        int k = 0;
//        for (int i = 0; i < 4; i++)
//        {
//            for (int j = 0; j < 4; j++)
//            {
//                m(i, j) = mat_mv[k];
//                k++;
//            }
//        }
//        Eigen::MatrixXf m_inv = m.inverse();
//        glPopMatrix();

//        std::vector<QPoint> tar_curve_pos_sel = calcScreenCurvePtsForEditing(tar_curve_pos_);
//        std::vector<trimesh::point> handle_pos;
//        for (int i=0; i<v_pick_idx_.size(); ++i)
//        {
//            trimesh::vec2 dis_vec = (trimesh::vec2(tar_curve_pos_sel[i].x(), tar_curve_pos_sel[i].y()) - trimesh::vec2(src_curve_pos_sel_[i].x(), src_curve_pos_sel_[i].y()))/this->height()*4.5f;

//            const int &id = v_pick_idx_[i];
//            Eigen::Vector4f v(mesh_->vertices[id][0], mesh_->vertices[id][1], mesh_->vertices[id][2], 1.f);
//            v = m_inv*v;

//            v[0] += dis_vec[0];
//            v[1] += dis_vec[1];

//            v = m*v;
//            handle_pos.push_back(trimesh::point(v[0], v[1], v[2]));
//        }

//        lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

//        base_processor_->doLaplacianSmooth(mesh_, v_pick_idx_);
//    //    mesh_->write("mesh_deform.ply");

////        src_curve_pos_sel_ = tar_curve_pos_sel;
//        src_curve_pos_.clear();
//        tar_curve_pos_.clear();

//        updateModelView();
//    }
//}

//void MeshWidget::doFeatureCurveEditing()
//{
//    int nv_pick = v_pick_idx_.size();
//    if (nv_pick > 0)
//    {
//        meshes_bk_.push_back(*mesh_);

//        double mat_mv[16], mat_proj[16];
//        int view_port[4];
//        glMatrixMode(GL_MODELVIEW);
//        glPushMatrix();
//        glMultMatrixf(getBallMatrix());
//        glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);
//        glGetDoublev(GL_PROJECTION_MATRIX, mat_proj);
//        glGetIntegerv(GL_VIEWPORT, view_port);

//        std::vector<QPoint> tar_curve_pos_sel = calcScreenCurvePtsForEditing(tar_curve_pos_);
//        std::vector<trimesh::point> handle_pos;
//        for (int i=0; i<v_pick_idx_.size(); ++i)
//        {
//            const int &id = v_pick_idx_[i];
//            GLdouble winx=0, winy=0, winz=0;
//            gluProject(mesh_->vertices[id][0], mesh_->vertices[id][1], mesh_->vertices[id][2], mat_mv, mat_proj, view_port, &winx, &winy, &winz);
//            GLdouble x=0, y=0, z=0;
//            gluUnProject(tar_curve_pos_sel[i].x(), tar_curve_pos_sel[i].y(), winz, mat_mv, mat_proj, view_port, &x, &y, &z);

//            handle_pos.push_back(trimesh::point(x, y, z));
//        }

//        glPopMatrix();

//        lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);
//    //    mesh_->write("mesh_deform.ply");

//        src_curve_pos_.clear();
//        tar_curve_pos_.clear();

//        updateModelView();
//    }
//}

void MeshWidget::doSculpture()
{
    int nv_pick = v_pick_idx_.size();

    if (nv_pick > 0)
    {
        if (sculpture_mode_ == Smooth)
        {
            meshes_bk_.push_back(*mesh_);

//            int sm_iters = 3;
//            mesh_->need_neighbors();
//            for (int k=0; k<sm_iters; ++k)
//            {
//                //  std::vector<trimesh::point> pts_smooth;
//                for (int i=0; i<nv_pick; ++i)
//                {
//                    int id = v_pick_idx_[i];
//                    const std::vector<int> &v_nei_id = mesh_->neighbors[id];
//                    trimesh::point p_ave(0, 0, 0);
//                    for (int j=0; j<v_nei_id.size(); ++j)
//                    {
//                        p_ave += mesh_->vertices[v_nei_id[j]];
//                    }
//                    //  p_ave /= v_nei_id.size();

//                    //  pts_smooth.push_back(0.9f*(p_ave-mesh_->vertices[id]) + mesh_->vertices[id]);
//                    mesh_->vertices[id] = p_ave/v_nei_id.size();
//                }

//                //  for (int i=0; i<nv_pick; ++i)
//                //  {
//                    //  int id = v_pick_idx_[i];
//                    //  mesh_->vertices[id] = pts_smooth[i];
//                //  }
//            }

            base_processor_->doLaplacianSmooth(mesh_, v_pick_idx_, 0.85, 5);
        }
        else if (sculpture_mode_ == Downward)
        {
            meshes_bk_.push_back(*mesh_);

            mesh_->clear_normals();
            mesh_->need_normals();

            std::vector<trimesh::point> handle_pos;
            for (int i=0; i<nv_pick; ++i)
            {
//                handle_pos.push_back(mesh_->vertices[v_pick_idx_[i]] - sculpture_scale_*mesh_->normals[v_pick_idx_[i]]*w_pick_sculpture_[i]);
                handle_pos.push_back(mesh_->vertices[v_pick_idx_[i]] - sculpture_scale_*mesh_->normals[v_pick_idx_[i]]);
            }
            lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

            base_processor_->doLaplacianSmooth(mesh_, v_pick_idx_, 0.85, 5);

            mesh_->clear_normals();
            mesh_->need_normals();
        }
        else if (sculpture_mode_ == Flatten)
        {

        }
        else if (sculpture_mode_ == Grab)
        {
            if (!meshes_bk_.size())
            {
                meshes_bk_.push_back(*mesh_);
            }

            const std::vector<trimesh::point> &vts_bk = meshes_bk_[meshes_bk_.size()-1].vertices;

            double mat_mv[16];
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glMultMatrixf(getBallMatrix());
            glGetDoublev(GL_MODELVIEW_MATRIX, mat_mv);

            Eigen::MatrixXf m(4, 4);
            int k = 0;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    m(i, j) = mat_mv[k];
                    k++;
                }
            }
            Eigen::MatrixXf m_inv = m.inverse();
            glPopMatrix();

            trimesh::vec2 trans_v(trimesh::vec2(mouse_pos_.rx(), this->height()-mouse_pos_.ry()) - trimesh::vec2(p_anchor_.rx(), this->height()-p_anchor_.ry()));
            float x_delta = trans_v[0]/this->height();
            float y_delta = trans_v[1]/this->height();
            std::vector<trimesh::point> handle_pos;
            for (int i=0; i<nv_pick; ++i)
            {
                const int &id = v_pick_idx_[i];
                Eigen::Vector4f v(vts_bk[id][0], vts_bk[id][1], vts_bk[id][2], 1.f);
                v = m_inv*v;

//                v[0] += x_delta*w_pick_sculpture_[i];
//                v[1] += y_delta*w_pick_sculpture_[i];
                v[0] += x_delta;
                v[1] += y_delta;

                v = m*v;
                handle_pos.push_back(trimesh::point(v[0], v[1], v[2]));
            }

            lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

            updateModelView();
        }
        else if (sculpture_mode_ == Upward)
        {
            meshes_bk_.push_back(*mesh_);

            mesh_->clear_normals();
            mesh_->need_normals();

            std::vector<trimesh::point> handle_pos;
            for (int i=0; i<nv_pick; ++i)
            {
//                handle_pos.push_back(mesh_->vertices[v_pick_idx_[i]] + sculpture_scale_*mesh_->normals[v_pick_idx_[i]]*w_pick_sculpture_[i]);
                handle_pos.push_back(mesh_->vertices[v_pick_idx_[i]] + sculpture_scale_*mesh_->normals[v_pick_idx_[i]]);
            }
            lap_deformer_->doLaplacianDeformation(v_pick_idx_, handle_pos);

            base_processor_->doLaplacianSmooth(mesh_, v_pick_idx_, 0.85, 5);

            mesh_->clear_normals();
            mesh_->need_normals();
        }

        updateModelView();
    }
}

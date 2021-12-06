#include "suggestivecontour.h"
#include <QPainter>
#include "XForm.h"

trimesh::TriMesh mesh_;
float sug_thresh_;
std::vector<trimesh::point> sketch_pts_;

SuggestiveContour::SuggestiveContour()
{

}

SuggestiveContour::~SuggestiveContour()
{
    mesh_.clear();
}

Camera SuggestiveContour::calcCamPose(float alpha, float fi, float d)
{
    Camera cam;
    point pos;
    pos[1] = d*sin(fi);
    float dxz = d*cos(fi);
    pos[0] = dxz*cos(alpha);
    pos[2] = dxz*sin(alpha);

    cam.pos = pos;
    point center(0.f, 0.f, 0.f);
    point dir = center - pos;
    dir = dir * (1.0f/sqrtf(len2(dir)));
    cam.dir = dir;
    // u and v
    point ydir(0.f, 1.f, 0.f);
    point v = ydir CROSS dir;
    v = v * (1.0f/sqrtf(len2(v)));

    point u = dir CROSS v;
    u = u * (1.0f/sqrtf(len2(u)));
    cam.u = u; cam.v = v;
    return cam;
}

void SuggestiveContour::rotateAroundVec(point p, point n, point &q, float theta)
{
    float R[4][4];
    float u, v, w, a, b, c;
    u = n[0]; v = n[1]; w = n[2];
    a = p[0]; b = p[1]; c = p[2];

    float st, ct;
    st = sin(theta); ct = cos(theta);

    R[0][0] = u*u+(v*v+w*w)*ct; R[0][1] = u*v*(1-ct)-w*st;  R[0][2] = u*w*(1-ct)+v*st;   R[0][3] = (a*(v*v+w*w)-u*(b*v+c*w))*(1-ct)+(b*w-c*v)*st;
    R[1][0]	= u*v*(1-ct)+w*st;  R[1][1] = v*v+(u*u+w*w)*ct; R[1][2] = v*w*(1-ct)-u*st;   R[1][3] = (b*(u*u+w*w)-v*(a*u+c*w))*(1-ct)+(c*u-a*w)*st;
    R[2][0]	= u*w*(1-ct)-v*st;  R[2][1] = v*w*(1-ct)+u*st;  R[2][2] = w*w+(u*u+v*v)*ct;  R[2][3] = (c*(u*u+v*v)-w*(a*u+b*v))*(1-ct)+(a*v-b*u)*st;
    R[3][0]	= 0;                R[3][1] = 0;                R[3][2] = 0;                 R[3][3] =  1;

    float x, y, z;
    x = R[0][0]*q[0] + R[0][1]*q[1] + R[0][2]*q[2] + R[0][3];
    y = R[1][0]*q[0] + R[1][1]*q[1] + R[1][2]*q[2] + R[1][3];
    z = R[2][0]*q[0] + R[2][1]*q[1] + R[2][2]*q[2] + R[2][3];

    q[0] = x; q[1] = y; q[2] = z;
}

void SuggestiveContour::rotateMesh(float ang)
{
    if (std::abs(ang) > 0.001)
    {
        int nv = mesh_.vertices.size();
        for(int i=0; i<nv; ++i)
        {
            trimesh::point p = mesh_.vertices[i];
            mesh_.vertices[i][0] = std::cos(ang)*p[0] + std::sin(ang)*p[2];
            mesh_.vertices[i][2] = -std::sin(ang)*p[0] + std::cos(ang)*p[2];
        }
    }
}

trimesh::vec SuggestiveContour::calcGradKr(int i, trimesh::vec viewpos)
{
    trimesh::vec viewdir = viewpos - mesh_.vertices[i];
    float rlen_viewdir = 1.0f / trimesh::len(viewdir);
    viewdir *= rlen_viewdir;

    float ndotv = viewdir DOT mesh_.normals[i];
    float sintheta = std::sqrt(1.0f - trimesh::sqr(ndotv));
    float csctheta = 1.0f / sintheta;
    float u = (viewdir DOT mesh_.pdir1[i]) * csctheta;
    float v = (viewdir DOT mesh_.pdir2[i]) * csctheta;
    float kr = mesh_.curv1[i] * u*u + mesh_.curv2[i] * v*v;
    float tr = u*v * (mesh_.curv2[i] - mesh_.curv1[i]);
    float kt = mesh_.curv1[i] * (1.0f - u*u) +
        mesh_.curv2[i] * (1.0f - v*v);
    trimesh::vec w     = u * mesh_.pdir1[i] + v * mesh_.pdir2[i];
    trimesh::vec wperp = u * mesh_.pdir2[i] - v * mesh_.pdir1[i];
    const trimesh::Vec<4> &C = mesh_.dcurv[i];

    trimesh::vec g = mesh_.pdir1[i] * (u*u*C[0] + 2.0f*u*v*C[1] + v*v*C[2]) +
        mesh_.pdir2[i] * (u*u*C[1] + 2.0f*u*v*C[2] + v*v*C[3]) -
        2.0f * csctheta * tr * (rlen_viewdir * wperp +
        ndotv * (tr * w + kt * wperp));
    g *= (1.0f - trimesh::sqr(ndotv));
    g -= 2.0f * kr * sintheta * ndotv * (kr * w + tr * wperp);
    return g;
}

float SuggestiveContour::findZeroLinear(float val0, float val1)
{
    return val0 / (val0 - val1);
}

float SuggestiveContour::findZeroHermite(int v0, int v1, float val0, float val1, trimesh::vec &grad0, trimesh::vec &grad1)
{
    if (unlikely(val0 == val1))
        return 0.5f;

    // Find derivatives along edge (of interpolation parameter in [0,1]
    // which means that e01 doesn't get normalized)
    trimesh::vec e01 = mesh_.vertices[v1] - mesh_.vertices[v0];
    float d0 = e01 DOT grad0, d1 = e01 DOT grad1;

    // This next line would reduce val to linear interpolation
    //d0 = d1 = (val1 - val0);

    // Use hermite interpolation:
    //   val(s) = h1(s)*val0 + h2(s)*val1 + h3(s)*d0 + h4(s)*d1
    // where
    //  h1(s) = 2*s^3 - 3*s^2 + 1
    //  h2(s) = 3*s^2 - 2*s^3
    //  h3(s) = s^3 - 2*s^2 + s
    //  h4(s) = s^3 - s^2
    //
    //  val(s)  = [2(val0-val1) +d0+d1]*s^3 +
    //            [3(val1-val0)-2d0-d1]*s^2 + d0*s + val0
    // where
    //
    //  val(0) = val0; val(1) = val1; val'(0) = d0; val'(1) = d1
    //

    // Coeffs of cubic a*s^3 + b*s^2 + c*s + d
    float a = 2 * (val0 - val1) + d0 + d1;
    float b = 3 * (val1 - val0) - 2 * d0 - d1;
    float c = d0, d = val0;

    // -- Find a root by bisection
    // (as Newton can wander out of desired interval)

    // Start with entire [0,1] interval
    float sl = 0.0f, sr = 1.0f, valsl = val0, valsr = val1;

    // Check if we're in a (somewhat uncommon) 3-root situation, and pick
    // the middle root if it happens (given we aren't drawing curvy lines,
    // seems the best approach..)
    //
    // Find extrema of derivative (a -> 3a; b -> 2b, c -> c),
    // and check if they're both in [0,1] and have different signs
    float disc = 4 * b - 12 * a * c;
    if (disc > 0 && a != 0) {
        disc = sqrt(disc);
        float r1 = (-2 * b + disc) / (6 * a);
        float r2 = (-2 * b - disc) / (6 * a);
        if (r1 >= 0 && r1 <= 1 && r2 >= 0 && r2 <= 1) {
            float vr1 = (((a * r1 + b) * r1 + c) * r1) + d;
            float vr2 = (((a * r2 + b) * r2 + c) * r2) + d;
            // When extrema have different signs inside an
            // interval with endpoints with different signs,
            // the middle root is in between the two extrema
            if (vr1 < 0.0f && vr2 >= 0.0f ||
                vr1 > 0.0f && vr2 <= 0.0f)
            {
                // 3 roots
                if (r1 < r2)
                {
                    sl = r1;
                    valsl = vr1;
                    sr = r2;
                    valsr = vr2;
                }
                else
                {
                    sl = r2;
                    valsl = vr2;
                    sr = r1;
                    valsr = vr1;
                }
            }
        }
    }

    // Bisection method (constant number of interactions)
    for (int iter = 0; iter < 10; iter++) {
        float sbi = (sl + sr) / 2.0f;
        float valsbi = (((a * sbi + b) * sbi) + c) * sbi + d;

        // Keep the half which has different signs
        if (valsl < 0.0f && valsbi >= 0.0f ||
            valsl > 0.0f && valsbi <= 0.0f)
        {
            sr = sbi;
            valsr = valsbi;
        } else
        {
            sl = sbi;
            valsl = valsbi;
        }
    }

    return 0.5f * (sl + sr);
}

void SuggestiveContour::computePerview(trimesh::vec viewpos, vector<float> &ndotv, vector<float> &kr, vector<float> &sctest_num, vector<float> &sctest_den)
{
    int nv = mesh_.vertices.size();
    float feature_size = mesh_.feature_size();

    float scthresh = sug_thresh_ / trimesh::sqr(feature_size);
    bool need_DwKr = true;

    ndotv.resize(nv);
    kr.resize(nv);
    sctest_num.resize(nv);
    sctest_den.resize(nv);

    // Compute quantities at each vertex
#pragma omp parallel for
    for (int i = 0; i < nv; i++) {
        // Compute n DOT v
        trimesh::vec viewdir = viewpos - mesh_.vertices[i];
        float rlv = 1.0f / len(viewdir);
        viewdir *= rlv;
        ndotv[i] = viewdir DOT mesh_.normals[i];

        float u = viewdir DOT mesh_.pdir1[i], u2 = u*u;
        float v = viewdir DOT mesh_.pdir2[i], v2 = v*v;

        // Note:  this is actually Kr * sin^2 theta
        kr[i] = mesh_.curv1[i] * u2 + mesh_.curv2[i] * v2;

        if (!need_DwKr) continue;

        // Use DwKr * sin(theta) / cos(theta) for cutoff test
        sctest_num[i] = u2 * (u*mesh_.dcurv[i][0] +
            3.0f*v*mesh_.dcurv[i][1]) +
            v2 * (3.0f*u*mesh_.dcurv[i][2] +
            v*mesh_.dcurv[i][3]);
        float csc2theta = 1.0f / (u2 + v2);
        sctest_num[i] *= csc2theta;
        float tr = (mesh_.curv2[i] - mesh_.curv1[i]) *
            u * v * csc2theta;
        sctest_num[i] -= 2.0f * ndotv[i] * trimesh::sqr(tr);

        sctest_den[i] = ndotv[i];
        sctest_num[i] -= scthresh * sctest_den[i];
    }
}

void SuggestiveContour::drawMeshIsoline2(trimesh::vec viewpos, int v0, int v1, int v2, vector<float> &val, vector<float> &test_num, vector<float> &test_den, bool do_hermite, bool do_test)
{
    trimesh::vec tmp0 = calcGradKr(v0, viewpos);
    trimesh::vec tmp1 = calcGradKr(v1, viewpos);
    trimesh::vec tmp2 = calcGradKr(v2, viewpos);

    // How far along each edge?
    float w10 = do_hermite ?
        findZeroHermite(v0, v1, val[v0], val[v1],
        tmp0, tmp1) :
        findZeroLinear(val[v0], val[v1]);
    float w01 = 1.0f - w10;
    float w20 = do_hermite ?
        findZeroHermite(v0, v2, val[v0], val[v2],
        tmp0, tmp2) :
        findZeroLinear(val[v0], val[v2]);
    float w02 = 1.0f - w20;

    // Points along edges
    point p1 = w01 * mesh_.vertices[v0] + w10 * mesh_.vertices[v1];
    point p2 = w02 * mesh_.vertices[v0] + w20 * mesh_.vertices[v2];

    float test_num1 = 1.0f, test_num2 = 1.0f;
    float test_den1 = 1.0f, test_den2 = 1.0f;
    float z1 = 0.0f, z2 = 0.0f;
    bool valid1 = true;

    if (do_test) {
        // Interpolate to find value of test at p1, p2
        test_num1 = w01 * test_num[v0] + w10 * test_num[v1];
        test_num2 = w02 * test_num[v0] + w20 * test_num[v2];
        if (!test_den.empty()) {
            test_den1 = w01 * test_den[v0] + w10 * test_den[v1];
            test_den2 = w02 * test_den[v0] + w20 * test_den[v2];
        }
        // First point is valid iff num1/den1 is positive,
        // i.e. the num and den have the same sign
        valid1 = ((test_num1 >= 0.0f) == (test_den1 >= 0.0f));
        // There are two possible zero crossings of the test,
        // corresponding to zeros of the num and den
        if ((test_num1 >= 0.0f) != (test_num2 >= 0.0f))
            z1 = test_num1 / (test_num1 - test_num2);
        if ((test_den1 >= 0.0f) != (test_den2 >= 0.0f))
            z2 = test_den1 / (test_den1 - test_den2);
        // Sort and order the zero crossings
        if (z1 == 0.0f)
            z1 = z2, z2 = 0.0f;
        else if (z2 < z1)
            std::swap(z1, z2);
    }

    // If the beginning of the segment was not valid, and
    // no zero crossings, then whole segment invalid
    if (!valid1 && !z1 && !z2)
    {
        return;
    }

    // Draw the valid piece(s)
    int npts = 0;
    if (valid1) {
        sketch_pts_.push_back(p1);
        npts++;
    }
    if (z1)
    {
        sketch_pts_.push_back((1.0f - z1) * p1 + z1 * p2);
        npts++;
    }
    if (z2)
    {
        sketch_pts_.push_back((1.0f - z2) * p1 + z2 * p2);
        npts++;
    }
    if (npts != 2)
    {
        sketch_pts_.push_back(p2);
    }
}

void SuggestiveContour::drawMeshIsoline(trimesh::vec viewpos, int v0, int v1, int v2, vector<float> &val, vector<float> &test_num, vector<float> &test_den, vector<float> &ndotv, bool do_bfcull, bool do_hermite, bool do_test)
{
    if (likely(do_bfcull && ndotv[v0] <= 0.0f &&
        ndotv[v1] <= 0.0f && ndotv[v2] <= 0.0f))
        return;

    // Quick reject if derivatives are negative
    if (do_test) {
        if (test_den.empty()) {
            if (test_num[v0] <= 0.0f &&
                test_num[v1] <= 0.0f &&
                test_num[v2] <= 0.0f)
                return;
        } else {
            if (test_num[v0] <= 0.0f && test_den[v0] >= 0.0f &&
                test_num[v1] <= 0.0f && test_den[v1] >= 0.0f &&
                test_num[v2] <= 0.0f && test_den[v2] >= 0.0f)
                return;
            if (test_num[v0] >= 0.0f && test_den[v0] <= 0.0f &&
                test_num[v1] >= 0.0f && test_den[v1] <= 0.0f &&
                test_num[v2] >= 0.0f && test_den[v2] <= 0.0f)
                return;
        }
    }


    // Figure out which val has different sign, and draw
    if (val[v0] < 0.0f && val[v1] >= 0.0f && val[v2] >= 0.0f ||
        val[v0] > 0.0f && val[v1] <= 0.0f && val[v2] <= 0.0f)
        drawMeshIsoline2(viewpos, v0, v1, v2, val, test_num, test_den,
        do_hermite, do_test);
    else if (val[v1] < 0.0f && val[v2] >= 0.0f && val[v0] >= 0.0f ||
        val[v1] > 0.0f && val[v2] <= 0.0f && val[v0] <= 0.0f)
        drawMeshIsoline2(viewpos, v1, v2, v0, val, test_num, test_den,
        do_hermite, do_test);
    else if (val[v2] < 0.0f && val[v0] >= 0.0f && val[v1] >= 0.0f ||
        val[v2] > 0.0f && val[v0] <= 0.0f && val[v1] <= 0.0f)
        drawMeshIsoline2(viewpos, v2, v0, v1, val, test_num, test_den,
        do_hermite, do_test);
}

void SuggestiveContour::initRenderSuggesContour()
{
    mesh_.clear_tstrips();
    mesh_.clear_normals();
    mesh_.clear_curvatures();
    mesh_.clear_dcurv();
    mesh_.clear_bsphere();

    mesh_.need_tstrips();
    mesh_.need_normals();
    mesh_.need_curvatures();
    mesh_.need_dcurv();
    mesh_.need_bsphere();
    sug_thresh_ = 0.15;
}

void SuggestiveContour::drawIsolines(trimesh::vec viewpos, vector<float> &val, vector<float> &test_num, vector<float> &test_den, vector<float> &ndotv, bool do_bfcull, bool do_hermite, bool do_test, bool b_render_half)
{
    const int *t = &mesh_.tstrips[0];
    const int *stripend = t;
    const int *end = t + mesh_.tstrips.size();

    // Walk through triangle strips
    while (1) {
        if (unlikely(t >= stripend)) {
            if (unlikely(t >= end))
                return;
            // New strip: each strip is stored as
            // length followed by indices
            stripend = t + 1 + *t;
            // Skip over length plus first two indices of
            // first face
            t += 3;
        }

        // if only rendering the left half model, then x <= 0;
        if (b_render_half)
        {
            if (mesh_.vertices[*(t-2)][2]<=0.f || mesh_.vertices[*(t-1)][2]<=0.f || mesh_.vertices[*t][2]<=0.f)
            {
                t++;
                continue;
            }
        }

        // Draw a line if, among the values in this triangle,
        // at least one is positive and one is negative
        const float &v0 = val[*t], &v1 = val[*(t-1)], &v2 = val[*(t-2)];
        if (unlikely((v0 > 0.0f || v1 > 0.0f || v2 > 0.0f) &&
            (v0 < 0.0f || v1 < 0.0f || v2 < 0.0f)))
            drawMeshIsoline(viewpos, *(t-2), *(t-1), *t, val, test_num, test_den, ndotv, do_bfcull, do_hermite, do_test);

        t++;
    }
}

void SuggestiveContour::calcSuggestiveContours(trimesh::vec viewpos, vector<float> &ndotv, vector<float> &kr, vector<float> &sctest_num, vector<float> &sctest_den, bool b_render_half)
{
//    drawIsolines(viewpos, kr, sctest_num, sctest_den, ndotv, true, false, true, b_render_half);
    vector<float> tmp;
    drawIsolines(viewpos, ndotv, kr, tmp, ndotv, true, false, true, b_render_half);
}

void SuggestiveContour::renderMeshSuggestiveContour(QImage &img, trimesh::TriMesh *mesh, const float &ang)
{
    const float alpha = 0.5f*PI;
    const float fi = 0.f;
    const float theta = 0.f;
    const float d = 2.75f;
    mesh_ = *mesh;
    rotateMesh(ang);
    initRenderSuggesContour();

    vector<float> ndotv, kr;
    vector<float> sctest_num, sctest_den;

    // calculate camera pose
    Camera cam = calcCamPose(alpha, fi, d);
    // calculate local camera axis u, v
    point u, v; u = cam.u; v = cam.v;

    rotateAroundVec(cam.pos, cam.dir, v, theta);
    v = v * (1.0f/sqrtf(len2(v)));
    u = cam.dir CROSS v;
    u = u * (1.0f/sqrtf(len2(u)));

//    point c = cam.dir * 1.2f + cam.pos;
    point c = cam.dir + cam.pos;
    trimesh::vec viewpos = cam.pos;

    point p = cam.pos; point pc = c-p;
    float lpc = pc DOT pc;

    sketch_pts_.clear();
    computePerview(viewpos, ndotv, kr, sctest_num, sctest_den);
    calcSuggestiveContours(viewpos, ndotv, kr, sctest_num, sctest_den);

    int img_w = img.width();
    int img_h = img.height();
    img.fill(255);
    QPainter painter(&img);
    QPen pen_black(Qt::black, 2);
    painter.setPen(pen_black);

    for (size_t i=0; i<sketch_pts_.size()-1; i+=2)
    {
        point q = sketch_pts_[i];
        point pq = q-p;
        float lmda = lpc/(pq DOT pc);
        point s = p + lmda*pq;
        float x, y;
        point cs = s - c;
        x = cs DOT v; y = cs DOT u;
        x = 0.5*img_w - img_w * x;
        y = 0.5*img_h - img_h * y;

        x = (int) x; y = (int) y;

        if(x<0||x>=img_w) continue;
        if(y<0||y>=img_h) continue;

        QPoint p1(x, y);
        q = sketch_pts_[i+1];

        pq = q-p;
        lmda = lpc/(pq DOT pc);
        s = p + lmda*pq; cs = s - c;
        x = cs DOT v; y = cs DOT u;
        x = 0.5*img_w - img_w * x;
        y = 0.5*img_h - img_h * y;

        x = (int) x; y = (int) y;

        if(x<0||x>=img_w) continue;
        if(y<0||y>=img_h) continue;

        QPoint p2(x, y);

        if(p1.rx()==p2.rx() && p1.ry()==p2.ry())
        {
            continue;
        }

        painter.drawLine(p1, p2);
    }
    painter.end();
    img.save("other_view.png");
}

void SuggestiveContour::getMeshSuggestiveContour(std::vector<std::vector<QPoint> > &pts, trimesh::TriMesh *mesh, const float &ang, const int img_w, const int img_h, bool b_render_half)
{
    const float alpha = 0.5f*PI;
    const float fi = 0.f;
    const float theta = 0.f;
    const float d = 2.5f;
    mesh_ = *mesh;
    rotateMesh(ang);
    initRenderSuggesContour();

    vector<float> ndotv, kr;
    vector<float> sctest_num, sctest_den;

    // calculate camera pose
    Camera cam = calcCamPose(alpha, fi, d);
    // calculate local camera axis u, v
    point u, v; u = cam.u; v = cam.v;

    rotateAroundVec(cam.pos, cam.dir, v, theta);
    v = v * (1.0f/sqrtf(len2(v)));
    u = cam.dir CROSS v;
    u = u * (1.0f/sqrtf(len2(u)));

//    point c = cam.dir * 1.2f + cam.pos;
    point c = cam.dir + cam.pos;
    trimesh::vec viewpos = cam.pos;
//    std::cout<<viewpos<<std::endl;

//    trimesh::xform xf = trimesh::xform::trans(0, 0, -5.f * mesh_.bsphere.r) * trimesh::xform::trans(-mesh_.bsphere.center);
//    trimesh::point viewpos_tmp = trimesh::inv(xf) * point(0,0,0);
//    std::cout<<"viewpos_tmp: "<<viewpos_tmp<<std::endl;

    point p = cam.pos; point pc = c-p;
    float lpc = pc DOT pc;

    pts.clear();
    sketch_pts_.clear();
    computePerview(viewpos, ndotv, kr, sctest_num, sctest_den);
    calcSuggestiveContours(viewpos, ndotv, kr, sctest_num, sctest_den, b_render_half);

    for (size_t i=0; i<sketch_pts_.size()-1; i+=2)
    {
        point q = sketch_pts_[i];
        point pq = q-p;
        float lmda = lpc/(pq DOT pc);
        point s = p + lmda*pq;
        float x, y;
        point cs = s - c;
        x = cs DOT v; y = cs DOT u;
        x = 0.5*img_w - img_w * x;
        y = 0.5*img_h - img_h * y;

        x = (int) x; y = (int) y;

        if(x<0||x>=img_w) continue;
        if(y<0||y>=img_h) continue;

        QPoint p1(x, y);
        q = sketch_pts_[i+1];

        pq = q-p;
        lmda = lpc/(pq DOT pc);
        s = p + lmda*pq; cs = s - c;
        x = cs DOT v; y = cs DOT u;
        x = 0.5*img_w - img_w * x;
        y = 0.5*img_h - img_h * y;

        x = (int) x; y = (int) y;

        if(x<0||x>=img_w) continue;
        if(y<0||y>=img_h) continue;

        QPoint p2(x, y);

        if(p1.rx()==p2.rx() && p1.ry()==p2.ry())
        {
            continue;
        }

        std::vector<QPoint> tmp_pts;
        tmp_pts.push_back(p1);
        tmp_pts.push_back(p2);
        pts.push_back(tmp_pts);
    }
}

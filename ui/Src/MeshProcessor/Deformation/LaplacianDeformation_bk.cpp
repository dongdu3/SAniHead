#include "LaplacianDeformation.h"
#include <iostream>

#include <igl/grad.h>
#include <igl/cotmatrix.h>
#include <igl/boundary_facets.h>
#include <igl/writeOBJ.h>
#include <igl/gaussian_curvature.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/writeDMAT.h>
#include <igl/doublearea.h>

LaplacianDeformation::LaplacianDeformation(trimesh::TriMesh *mesh)
{
    mesh_ = mesh;
    initMeshInformation();
}

LaplacianDeformation::~LaplacianDeformation()
{
//    if (mesh_)
//    {
//        delete mesh_;
//        mesh_ = nullptr;
//    }
}

void LaplacianDeformation::initMeshInformation()
{
    if (!mesh_)
    {
        std::cout<<"The mesh for doing laplacian deformation is null!"<<std::endl;
        return;
    }

    const int nv = mesh_->vertices.size();
    const int nf = mesh_->faces.size();

    assert(nv > 0);
    assert(nf > 0);

    verts_.resize(nv, 3);
    for (int i=0; i<nv; i++)
    {
        verts_(i, 0) = mesh_->vertices[i][0];
        verts_(i, 1) = mesh_->vertices[i][1];
        verts_(i, 2) = mesh_->vertices[i][2];
    }

//    b_v_fixed_.resize(nv, false);
    tar_verts_ = mesh_->vertices;

    faces_.resize(nf, 3);
    for (int i=0; i<nf; i++)
    {
        for (int j=0; j<3; j++)
        {
            faces_(i, j) = mesh_->faces[i][j];
        }
    }

    // Alternative construction of same Laplacian
    Eigen::SparseMatrix<double> G;
    // Gradient/Divergence
    igl::grad(verts_, faces_, G);
    // Diagonal per-triangle "mass matrix"
    Eigen::VectorXd dbl_area;
    igl::doublearea(verts_, faces_, dbl_area);
    // Place areas along diagonal #dim times
    const auto &T = 1.*(dbl_area.replicate(3, 1)*0.5).asDiagonal();
    lap_mat_ = G.transpose() * T * G;
    lap_val_ = lap_mat_ * verts_;
    //cout<<"|K-L|: "<<(K-L).norm()<<endl;
}

void LaplacianDeformation::preCalcDeformationSolver(const std::vector<bool> &b_v_fixed)
{
    assert(mesh_->vertices.size() == b_v_fixed.size());

    const int nv = b_v_fixed.size();
    for (int i=0; i<nv; ++i)
    {
        if(b_v_fixed[i])
        {
            fixed_id_.push_back(i);
        }
    }

    const int n_fix = fixed_id_.size();
    Eigen::MatrixXd temp_mat = (Eigen::MatrixXd)lap_mat_;
    Eigen::MatrixXd coeff_mat(lap_mat_.rows()+n_fix, lap_mat_.cols());
    mat_b_.resize(lap_mat_.rows()+n_fix, 3);
    for (int i=0; i<temp_mat.rows(); i++)
    {
        coeff_mat.row(i) = temp_mat.row(i);
        for (int j=0; j<3; j++)
        {
            mat_b_(i, j) = lap_val_(i, j);
        }
    }
    for (int i=temp_mat.rows(); i<coeff_mat.rows(); i++)
    {
        int id = fixed_id_[i-temp_mat.rows()];
        coeff_mat.row(i).setZero();
        coeff_mat(i, id) = 1;
    }

    L_mat_.resize(coeff_mat.rows(), coeff_mat.cols());
    L_mat_ = coeff_mat.sparseView();

    Eigen::SparseMatrix<double> mat_a = L_mat_.transpose() * L_mat_;
    solver_.compute(mat_a);
    assert(solver_.info() == Eigen::Success);
}

void LaplacianDeformation::doLaplacianDeformation()
{
//    std::cout<<"lap_mat_rows: "<<lap_mat_.rows()<<std::endl;
//    std::cout<<"L_mat_rows: "<<L_mat_.rows()<<std::endl;
    int id_start = lap_mat_.rows();
    for (int i=id_start; i<L_mat_.rows(); ++i)
    {
        int id = fixed_id_[i-id_start];
        for (int j=0; j<3; ++j)
        {
            mat_b_(i, j) = mesh_->vertices[id][j];      // here mesh_.vertices are different from the verts_, since they will be changed by handers' displacement;
        }
    }

    verts_ = solver_.solve(L_mat_.transpose()*mat_b_).eval();

    for (int i=0; i<mesh_->vertices.size(); i++)
    {
        for (int j=0; j<3;j++)
        {
            mesh_->vertices[i][j] = verts_(i, j);
        }
    }
}


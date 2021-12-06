#ifndef LAPLACIANDEFORMATION_H
#define LAPLACIANDEFORMATION_H

#include <map>
#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Sparse>
#include "TriMesh.h"

class LaplacianDeformation
{
public:
    trimesh::TriMesh *mesh_;        // the same to the mesh in the main widget; (public member) just for convenience, but not safe

private:
    Eigen::MatrixXd verts_;
    Eigen::MatrixXi faces_;
    Eigen::SparseMatrix<double> lap_mat_;
    Eigen::MatrixXd lap_val_;

//    std::vector<bool> b_v_fixed_;
    std::vector<trimesh::point> tar_verts_;

    // information for matrix solver; calculate after the selection of vertex handles, before deformation;
    std::vector<int> fixed_id_;
    Eigen::SparseMatrix<double> L_mat_;
    Eigen::MatrixXd mat_b_;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_;

public:
    LaplacianDeformation(trimesh::TriMesh *mesh);
    ~LaplacianDeformation();

    void preCalcDeformationSolver(const std::vector<bool> &b_v_fixed);
    void doLaplacianDeformation();

private:
    void initMeshInformation();

};

#endif // LAPLACIANDEFORMATION_H

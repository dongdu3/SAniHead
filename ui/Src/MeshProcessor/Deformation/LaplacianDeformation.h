#ifndef LAPLACIANDEFORMATION_H
#define LAPLACIANDEFORMATION_H

//#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include "TriMesh.h"

class LaplacianDeformation
{
public:
    trimesh::TriMesh *mesh_;        // the same to the mesh in the main widget; (public member) just for convenience, but not safe

private:
    std::vector<trimesh::vec3> lap_val_;
    std::vector<std::vector<double>> lap_w_;
//    std::vector<bool> b_v_fixed_;

    double  lambda_;    // energy weight for laplacian value
    Eigen::SparseMatrix<double> lap_mat_;
    Eigen::VectorXd bx_, by_, bz_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_lu_;
//    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver_ldlt_;

public:
    LaplacianDeformation(trimesh::TriMesh *mesh, double lambda=1.);
    ~LaplacianDeformation();

    void preCalcDeformationSolver(const std::vector<bool> &b_v_fixed);
    void doLaplacianDeformation(const std::vector<int> &handle_id, const std::vector<trimesh::point> &handle_pos);
    void doLaplacianDeformation(const std::vector<int> &handle_id);

private:
    void initMeshInformation();
    bool calcCotangentWeights(const trimesh::point &p, const std::vector<trimesh::point> &p_nei, std::vector<double> &w);
};

#endif // LAPLACIANDEFORMATION_H

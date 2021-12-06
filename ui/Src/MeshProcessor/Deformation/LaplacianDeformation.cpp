#include "LaplacianDeformation.h"
#include <iostream>

LaplacianDeformation::LaplacianDeformation(trimesh::TriMesh *mesh, double lambda)
{
    lambda_ = lambda;
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

    // calculate the initial lap_val, lap_w
    lap_val_.clear();
    lap_w_.clear();
    int nv = mesh_->vertices.size();
    mesh_->clear_neighbors();
    mesh_->need_neighbors();
    for (int i=0; i<nv; ++i)
    {
        int nn = mesh_->neighbors[i].size();
        std::vector<double> w;
        std::vector<trimesh::point> p_nei;
        for (int j=0; j<nn; ++j)
        {
            p_nei.push_back(mesh_->vertices[mesh_->neighbors[i][j]]);
        }
        calcCotangentWeights(mesh_->vertices[i], p_nei, w);
        lap_w_.push_back(w);

        trimesh::vec3 v_lap = mesh_->vertices[i];
        for (int j=0; j<nn; ++j)
        {
            v_lap -= w[j]*mesh_->vertices[mesh_->neighbors[i][j]];
        }
        lap_val_.push_back(v_lap);
    }
}

bool LaplacianDeformation::calcCotangentWeights(const trimesh::point &p, const std::vector<trimesh::point> &p_nei, std::vector<double> &w)
{
    int n = p_nei.size();
    if (n < 3)
    {
        return false;
    }

    w.resize(n, 0);

    // for the first neighbor vertex
    float sum_w = 0;

    trimesh::vec3 a1 = p_nei[0] - p_nei[n-1];
    trimesh::vec3 b1 = p - p_nei[n-1];
    trimesh::vec3 a2 = p - p_nei[1];
    trimesh::vec3 b2 = p_nei[0] - p_nei[1];
    w[0] = a1.dot(b1)/trimesh::len(a1.cross(b1)) + a2.dot(b2)/trimesh::len(a2.cross(b2));

    sum_w += w[0];

    // for the last neighbor vertex
    a1 = p_nei[n-1] - p_nei[n-2];
    b1 = p - p_nei[n-2];
    a2 = p - p_nei[0];
    b2 = p_nei[n-1] - p_nei[0];
    w[n-1] = a1.dot(b1)/trimesh::len(a1.cross(b1)) + a2.dot(b2)/trimesh::len(a2.cross(b2));

    sum_w += w[n-1];

    // for the middle neighbor vertices
    for (int i=1; i<n-1; ++i)
    {
        a1 = p_nei[i] - p_nei[i-1];
        b1 = p - p_nei[i-1];
        a2 = p - p_nei[i+1];
        b2 = p_nei[i] - p_nei[i+1];
        w[i] = a1.dot(b1)/trimesh::len(a1.cross(b1)) + a2.dot(b2)/trimesh::len(a2.cross(b2));

        sum_w += w[i];
    }

    if (sum_w>0 && sum_w<1e8)
    {
        for (int i=0; i<n; ++i)
        {
            w[i] /= sum_w;
        }

        return true;
    }
    else
    {
        w.clear();
        w.resize(n, 1./n);

        return false;
    }
}

void LaplacianDeformation::preCalcDeformationSolver(const std::vector<bool> &b_v_fixed)
{
    assert(mesh_->vertices.size() == b_v_fixed.size());

    int nv = mesh_->vertices.size();
    lap_mat_.resize(nv, nv);
    lap_mat_.setZero();
    bx_.resize(nv);
    by_.resize(nv);
    bz_.resize(nv);
    bx_.setZero();
    by_.setZero();
    bz_.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    for (int i=0; i<nv; ++i)
    {
        bx_(i) += lap_val_[i][0] * lambda_;
        by_(i) += lap_val_[i][1] * lambda_;
        bz_(i) += lap_val_[i][2] * lambda_;

        const std::vector<int> &v_neighbor_id = mesh_->neighbors[i];
        for (int j=0; j<v_neighbor_id.size(); ++j)
        {
            triplets.push_back(Eigen::Triplet<double>(i, v_neighbor_id[j], -lap_w_[i][j] * lambda_));
        }

        if (b_v_fixed[i])
        {
            bx_(i) += mesh_->vertices[i][0];
            by_(i) += mesh_->vertices[i][1];
            bz_(i) += mesh_->vertices[i][2];

            triplets.push_back(Eigen::Triplet<double>(i, i, 1. + lambda_));
        }
        else
        {
            triplets.push_back(Eigen::Triplet<double>(i, i, lambda_));
        }
    }

    lap_mat_.setFromTriplets(triplets.begin(), triplets.end());
    solver_lu_.compute(lap_mat_);
    assert(solver_lu_.info() == Eigen::Success);
}

void LaplacianDeformation::doLaplacianDeformation(const std::vector<int> &handle_id, const std::vector<trimesh::point> &handle_pos)
{
    if(handle_id.size() != handle_pos.size())
    {
        std::cout<<"handle_id.size: "<<handle_id.size()<<", "<< "handle_pos.size: "<<handle_pos.size()<<std::endl;
    }

    for (int i=0; i<handle_id.size(); ++i)
    {
        const int &id = handle_id[i];

        bx_(id) = handle_pos[i][0] + lap_val_[id][0] * lambda_;
        by_(id) = handle_pos[i][1] + lap_val_[id][1] * lambda_;
        bz_(id) = handle_pos[i][2] + lap_val_[id][2] * lambda_;
    }

    Eigen::VectorXd x = solver_lu_.solve(bx_);
    Eigen::VectorXd y = solver_lu_.solve(by_);
    Eigen::VectorXd z = solver_lu_.solve(bz_);
    for (int i=0; i<mesh_->vertices.size(); ++i)
    {
        mesh_->vertices[i][0] = x(i);
        mesh_->vertices[i][1] = y(i);
        mesh_->vertices[i][2] = z(i);
    }
}

void LaplacianDeformation::doLaplacianDeformation(const std::vector<int> &handle_id)
{
    if(handle_id.size() > 0)
    {
        for (int i=0; i<handle_id.size(); ++i)
        {
            const int &id = handle_id[i];

            bx_(id) = mesh_->vertices[id][0] + lap_val_[id][0] * lambda_;
            by_(id) = mesh_->vertices[id][1] + lap_val_[id][1] * lambda_;
            bz_(id) = mesh_->vertices[id][2] + lap_val_[id][2] * lambda_;
        }

        Eigen::VectorXd x = solver_lu_.solve(bx_);
        Eigen::VectorXd y = solver_lu_.solve(by_);
        Eigen::VectorXd z = solver_lu_.solve(bz_);
        for (int i=0; i<mesh_->vertices.size(); ++i)
        {
            mesh_->vertices[i][0] = x(i);
            mesh_->vertices[i][1] = y(i);
            mesh_->vertices[i][2] = z(i);
        }
    }
}




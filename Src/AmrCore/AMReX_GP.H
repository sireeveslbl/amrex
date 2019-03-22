#ifndef AMREX_GP_H
#define AMREX_GP_H
#include <AMReX_REAL.H> 
#include <vector> 
#include <cmath> 

// Contains the GP class for Steven Reeves' research into Gaussian Process based Models for AMR prolongation 

class GP
{
    public: 
    GP(){}; 
    ~GP(){};

// Member data 
    //
    // Eigen Values of Covariance Matrix 
    //
    amrex::Real lam[5]; 
    //
    //  Eigen Vectors of Covariance Matrix
    //
    amrex::Real V[5][5];
    //
    //  Weights to be applied for interpolation
    //
    amrex::Real ks[16][5][5]; 
    //
    //  Gammas needed for smoothness indication
    //
    amrex::Real gam[16][5];
    amrex::Real l; 

// Linear Algebra Functions
    template<int n>
    inline
    static amrex::Real inner_prod(const amrex::Real x[n], const amrex::Real y[n])
    {
        amrex::Real result = 0.e0; 
        for(int i = 0; i < n; ++i) result += x[i]*y[i];
        return result;  
    }
    template<int n> void CholeskyDecomp(amrex::Real (&K)[n][n]);
 
    template<int n> void matmul(const amrex::Real (&A)[n][n], const amrex::Real (&X)[n][n], 
                                   amrex::Real (&B)[n][n]); 

    template<int n> void matmul(const std::vector<std::vector<amrex::Real>> &A, 
                                const std::vector<std::vector<amrex::Real>> &X, 
                                std::vector<std::vector<amrex::Real>> &B); 

    template<int n> 
    void
    cholesky(amrex::Real (&b)[n], amrex::Real const K[n][n]); 

    template<int n> 
    void 
    Ux_solve(const amrex::Real R[n][n], amrex::Real (&x)[n], const amrex::Real b[n]); 

    void
    qr_decomp(std::vector<std::vector<amrex::Real>> &R, std::vector<std::vector<amrex::Real>> &Q, const int n);

    void 
    QR(amrex::Real (&A)[13][5], amrex::Real (&Q)[13][13],amrex::Real (&R)[5][5]);

    void
    q_appl(std::vector<std::vector<amrex::Real>> &A, const std::vector<std::vector<amrex::Real>> V, const int rows, 
           int upper=0, int mode=0);
    void
    Decomp(amrex::Real (&K)[5][5], amrex::Real (&Kt)[13][13]); 

//GP functions! 
    template<class T> 
    amrex::Real sqrexp(const T x[2],const T y[2], const amrex::Real *dx);

    // Set up for the multi-sampled Weighted GP interpolation 
    void InitGP(const int rx, const int ry, const amrex::Real *dx); 
    // Build K makes the Coviarance Kernel Matrices for each Samples 
    // And for Total Stencil
    void GetK(amrex::Real (&K)[5][5], amrex::Real (&Ktot)[13][13], const amrex::Real *dx); 
    //
    // Get Weights builds k*Kinv for each stencil
    //
    void GetKs(const amrex::Real K[5][5], const amrex::Real *dx);
    //
    //  Get Weights for the LSQ RHS
    //
    void GetKtotks(const amrex::Real K1[13][13], amrex::Real (&kt)[16][13], const amrex::Real *dx); 
    //
    // Get Gamma by solving a LSQ problem only need this once. 
    //
    void GetGamma(const amrex::Real ks[5][5], const amrex::Real kt[13], amrex::Real (&ga)[5]); 
    //
    //  Get EigenVecs and EigenValues for smoothness indicators. 
    //  Will use the Shifted QR algorithm with deflation 
    //
    template<int n> 
    void GetEigen(const amrex::Real K[n][n]);
};



#endif 

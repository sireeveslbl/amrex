#ifndef AMREX_GP_H
#define AMREX_GP_H
#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H> 
#include <vector>
#include <array> 
#include <cmath> 

// Contains the GP class for Steven Reeves' research into Gaussian Process based Models for AMR prolongation 

class GP
{
    public: 
    GP(const amrex::IntVect Ratio, const amrex::Real *del); 
    ~GP(){};      
    
    // Member data
    amrex::IntVect r; 
    amrex::Real dx[2];  
    //
    // Eigen Values of Covariance Matrix 
    //
    amrex::Real lam[5] = {}; 
    //
    //  Eigen Vectors of Covariance Matrix
    //
    amrex::Real V[5][5] = {};
    //
    //  Weights to be applied for interpolation
    //
//    amrex::Real ks[16][5][5] = {}; //Needs rx*ryin the first dimension
    std::vector<std::array<std::array<amrex::Real, 5>, 5>> ks; 
    AMREX_GPU_DEVICE_MANAGED amrex::Real (*ktd)[5][5]; 
    //
    //  Gammas needed for smoothness indication
    //
//    amrex::Real gam[16][5] = {};  //Needs rx*ry in the first dimension
    std::vector<std::array<amrex::Real, 5>> gam;
    AMREX_GPU_DEVICE_MANAGED amrex::Real (*gamd)[5]; 
    amrex::Real l;
    amrex::Real sig;  

// Linear Algebra Functions
    template<int n>
    inline
    static amrex::Real inner_prod(const amrex::Real x[n], const amrex::Real y[n])
    {
        amrex::Real result = 0.e0; 
        for(int i = 0; i < n; ++i) result += x[i]*y[i];
        return result;  
    }

    template<int n>
    inline
    static amrex::Real inner_prod(std::array<amrex::Real, n> &x, const amrex::Real y[n])
    {
        amrex::Real result = 0.e0; 
        for(int i = 0; i < n; ++i) result += x[i]*y[i];
        return result;  
    }


    template<int n> void CholeskyDecomp(amrex::Real (&K)[n][n]);
 
    template<int n> 
    void
    cholesky(amrex::Real (&b)[n], amrex::Real const K[n][n]); 
    
    template<int n> 
    void
    cholesky(std::array<amrex::Real, n> &b, amrex::Real const K[n][n]);

    void
    Decomp(amrex::Real (&K)[5][5], amrex::Real (&Kt)[13][13]); 

//GP functions! 
    template<class T> 
    amrex::Real sqrexp(const T x[2],const T y[2]);

    amrex::Real sqrexp(const std::array<amrex::Real, 2> x, const amrex::Real y[2]); 
    amrex::Real sqrexp2(const amrex::Real x[2], const amrex::Real y[2]); 

    // Set up for the multi-sampled Weighted GP interpolation 
    // Build K makes the Coviarance Kernel Matrices for each Samples 
    // And for Total Stencil
    void GetK(amrex::Real (&K)[5][5], amrex::Real (&Ktot)[13][13]); 
    //
    // Get Weights builds k*Kinv for each stencil
    //
    void GetKs(const amrex::Real K[5][5]);
    //
    //  Get Weights for the LSQ RHS
    //
    void GetKtotks(const amrex::Real K1[13][13], std::vector<std::array<amrex::Real, 13>> &kt); 
    //
    // Get Gamma by solving a LSQ problem only need this once. 
    //
    void GetGamma(std::array<std::array<amrex::Real, 5>, 5> const& k,
                  std::array<amrex::Real,13> const &kt, std::array<amrex::Real,5> &ga); 
    //
    //  Get EigenVecs and EigenValues for smoothness indicators. 
    //  Will use the Shifted QR algorithm with deflation 
    //
    void GetEigen();
};



#endif 

#ifndef AMREX_GP_3D_H
#define AMREX_GP_3D_H
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
    amrex::Real dx[3]; 
    //
    // Eigen Values of Covariance Matrix 
    //
    amrex::Real lam[7] = {}; 
    //
    //  Eigen Vectors of Covariance Matrix
    //
    amrex::Real V[7][7] = {};
    //
    //  Weights to be applied for interpolation
    //
    std::vector<std::array<std::array<amrex::Real, 7>, 7>> ks; 
    AMREX_GPU_DEVICE_MANAGED amrex::Real (*ktd)[7][7]; 
    //
    //  Gammas needed for smoothness indication
    //
    std::vector<std::array<amrex::Real, 7>> gam;
    AMREX_GPU_DEVICE_MANAGED amrex::Real (*gamd)[7]; 
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
    Decomp(amrex::Real (&K)[7][7], amrex::Real (&Kt)[25][25]); 

//GP functions! 
    template<class T> 
    amrex::Real sqrexp(const T x[3],const T y[3]);

    amrex::Real sqrexp(const std::array<amrex::Real, 3> x, const amrex::Real y[3]); 
    
    amrex::Real sqrexp2(const amrex::Real x[3], const amrex::Real y[3]); 
 


    // Set up for the multi-sampled Weighted GP interpolation 
    // Build K makes the Coviarance Kernel Matrices for each Samples 
    // And for Total Stencil
    void GetK(amrex::Real (&K)[7][7], amrex::Real (&Ktot)[25][25]); 
    //
    // Get Weights builds k*Kinv for each stencil
    //
    void GetKs(const amrex::Real K[7][7]);
    //
    //  Get Weights for the LSQ RHS
    //
    void GetKtotks(const amrex::Real K1[25][25], std::vector<std::array<amrex::Real, 25>> &kt); 
    //
    // Get Gamma by solving a LSQ problem only need this once. 
    //
    void GetGamma(std::array<std::array<amrex::Real, 7>, 7> const& k,
                  std::array<amrex::Real,25> const &kt, std::array<amrex::Real,7> &ga); 
    //
    //  Get EigenVecs and EigenValues for smoothness indicators. 
    //  Will use the Shifted QR algorithm with deflation 
    //
    void GetEigen();
};



#endif 

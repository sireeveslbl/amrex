#include <AMReX_GP_2D.H>
#include <iostream> 
#include <fstream> 
#include <iomanip>
#include <lapacke.h> 

template<class T>
amrex::Real
GP::sqrexp(const T x[2],const T y[2])
{
    amrex::Real result = std::exp(-0.5*((x[0] - y[0])*(x[0] - y[0])*dx[0]*dx[0] + 
                                        (x[1] - y[1])*(x[1] - y[1])*dx[1]*dx[1])/(l*l));
    return result;    
}
 
amrex::Real
GP::sqrexp(const std::array<amrex::Real, 2> x, const amrex::Real y[2])
{
    amrex::Real result = std::exp(-0.5*((x[0] - y[0])*(x[0] - y[0])*dx[0]*dx[0] + 
                                        (x[1] - y[1])*(x[1] - y[1])*dx[1]*dx[1])/(l*l));
    return result;    
} 

amrex::Real
GP::sqrexp2(const amrex::Real x[2], const amrex::Real y[2])
{
    amrex::Real result = std::exp(-0.5*((x[0] - y[0])*(x[0] - y[0])*dx[0]*dx[0] + 
                                        (x[1] - y[1])*(x[1] - y[1])*dx[1]*dx[1])/(sig*sig));
    return result;    
} 

    //Perfroms Cholesky Decomposition on covariance matrix K

GP::GP (const amrex::IntVect Ratio, const amrex::Real *del)
{
    D_DECL(dx[0] = del[0], dx[1] = del[1], dx[2] = del[2]); 
    r = Ratio;
    l = 12*std::min(dx[0], dx[1]);  //std::sqrt(dx[0]*dx[0] + dx[1]*dx[1]); 
    sig = 3.*std::min(dx[0],dx[1]); //std::sqrt(dx[0]*dx[0] + dx[1]*dx[1]); 

    amrex::Real K[5][5] = {}; //The same for every ratio;  
    amrex::Real Ktot[13][13] = {}; // The same for every ratio; 
    std::vector<std::array<amrex::Real, 13>> kt(r[0]*r[1], std::array<amrex::Real, 13>{{0}});
         // First dim is rx*ry; 
    GetK(K, Ktot); // Builds Covariance Matrices of Base Sample and Extended Samples/stencils  
    GetEigen(); //Gets Eigenvalues and Vectors from K for use in the interpolation 
    Decomp(K, Ktot); //Decomposes K and Ktot into their Cholesky Versions
    ks.resize(r[0]*r[1], std::array<std::array<amrex::Real, 5>, 5>() ); 
    GetKs(K); 
    // K and Ktot are not actually necessary for the rest of the GP interpolation 
    // They are only used to construct the weights w = ks^T Kinv 
    // and gam = Rinv Q^T kt; 
    // ks, gam, lam and V are part of the class and will be used in the main interpolation routine. 
    gam.resize(r[0]*r[1], std::array<amrex::Real, 5>()); 
    GetKtotks(Ktot, kt); 
    for(int i = 0; i < r[0]*r[1]; ++i){
        GetGamma(ks[i], kt[i], gam[i]); //Gets the gamma's 
    }
}

template<int n>
void
GP::CholeskyDecomp(amrex::Real (&K)[n][n])
{
     for(int j = 0; j < n; ++j){
        for( int k = 0; k < j; ++k){
            K[j][j] -= (K[j][k]*K[j][k]);
        }
        K[j][j] = std::sqrt(K[j][j]);
        for(int i = j+1; i < n; ++i){
            for(int k = 0; k < j; ++k){
                K[i][j] -= K[i][k]*K[j][k];
            }
            K[i][j] /= K[j][j];
        }
    }
}

//Performs Cholesky Backsubstitution
template<int n> 
void
GP::cholesky(amrex::Real (&b)[n], amrex::Real const K[n][n])
{
    /* Forward sub Ly = b */
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j) b[i] -= b[j]*K[i][j];
        b[i] /= K[i][i];
    }

    /* Back sub Ux = y */
    for(int i = n-1; i >= 0; --i){
        for(int j = i+1; j < n; ++j) b[i] -= K[j][i]*b[j];
        b[i] /= K[i][i];
    }
}

//Performs Cholesky Backsubstitution
template<int n> 
void
GP::cholesky(std::array<amrex::Real, n> &b, amrex::Real const K[n][n])
{
    /* Forward sub Ly = b */
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < i; ++j) b[i] -= b[j]*K[i][j];
        b[i] /= K[i][i];
    }

    /* Back sub Ux = y */
    for(int i = n-1; i >= 0; --i){
        for(int j = i+1; j < n; ++j) b[i] -= K[j][i]*b[j];
        b[i] /= K[i][i];
    }
}

//Builds the Covariance matrix K if uninitialized --> if(!init) GetK, weights etc.
//Four K totals to make the gammas.  
void
GP::GetK(amrex::Real (&K)[5][5], amrex::Real (&Ktot)[13][13])
{

    amrex::Real pnt[5][2] = {{ 0, -1}, 
                             {-1,  0}, 
                             { 0,  0}, 
                             { 1,  0}, 
                             { 0,  1}}; 

//    for(int i = 0; i < 5; ++i) K[i][i] = 1.e0; 
//Small K
    for(int i = 0; i < 5; ++i){
        for(int j = i; j < 5; ++j){
            K[i][j] = cov1(pnt[i], pnt[j], l);
            K[j][i] = K[i][j]; 
        }
    }
//    for(int i = 0; i < 13; ++i) Ktot[i][i] = 1.e0; 

    amrex::Real spnt[13][2] =  {{ 0, -2}, 
                                {-1, -1}, 
                                { 0, -1},
                                { 1, -1}, 
                                {-2,  0}, 
                                {-1,  0}, 
                                { 0,  0}, 
                                { 1,  0}, 
                                { 2,  0}, 
                                {-1,  1}, 
                                { 0,  1}, 
                                { 1,  1}, 
                                { 0,  2}}; 

    for(int i = 0; i < 13; ++i)
        for(int j = i; j <13; ++j){
            Ktot[i][j] = cov1(spnt[i], spnt[j], l); 
            Ktot[j][i] = Ktot[i][j]; 
        }
}

//We need to to the decomposition outside of the GetK routine so we can use K to get the 
//EigenVectors and Values. 

void
GP::Decomp(amrex::Real (&K)[5][5], amrex::Real (&Kt)[13][13])
{
    CholeskyDecomp<5>(K); 
    CholeskyDecomp<13>(Kt); 
}

//Use a Cholesky Decomposition to solve for k*K^-1 
//Inputs: K, outputs w = k*K^-1. 
//We need weights for each stencil. Therefore we'll have 5 arrays of 16 X 5 each. 

void 
GP::GetKs(const amrex::Real K[5][5])
{
    //Locations of new points relative to i,j 
    std::vector<std::array<amrex::Real,2>> pnt(r[0]*r[1], std::array<amrex::Real, 2>()); 
//    amrex::Real pnt[16][2]; 
    if(r[0] == 2 && r[1] == 2){
        pnt[0][0] = -0.25,  pnt[0][1] = -0.25; 
        pnt[1][0] =  0.25,  pnt[1][1] = -0.25; 
        pnt[2][0] = -0.25,  pnt[2][1] =  0.25; 
        pnt[3][0] =  0.25,  pnt[3][1] =  0.25; 
    }
    else if(r[0] == 4 && r[1]==4){
        pnt[0][0] = -.375,  pnt[0][1] = -.375; 
        pnt[1][0] = -.125,  pnt[1][1] = -.375; 
        pnt[2][0] = 0.125,  pnt[2][1] = -.375; 
        pnt[3][0] = 0.375,  pnt[3][1] = -.375; 
        pnt[4][0] = -.375,  pnt[4][1] = -.125; 
        pnt[5][0] = -.125,  pnt[5][1] = -.125; 
        pnt[6][0] = 0.125,  pnt[6][1] = -.125; 
        pnt[7][0] = 0.375,  pnt[7][1] = -.125; 
        pnt[8][0] = -.375,  pnt[8][1] = 0.125; 
        pnt[9][0] = -.125,  pnt[9][1] = 0.125; 
        pnt[10][0] = 0.125, pnt[10][1] = 0.125; 
        pnt[11][0] = 0.375, pnt[11][1] = 0.125; 
        pnt[12][0] = -.375, pnt[12][1] = 0.375; 
        pnt[13][0] = -.125, pnt[13][1] = 0.375; 
        pnt[14][0] = 0.125, pnt[14][1] = 0.375; 
        pnt[15][0] = 0.375, pnt[15][1] = 0.375; 
    }

    amrex::Real spnt[5][2]; 
    spnt[0][0] =  0 , spnt[0][1] = -1; 
    spnt[1][0] = -1,  spnt[1][1] = 0; 
    spnt[2][0] =  0 , spnt[2][1] = 0; 
    spnt[3][0] =  1 , spnt[3][1] = 0; 
    spnt[4][0] =  0 , spnt[4][1] = 1; 

    amrex::Real temp[2]; 
    //Build covariance vector between interpolant points and stencil 
     for(int i = 0; i < r[0]*r[1]; ++i){
        for(int j = 0; j < 5; ++j){
            temp[0] = spnt[j][0], temp[1] = spnt[j][1] - 1.0; //sten_jm
            ks[i][0][j] = cov2(pnt[i], temp);

            temp[0] = spnt[j][0] - 1.0, temp[1] =   spnt[j][1]; //sten_im
            ks[i][1][j] = cov2(pnt[i], temp);

            ks[i][2][j] = cov2(pnt[i], spnt[j]); //sten_cen
    
            temp[0] = spnt[j][0] + 1.0, temp[1] = spnt[j][1];
            ks[i][3][j] = cov2(pnt[i], temp); //sten_ip

            temp[0] = spnt[j][0], temp[1] = spnt[j][1] + 1.0; 
            ks[i][4][j] = cov2(pnt[i], temp); //sten_jp
        }
     //Backsubstitutes for k^TK^{-1} 
        for(int k = 0; k < 5; ++k)
            cholesky<5>(ks[i][k], K); 
   }
}

// Here we are using Kt to get the weights for the overdetermined  
// In this case, we will have 16 new points
// Therefore, we will need 16 b =  k*^T Ktot^(-1)
// K1 is already Choleskied  
void 
GP::GetKtotks(const amrex::Real K1[13][13], std::vector<std::array<amrex::Real, 13>> &kt)
{
   //Locations of new points relative to i,j 
    std::vector<std::array<amrex::Real,2>> pnt(r[0]*r[1], std::array<amrex::Real,2>()); 
//    amrex::Real pnt[16][2]; 
    if(r[0] == 2 && r[1] == 2){
        pnt[0][0] = -0.25,  pnt[0][1] = -0.25; 
        pnt[1][0] =  0.25,  pnt[1][1] = -0.25; 
        pnt[2][0] = -0.25,  pnt[2][1] =  0.25; 
        pnt[3][0] =  0.25,  pnt[3][1] =  0.25; 
    }
    else if(r[0] == 4 && r[1]==4){
        pnt[0][0] = -.375,  pnt[0][1] = -.375; 
        pnt[1][0] = -.125,  pnt[1][1] = -.375; 
        pnt[2][0] = 0.125,  pnt[2][1] = -.375; 
        pnt[3][0] = 0.375,  pnt[3][1] = -.375; 
        pnt[4][0] = -.375,  pnt[4][1] = -.125; 
        pnt[5][0] = -.125,  pnt[5][1] = -.125; 
        pnt[6][0] = 0.125,  pnt[6][1] = -.125; 
        pnt[7][0] = 0.375,  pnt[7][1] = -.125; 
        pnt[8][0] = -.375,  pnt[8][1] = 0.125; 
        pnt[9][0] = -.125,  pnt[9][1] = 0.125; 
        pnt[10][0] = 0.125, pnt[10][1] = 0.125; 
        pnt[11][0] = 0.375, pnt[11][1] = 0.125; 
        pnt[12][0] = -.375, pnt[12][1] = 0.375; 
        pnt[13][0] = -.125, pnt[13][1] = 0.375; 
        pnt[14][0] = 0.125, pnt[14][1] = 0.375; 
        pnt[15][0] = 0.375, pnt[15][1] = 0.375; 
    }
    //Super K positions 
    amrex::Real spnt[13][2] = {{ 0, -2},  
                               {-1, -1}, 
                               { 0, -1},
                               { 1, -1}, 
                               {-2,  0}, 
                               {-1,  0}, 
                               { 0,  0}, 
                               { 1,  0}, 
                               { 2,  0}, 
                               {-1,  1}, 
                               { 0,  1}, 
                               { 1,  1}, 
                               { 0,  2}}; 

    for(int i = 0; i < r[0]*r[1]; i++){
       for (int j = 0; j < 13; j++){
            kt[i][j] = cov2(pnt[i], spnt[j]); 
       }
       cholesky<13>(kt[i], K1); 
    } 
}

//Each point will have its
//own set of gammas. 
//Use x = R^-1Q'b 
void
GP::GetGamma(std::array<std::array<amrex::Real, 5>, 5> const& k,
             std::array<amrex::Real, 13> const& kt, 
             std::array<amrex::Real,5> &ga)
{
//Extended matrix Each column contains the vector of coviarances corresponding 
//to each sample (weno-like stencil)

    amrex::Real A[13*5] = { k[0][0], 0.e0   , 0.e0   , 0.e0   , 0.e0   , // i   j-2 
                            k[0][1], k[1][0], 0.e0   , 0.e0   , 0.e0   , // i-1 j-1
                            k[0][2], 0.e0   , k[2][0], 0.e0   , 0.e0   , // i   j-1
                            k[0][3], 0.e0   , 0.e0   , k[3][0], 0.e0   , // i+1 j-1
                            0.e0   , k[1][1], 0.e0   , 0.e0   , 0.e0   , // i-2 j
                            0.e0   , k[1][2], k[2][1], 0.e0   , 0.e0   , // i-1 j 
                            k[0][4], k[1][3], k[2][2], k[3][1], k[4][0], // i   j 
                            0.e0   , 0.e0   , k[2][3], k[3][2], 0.e0   , // i+1 j
                            0.e0   , 0.e0   , 0.e0   , k[3][3], 0.e0   , // i+2 j
                            0.e0   , k[1][4], 0.e0   , 0.e0   , k[4][1], // i-1 j+1
                            0.e0   , 0.e0   , k[2][4], 0.e0   , k[4][2], // i   j+1
                            0.e0   , 0.e0   , 0.e0   , k[3][4], k[4][3], // i+1 j+1
                            0.e0   , 0.e0   , 0.e0   , 0.e0   , k[4][4]};// i   j+2
 
    int m = 13, n = 5, nrhs = 1; 
    int lda = 5, ldb = 1, lwork = -1, info; 
    double temp[13]; 
    for(int i = 0; i < 13; i++) temp[i] = kt[i]; 
    double workt;
    info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, A, lda, temp, ldb);
    for(int i = 0; i < 5; ++i) ga[i] = temp[i];

}
 
extern "C"
{
 void dsyev_( char* jobz, char* uplo, int* n, double* a, int* lda,
                double* w, double* work, int* lwork, int* info );
}
 

void 
GP::GetEigen()
{
    double A[25];
    amrex::Real pnt[5][2] = {{ 0, -1}, 
                             {-1,  0}, 
                             { 0,  0}, 
                             { 1,  0}, 
                             { 0,  1}}; 

    for (int j = 0; j < 5; ++j){
//        A[j + 5*j] = 1.e0;  
        for(int i = j; i < 5; ++i){
             A[i + j*5] = cov1(pnt[i], pnt[j], sig); //this is K_sig
             A[j + 5*i] = A[i + j*5]; 
        }
    }

    int N = 5, lda = 5, info, lwork;
    double wkopt;
    double* work;
    lwork = -1;
    dsyev_( "Vectors", "Upper", &N, A, &lda, lam, &wkopt, &lwork, &info );
    lwork = (int)wkopt;
    work = (double*)malloc( lwork*sizeof(double) );
    dsyev_( "Vectors", "Upper", &N, A, &lda, lam, work, &lwork, &info );
    for (int j = 0; j < 5; ++j){
        for(int i = 0; i < 5; ++i) V[i][j] =  A[i + j*5];
    }
    free(work);


}
  

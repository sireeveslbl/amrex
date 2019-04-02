#include <AMReX_GP_2D.H>
#include <iostream> 
#include <fstream> 
#include <iomanip>

template<class T>
amrex::Real
GP::sqrexp(const T x[2],const T y[2], const amrex::Real *dx)
{
    amrex::Real result = std::exp(-0.5*((x[0] - y[0])*(x[0] - y[0])*dx[0]*dx[0] + 
                                        (x[1] - y[1])*(x[1] - y[1])*dx[1]*dx[1])/(l*l));
    return result;    
}
 
amrex::Real
GP::sqrexp(const std::array<amrex::Real, 2> x, const amrex::Real y[2], const amrex::Real *dx)
{
    amrex::Real result = std::exp(-0.5*((x[0] - y[0])*(x[0] - y[0])*dx[0]*dx[0] + 
                                        (x[1] - y[1])*(x[1] - y[1])*dx[1]*dx[1])/(l*l));
    return result;    
} 

    //Perfroms Cholesky Decomposition on covariance matrix K
void
GP::InitGP (const amrex::IntVect Ratio, const amrex::Real *dx)
{
    r = Ratio;
    l = 12*std::sqrt(dx[0]*dx[0] + dx[1]*dx[1]);  
    amrex::Real K[5][5] = {}; //The same for every ratio;  
    amrex::Real Ktot[13][13] = {}; // The same for every ratio; 
    std::vector<std::array<amrex::Real, 13>> kt(r[0]*r[1], std::array<amrex::Real, 13>{{0}});
         // First dim is rx*ry; 
    GetK(K, Ktot, dx); // Builds Covariance Matrices of Base Sample and Extended Samples/stencils  
    GetEigen(K); //Gets Eigenvalues and Vectors from K for use in the interpolation 
    Decomp(K, Ktot); //Decomposes K and Ktot into their Cholesky Versions
    ks.resize(r[0]*r[1], std::array<std::array<amrex::Real, 5>, 5>() ); 
    GetKs(K, dx); 
    // K and Ktot are not actually necessary for the rest of the GP interpolation 
    // They are only used to construct the weights w = ks^T Kinv 
    // and gam = Rinv Q^T kt; 
    // ks, gam, lam and V are part of the class and will be used in the main interpolation routine. 
    gam.resize(r[0]*r[1], std::array<amrex::Real, 5>()); 
    GetKtotks(Ktot, kt, dx); 
    for(int i = 0; i < r[0]*r[1]; ++i){
        GetGamma(ks[i], kt[i], gam[i]); //Gets the gamma's for the 
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

template<int n> 
void 
GP::matmul(const amrex::Real (&A)[n][n], const amrex::Real (&X)[n][n], 
            amrex::Real (&B)[n][n])
{
    amrex::Real temp; 
    for(int i = 0; i < n; i++) 
        for(int j = 0; j < n; j++){
           temp = 0.0; 
           for(int k = 0; k < n; k++){
               temp += A[i][k]*X[k][j];             
            }
            B[i][j] = temp; 
        }
}

template<int n> 
void 
GP::matmul(const std::vector<std::vector<amrex::Real>> &A, 
           const std::vector<std::vector<amrex::Real>> &X, 
            std::vector<std::vector<amrex::Real>> &B)
{
    amrex::Real temp; 
    for(int i = 0; i < n; i++) 
        for(int j = 0; j < n; j++){
           temp = 0.0; 
           for(int k = 0; k < n; k++){
               temp += A[i][k]*X[k][j];             
            }
            B[i][j] = temp; 
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
GP::GetK(amrex::Real (&K)[5][5], amrex::Real (&Ktot)[13][13],
                          const amrex::Real *dx)
{

    amrex::Real pnt[5][2] = {{ 0, -1}, 
                             {-1,  0}, 
                             { 0,  0}, 
                             { 1,  0}, 
                             { 0,  1}}; 

    for(int i = 0; i < 5; ++i) K[i][i] = 1.e0; 
//Small K
    for(int i = 0; i < 5; ++i)
        for(int j = i+1; j < 5; ++j){
            K[i][j] = sqrexp(pnt[i], pnt[j], dx); 
            K[j][i] = K[i][j]; 
        }
    for(int i = 0; i < 13; ++i) Ktot[i][i] = 1.e0; 

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
        for(int j = i+1; j <13; ++j){
            Ktot[i][j] = sqrexp(spnt[i], spnt[j], dx); 
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
GP::GetKs(const amrex::Real K[5][5],
          const amrex::Real *dx)
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
            ks[i][0][j] = sqrexp(pnt[i], temp, dx);

            temp[0] = spnt[j][0] - 1.0, temp[1] =   spnt[j][1]; //sten_im
            ks[i][1][j] = sqrexp(pnt[i], temp, dx);

            ks[i][2][j] = sqrexp(pnt[i], spnt[j], dx); //sten_cen
    
            temp[0] = spnt[j][0] + 1.0, temp[1] = spnt[j][1];
            ks[i][3][j] = sqrexp(pnt[i], temp, dx); //sten_ip

            temp[0] = spnt[j][0], temp[1] = spnt[j][1] + 1.0; 
            ks[i][4][j] = sqrexp(pnt[i], temp, dx); //sten_jp
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
GP::GetKtotks(const amrex::Real K1[13][13], std::vector<std::array<amrex::Real, 13>> &kt, 
                               const amrex::Real *dx)
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
            kt[i][j] = sqrexp(pnt[i], spnt[j], dx); 
       }
       cholesky<13>(kt[i], K1); 
    } 
}

//Solves Ux = b where U is upper triangular
template<int n> 
void 
GP::Ux_solve(const amrex::Real R[n][n], std::array<amrex::Real,n> &x, const amrex::Real b[n])
{
        double summ;
 
        for(int k = n-1; k>=0; --k){
            summ = 0.e0; 
            for(int i = k+1; i < n; i++){
                 summ += x[i]*(R[k][i]);
            }
            x[k] = (b[k] - summ)/R[k][k];
            
        }
}

// QR Decomposition routines! 
// In qr_decomp A is decomposed into R and V contains Q. 
void
GP::qr_decomp(std::vector<std::vector<amrex::Real>> &R, std::vector<std::vector<amrex::Real>> &Q, const int n)
{

    amrex::Real s, anorm, vnorm, innerprod;    
    std::vector<std::vector<amrex::Real>> v(n, std::vector<amrex::Real>(n,0)); 
    for(int j = 0; j < n; j++){
        anorm = 0.e0;
        vnorm = 0.e0;

        for(int i = j; i < n; i++) anorm += R[i][j]*R[i][j];
        anorm = std::sqrt(anorm);
        s = std::copysign(anorm, R[j][j]);         
        for(int i = j; i < n; i++){
            v[i][j] = R[i][j];
        }
        v[j][j] += s;
        for(int i = 0; i < n; i++) vnorm += v[i][j]*v[i][j];
        vnorm = std::sqrt(vnorm);
        if(vnorm>0) for(int i =0; i < n; i++) v[i][j] /= vnorm;

        for(int k = 0; k < n; k++){
            innerprod = 0.e0;
            for(int i = 0; i < n; i++) innerprod += v[i][j]*R[i][k];

            for(int i = 0; i < n; i++) R[i][k] -= 2.e0*v[i][j]*innerprod;
        }
    }
    q_appl(Q, v, n);
}

//QR decomp for the non-square matrix 
void 
GP::QR(amrex::Real (&A)[13][5], amrex::Real (&Q)[13][13],amrex::Real (&R)[5][5])
{
    //Q = I 
    double v[13] = {};  
    double norm, inner, s; 
    for(int i = 0; i < 13; ++i){
        for(int j = 0; j < 13; ++j)
            Q[i][j] = 0.e0;
        Q[i][i] = 1.e0;
    }

    for(int j = 0; j < 5; ++j){
       norm = 0.0; 
       for(int i = 0; i < j; ++i) v[i] = 0;
       for(int i = j; i < 13; ++i){
             v[i] = A[i][j]; 
             norm += v[i]*v[i];
        }
        norm = std::sqrt(norm); 
        s = std::copysign(norm, A[j][j]);
        v[j] += s; 
        norm = 0.e0; 
        for(int i = j; i < 13; i++) norm += v[i]*v[i];
        norm = std::sqrt(norm); 
        if(norm > 1e-14)
        {
            for(int i = j; i < 13; ++i) v[i] /= norm; //Normalize vector v
            for(int k = j; k < 5; ++k){
                inner = 0.e0;  
                for(int i = 0; i < 13; ++i) inner += v[i]*A[i][k]; 
                for(int i = 0; i < 13; ++i) A[i][k] -= 2.e0*inner*v[i];
            }
            for(int k = 0; k < 13; ++k){
                inner = 0.e0;  
                for(int i = 0; i < 13; ++i) inner += v[i]*Q[k][i]; 
                for(int i = 0; i < 13; ++i) Q[k][i] -= 2.e0*inner*v[i];
            }
        }
    }
    for(int i = 0; i < 5; i++) 
        for( int j = 0; j <5; j++) R[i][j] = A[i][j]; 
}

//Applies V onto A -> VA 
void
GP::q_appl(std::vector<std::vector<amrex::Real>> &A, const std::vector<std::vector<amrex::Real>> v, const int rows, 
       int upper/*=0 */, int mode /*=0*/) //may remove these in the future 
{
    if(upper == 0){
        if(mode == 0){ //Right Multiplication Q*A 
            for(int k = rows-1; k >= 0; --k){
                for(int i = rows-1; i >=0; --i){
                    amrex::Real temp = 0.0;
                    for(int j = 0; j < rows; ++j)
                        temp += A[j][k]*v[j][i];
                    for(int j = 0; j < rows; ++j){
                        A[j][k] -= 2.0*v[j][i]*temp;
                    }
                }
            }
        }
        else{ // Left Multiplication A*Q
            std::vector<std::vector<amrex::Real>> temp1(rows, std::vector<amrex::Real>(rows,0));
            //T =  
            for(int k = 0; k < rows; ++k)
                for(int j = 0; j < rows; ++j) 
                    temp1[k][j] = A[j][k]; 

            for(int k = 0; k < rows; k++){
                for(int i = rows-1; i >=0; --i){
                    amrex::Real temp = 0.0;
                    for(int j = 0; j < rows; ++j)
                        temp += temp1[j][k]*v[j][i];
                    for(int j = 0; j < rows; ++j)
                        temp1[j][k] -= 2.0*v[j][i]*temp;
                }
            }
            for(int k = 0; k < rows; ++k)
                for(int j = 0; j < rows; ++j) 
                   A[k][j] = temp1[j][k]; 
        }
    }

    else{
        std::vector<std::vector<amrex::Real>> temp1(rows, std::vector<amrex::Real>(rows, 0)); 
        for(int k = 0; k < rows; k++)
            for(int j = 0; j < upper - rows; j++) 
                temp1[k][j] = A[k][rows+j]; 
         for(int k = rows-1; k >= 0; --k){
            for(int i = rows-1; i >=0; --i){
                amrex::Real temp = 0;
                for(int j = 0; j < rows; ++j)
                    temp += temp1[j][k]*v[j][i];
                for(int j = 0; j < rows; ++j)
                    temp1[j][k] -= 2.0*v[j][i]*temp;
            }
        }
        for(int k = 0; k < rows; k++)
            for(int j = 0; j < upper - rows; j++){
                A[k][rows+j] = temp1[k][j]; 
        }
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
/*    amrex::Real A[13][5] = {{k[0][0], 0.e0   , 0.e0   , 0.e0   , 0.e0   }, 
                            {0.e0   , k[0][1], 0.e0   , 0.e0   , 0.e0   }, 
                            {k[1][0], 0.e0   , k[0][2], 0.e0   , 0.e0   }, 
                            {k[2][0], k[1][1], 0.e0   , k[0][3], 0.e0   }, 
                            {k[3][0], k[2][1], k[1][2], 0.e0   , k[0][4]}, 
                            {0.e0   , k[3][1], k[2][2], k[1][3], 0.e0   }, 
                            {0.e0   , 0.e0   , k[3][2], k[2][3], k[1][4]}, 
                            {0.e0   , 0.e0   , 0.e0   , k[3][3], k[2][4]}, 
                            {k[4][0], 0.e0   , 0.e0   , 0.e0   , k[3][4]}, 
                            {0.e0   , k[4][1], 0.e0   , 0.e0   , 0.e0   }, 
                            {0.e0   , 0.e0   , k[4][2], 0.e0   , 0.e0   }, 
                            {0.e0   , 0.e0   , 0.e0   , k[4][3], 0.e0   }, 
                            {0.e0   , 0.e0   , 0.e0   , 0.e0   , k[4][4]}}; // */ 

    amrex::Real A[13][5] = {{k[0][0], 0.e0   , 0.e0   , 0.e0   , 0.e0   }, 
                            {0.e0   , k[1][0], 0.e0   , 0.e0   , 0.e0   }, 
                            {k[0][2], 0.e0   , k[2][0], 0.e0   , 0.e0   }, 
                            {k[0][3], k[1][1], 0.e0   , k[3][0], 0.e0   }, 
                            {k[0][3], k[1][2], k[2][1], 0.e0   , k[4][0]}, 
                            {0.e0   , k[1][3], k[2][2], k[3][1], 0.e0   }, 
                            {0.e0   , 0.e0   , k[2][3], k[3][2], k[4][1]}, 
                            {0.e0   , 0.e0   , 0.e0   , k[3][3], k[4][2]}, 
                            {k[0][4], 0.e0   , 0.e0   , 0.e0   , k[4][3]}, 
                            {0.e0   , k[1][4], 0.e0   , 0.e0   , 0.e0   }, 
                            {0.e0   , 0.e0   , k[2][4], 0.e0   , 0.e0   }, 
                            {0.e0   , 0.e0   , 0.e0   , k[3][4], 0.e0   }, 
                            {0.e0   , 0.e0   , 0.e0   , 0.e0   , k[4][4]}};// */
   amrex::Real Q[13][13]; 
   amrex::Real R[5][5];
   QR(A, Q, R); // This one is for non-square matrices
   amrex::Real temp[5] ={0};

   //Q'*Kt 
   for(int i = 0; i < 5; i++)
      for(int j = 0; j < 13; j++)
        temp[i] += Q[j][i]*kt[j]; //Q'kt 
    //gam = R^-1 Q'kt 
    
    Ux_solve<5>(R, ga, temp);

}

template<int n> 
void 
GP::GetEigen(const amrex::Real K[n][n])
{
    std::vector<std::vector<amrex::Real>> Q(n, std::vector<amrex::Real>(n, 0.));
    std::vector<std::vector<amrex::Real>> temp(n, std::vector<amrex::Real>(n, 0.)); 
    std::vector<std::vector<amrex::Real>> vtemp(n, std::vector<amrex::Real>(n, 0.)); 
    std::vector<std::vector<amrex::Real>> v(n, std::vector<amrex::Real>(n,0.)); 
    std::vector<std::vector<amrex::Real>> B(n, std::vector<amrex::Real>(n, 0.));
    for(int i = 0; i < n; ++i) 
        for(int j = 0; j < n; ++j) B[i][j] = K[i][j];
 
    int iter;  

    for(int i = 0; i < n; i++){
        v[i][i] = 1.0; 
    }

    amrex::Real mu, er; 
    for(int j = n-1; j > 0; j--){
        er = 1.e0; 
        iter =0; 
        while(er > 1e-16){
            mu = B[j][j]; 
            for(int i = 0; i < n; i++){ B[i][i] -= mu;
                for(int jj = 0; jj < n; jj++) Q[i][jj] = (i == jj) ? 1.0 : 0.0; 
            }
            qr_decomp(B, Q, n);
            matmul<n>(B, Q, temp); 
            matmul<n>(v, Q, vtemp);
            B = temp; 
            v = vtemp;

            for(int i = 0; i < n; i++) B[i][i] += mu;
            er = std::abs(B[j][j-1]);
//            er = std::abs(lamt - B[j][j]); 
            iter++;
            if(iter>500){
                 for(int ii = 0; ii < n; ii++)
                    { for(int jj = 0; jj < n; jj++) std::cout<< K[ii][jj] << '\t';
                    }
                    std::cout<< std::endl;  
                 std::cin.get(); 
                }
        }
    }
    for(int i = 0; i < n; i++){
         lam[i] = B[i][i]; 
         for(int j = 0; j < n; j++){
                 V[i][j] = v[i][j]; 
         }
         
    }
}
  

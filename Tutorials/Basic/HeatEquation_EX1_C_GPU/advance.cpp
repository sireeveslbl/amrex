
#include "myfunc.H"
#include "myfunc_F.H"
#include <AMReX_BLProfiler.H>
#define ARRAY_2D(PHI, LO_X, LO_Y, HI_X, HI_Y, I, J) PHI[(J-LO_Y)*(HI_X-LO_X+1)+I-LO_X]


using namespace amrex;


void advance_phi(const int& lox, const int& loy, const int& hix, const int& hiy,
                const amrex::Real* phi_old,
                const int& phi_old_lox, const int& phi_old_loy, const int& phi_old_hix, const int& phi_old_hiy,
                amrex::Real* phi_new,
                const int& phi_new_lox, const int& phi_new_loy, const int& phi_new_hix, const int& phi_new_hiy,
                const amrex::Real& dx, const amrex::Real& dy, const amrex::Real& dt)
{
    for (int j = loy; j <= hiy; ++j ) {
        for (int i = lox; i <= hix; ++i ) {
            ARRAY_2D(phi_new,phi_new_lox,phi_new_loy,phi_new_hix,phi_new_hiy,i,j) =
                ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) +
                dt/(dx*dx) * (ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i+1,j) - 
                              2*ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) + 
                              ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i-1,j)
                             ) +
                dt/(dy*dy) * (ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j+1) - 
                              2*ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j) + 
                              ARRAY_2D(phi_old,phi_old_lox,phi_old_loy,phi_old_hix,phi_old_hiy,i,j-1)
                             );
        }
    }
}

void advance (MultiFab& old_phi, MultiFab& new_phi,
	      std::array<MultiFab, AMREX_SPACEDIM>& flux,
	      Real dt, const Geometry& geom)
{
    BL_PROFILE("advance");
    // Fill the ghost cells of each grid from the other grids
    // includes periodic domain boundaries
    old_phi.FillBoundary(geom.periodicity());

    // Fill non-periodic physical boundaries
    fill_physbc(old_phi, geom);

    int Ncomp = old_phi.nComp();
    int ng_p = old_phi.nGrow();
    int ng_f = flux[0].nGrow();

    const Real* dx = geom.CellSize();

// #ifdef _OPENMP
// #pragma omp parallel
// #endif
    for ( MFIter mfi(old_phi); mfi.isValid(); ++mfi )
    {
        const Box& bx = mfi.validbox();
        const int* lo = bx.loVect();
        const int* hi = bx.hiVect();
        advance_phi(lo[0],lo[1],hi[0],hi[1],
                old_phi[mfi].dataPtr(), 
                old_phi[mfi].loVect()[0], old_phi[mfi].loVect()[1],
                old_phi[mfi].hiVect()[0], old_phi[mfi].hiVect()[1],
                new_phi[mfi].dataPtr(), 
                new_phi[mfi].loVect()[0], new_phi[mfi].loVect()[1],
                new_phi[mfi].hiVect()[0], new_phi[mfi].hiVect()[1],
                dx[0], dx[1], dt);

    }
    
}

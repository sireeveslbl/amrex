
#include <climits>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_INTERP_F.H>
#include <AMReX_Interp_C.H>
#include <AMReX_GP.H>

namespace amrex {

//
// PCInterp, NodeBilinear, and CellConservativeLinear are supported for all dimensions on cpu and gpu.
//
// CellConsertiveProtected only works in 2D and 3D on cpu.
//
// CellBilinear only works in 1D and 2D on cpu.
//
// CellQuadratic only works in 2D and 3D on cpu.
//
// CellConservativeQuartic only works with ref ratio of 2 on cpu
//

//
// CONSTRUCT A GLOBAL OBJECT OF EACH VERSION.
//
PCInterp                  pc_interp;
NodeBilinear              node_bilinear_interp;
CellBilinear              cell_bilinear_interp;
CellQuadratic             quadratic_interp;
CellConservativeLinear    lincc_interp;
CellConservativeLinear    cell_cons_interp(0);
CellConservativeProtected protected_interp;
CellConservativeQuartic   quartic_interp;
CellGaussianProcess       gp_interp; 

Interpolater::~Interpolater () {}

InterpolaterBoxCoarsener
Interpolater::BoxCoarsener (const IntVect& ratio)
{
    return InterpolaterBoxCoarsener(this, ratio);
}

Box
InterpolaterBoxCoarsener::doit (const Box& fine) const
{
    return mapper->CoarseBox(fine, ratio);
}

BoxConverter*
InterpolaterBoxCoarsener::clone () const
{
    return new InterpolaterBoxCoarsener(mapper, ratio);
}

NodeBilinear::~NodeBilinear () {}

Box
NodeBilinear::CoarseBox (const Box& fine,
                         int        ratio)
{
    Box b = amrex::coarsen(fine,ratio);

    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        if (b.length(i) < 2)
        {
            //
            // Don't want degenerate boxes.
            //
            b.growHi(i,1);
        }
    }

    return b;
}

Box
NodeBilinear::CoarseBox (const Box&     fine,
                         const IntVect& ratio)
{
    Box b = amrex::coarsen(fine,ratio);

    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        if (b.length(i) < 2)
        {
            //
            // Don't want degenerate boxes.
            //
            b.growHi(i,1);
        }
    }

    return b;
}

void
NodeBilinear::interp (const FArrayBox&  crse,
                      int               crse_comp,
                      FArrayBox&        fine,
                      int               fine_comp,
                      int               ncomp,
                      const Box&        fine_region,
                      const IntVect&    ratio,
                      const Geometry& /*crse_geom */,
                      const Geometry& /*fine_geom */,
                      Vector<BCRec>&   /*bcr*/,
                      int               /*actual_comp*/,
                      int               /*actual_state*/)
{
    BL_PROFILE("NodeBilinear::interp()");

    FArrayBox const* crsep = &crse;
    FArrayBox* finep = &fine;

    Gpu::LaunchSafeGuard lg(Gpu::isGpuPtr(crsep) && Gpu::isGpuPtr(finep));

    int num_slope  = ncomp*(AMREX_D_TERM(2,*2,*2)-1);
    const Box cslope_bx = amrex::enclosedCells(CoarseBox(fine_region, ratio));
    AsyncFab as_slopefab(cslope_bx, num_slope);
    FArrayBox* slopefab = as_slopefab.fabPtr();

    AMREX_LAUNCH_HOST_DEVICE_LAMBDA (cslope_bx, tbx,
    {
        amrex::nodebilin_slopes(tbx, *slopefab, *crsep, crse_comp, ncomp, ratio);
    });

    AMREX_LAUNCH_HOST_DEVICE_LAMBDA (fine_region, tbx,
    {
        amrex::nodebilin_interp(tbx, *finep, fine_comp, ncomp, *slopefab, *crsep, crse_comp, ratio);
    });
}

CellBilinear::~CellBilinear () {}

Box
CellBilinear::CoarseBox (const Box& fine,
                         int        ratio)
{
    return CoarseBox(fine, ratio*IntVect::TheUnitVector());
}

Box
CellBilinear::CoarseBox (const Box&     fine,
                         const IntVect& ratio)
{
    const int* lo = fine.loVect();
    const int* hi = fine.hiVect();

    Box crse(amrex::coarsen(fine,ratio));
    const int* clo = crse.loVect();
    const int* chi = crse.hiVect();

    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        int iratio = ratio[i];
        int hrat   = iratio/2;
        if (lo[i] <  clo[i]*ratio[i] + hrat)
            crse.growLo(i,1);
        if (hi[i] >= chi[i]*ratio[i] + hrat)
            crse.growHi(i,1);
    }
    return crse;
}

void
CellBilinear::interp (const FArrayBox&  crse,
                      int               crse_comp,
                      FArrayBox&        fine,
                      int               fine_comp,
                      int               ncomp,
                      const Box&        fine_region,
                      const IntVect &   ratio,
                      const Geometry& /*crse_geom*/,
                      const Geometry& /*fine_geom*/,
                      Vector<BCRec>&   /*bcr*/,
                      int               actual_comp,
                      int               actual_state)
{
    BL_PROFILE("CellBilinear::interp()");
#if (AMREX_SPACEDIM == 3)
    amrex::Error("interp: not implemented");
#endif
    //
    // Set up to call FORTRAN.
    //
    const int* clo = crse.box().loVect();
    const int* chi = crse.box().hiVect();
    const int* flo = fine.loVect();
    const int* fhi = fine.hiVect();
    const int* lo  = fine_region.loVect();
    const int* hi  = fine_region.hiVect();
    int num_slope  = AMREX_D_TERM(2,*2,*2)-1;
    int len0       = crse.box().length(0);
    int slp_len    = num_slope*len0;

    Vector<Real> slope(slp_len);

    int strp_len = len0*ratio[0];

    Vector<Real> strip(strp_len);

    int strip_lo = ratio[0] * clo[0];
    int strip_hi = ratio[0] * chi[0];

    const Real* cdat  = crse.dataPtr(crse_comp);
    Real*       fdat  = fine.dataPtr(fine_comp);
    const int* ratioV = ratio.getVect();

    amrex_cbinterp (cdat,AMREX_ARLIM(clo),AMREX_ARLIM(chi),AMREX_ARLIM(clo),AMREX_ARLIM(chi),
                   fdat,AMREX_ARLIM(flo),AMREX_ARLIM(fhi),AMREX_ARLIM(lo),AMREX_ARLIM(hi),
                   AMREX_D_DECL(&ratioV[0],&ratioV[1],&ratioV[2]),&ncomp,
                   slope.dataPtr(),&num_slope,strip.dataPtr(),&strip_lo,&strip_hi,
                   &actual_comp,&actual_state);
}

Vector<int>
Interpolater::GetBCArray (const Vector<BCRec>& bcr)
{
    Vector<int> bc(2*AMREX_SPACEDIM*bcr.size());

    for (int n = 0; n < bcr.size(); n++)
    {
        const int* b_rec = bcr[n].vect();

        for (int m = 0; m < 2*AMREX_SPACEDIM; m++)
        {
            bc[2*AMREX_SPACEDIM*n + m] = b_rec[m];
        }
    }

    return bc;
}

CellConservativeLinear::CellConservativeLinear (bool do_linear_limiting_)
{
    do_linear_limiting = do_linear_limiting_;
}

CellConservativeLinear::~CellConservativeLinear ()
{}

Box
CellConservativeLinear::CoarseBox (const Box&     fine,
                                   const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

Box
CellConservativeLinear::CoarseBox (const Box& fine,
                                   int        ratio)
{
    Box crse(amrex::coarsen(fine,ratio));
    crse.grow(1);
    return crse;
}

void
CellConservativeLinear::interp (const FArrayBox& crse,
                                int              crse_comp,
                                FArrayBox&       fine,
                                int              fine_comp,
                                int              ncomp,
                                const Box&       fine_region,
                                const IntVect&   ratio,
                                const Geometry&  crse_geom,
                                const Geometry&  fine_geom,
                                Vector<BCRec>&   bcr,
                                int              /*actual_comp*/,
                                int              /*actual_state*/)
{
    BL_PROFILE("CellConservativeLinear::interp()");
    BL_ASSERT(bcr.size() >= ncomp);

    AMREX_ASSERT(fine.box().contains(fine_region));

    FArrayBox const* crsep = &crse;
    FArrayBox* finep = &fine;

    Gpu::LaunchSafeGuard lg(Gpu::isGpuPtr(crsep) && Gpu::isGpuPtr(finep));

    const Box& crse_region = CoarseBox(fine_region,ratio);
    const Box& cslope_bx = amrex::grow(crse_region,-1);

    AsyncArray<BCRec> async_bcr(bcr.data(), ncomp);
    BCRec* bcrp = async_bcr.data();

    // component of ccfab : slopes for first compoent for x-direction
    //                      slopes for second component for x-direction
    //                      ...
    //                      slopes for last component for x-direction
    //                      slopes for y-direction
    //                      slopes for z-drction
    // then followed by
    //      lin_lim = true : factors (one for all components) for x, y and z-direction
    //      lin_lim = false: min for every component followed by max for every component
    const int ntmp = do_linear_limiting ? (ncomp+1)*AMREX_SPACEDIM : ncomp*(AMREX_SPACEDIM+2);
    AsyncFab as_ccfab(cslope_bx, ntmp);
    FArrayBox* ccfab = as_ccfab.fabPtr();

    const Vector<Real>& vec_voff = amrex::ccinterp_compute_voff(cslope_bx, ratio, crse_geom, fine_geom);

    AsyncArray<Real> async_voff(vec_voff.data(), vec_voff.size());
    Real const* voff = async_voff.data();

    if (do_linear_limiting) {
        AMREX_LAUNCH_HOST_DEVICE_LAMBDA (cslope_bx, tbx,
        {
            amrex::cellconslin_slopes_linlim(tbx, *ccfab, *crsep, crse_comp, ncomp, bcrp);
        });

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA (fine_region, tbx,
        {
            amrex::cellconslin_interp(tbx, *finep, fine_comp, ncomp, *ccfab, *crsep, crse_comp,
                                      voff, ratio);
        });
    } else {
        const Box& fslope_bx = amrex::refine(cslope_bx,ratio);
        AsyncFab as_fafab(fslope_bx, ncomp);
        FArrayBox* fafab = as_fafab.fabPtr();

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA (cslope_bx, tbx,
        {
            amrex::cellconslin_slopes_mclim(tbx, *ccfab, *crsep, crse_comp, ncomp, bcrp);
        });

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA (fslope_bx, tbx,
        {
            amrex::cellconslin_fine_alpha(tbx, *fafab, *ccfab, ncomp, voff, ratio);
        });

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA (cslope_bx, tbx,
        {
            amrex::cellconslin_slopes_mmlim(tbx, *ccfab, *fafab, ncomp, ratio);
        });

        AMREX_LAUNCH_HOST_DEVICE_LAMBDA (fine_region, tbx,
        {
            amrex::cellconslin_interp(tbx, *finep, fine_comp, ncomp, *ccfab, *crsep, crse_comp,
                                      voff, ratio);
        });
    }
}

CellQuadratic::CellQuadratic (bool limit)
{
    do_limited_slope = limit;
}

CellQuadratic::~CellQuadratic () {}

Box
CellQuadratic::CoarseBox (const Box&     fine,
                          const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

Box
CellQuadratic::CoarseBox (const Box& fine,
                          int        ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

void
CellQuadratic::interp (const FArrayBox& crse,
                       int              crse_comp,
                       FArrayBox&       fine,
                       int              fine_comp,
                       int              ncomp,
                       const Box&       fine_region,
                       const IntVect&   ratio,
                       const Geometry&  crse_geom,
                       const Geometry&  fine_geom,
                       Vector<BCRec>&    bcr,
                       int              actual_comp,
                       int              actual_state)
{
    BL_PROFILE("CellQuadratic::interp()");
    BL_ASSERT(bcr.size() >= ncomp);
    //
    // Make box which is intersection of fine_region and domain of fine.
    //
    Box target_fine_region = fine_region & fine.box();

    Box crse_bx(amrex::coarsen(target_fine_region,ratio));
    Box fslope_bx(amrex::refine(crse_bx,ratio));
    Box cslope_bx(crse_bx);
    cslope_bx.grow(1);
    BL_ASSERT(crse.box().contains(cslope_bx));
    //
    // Alloc temp space for coarse grid slopes: here we use 5
    // instead of AMREX_SPACEDIM because of the x^2, y^2 and xy terms
    //
    long t_long = cslope_bx.numPts();
    BL_ASSERT(t_long < INT_MAX);
    int c_len = int(t_long);

    Vector<Real> cslope(5*c_len);

    int loslp = cslope_bx.index(crse_bx.smallEnd());
    int hislp = cslope_bx.index(crse_bx.bigEnd());

    t_long = cslope_bx.numPts();
    BL_ASSERT(t_long < INT_MAX);
    int cslope_vol = int(t_long);
    int clo        = 1 - loslp;
    int chi        = clo + cslope_vol - 1;
    c_len          = hislp - loslp + 1;
    //
    // Alloc temp space for one strip of fine grid slopes: here we use 5
    // instead of AMREX_SPACEDIM because of the x^2, y^2 and xy terms.
    //
    int dir;
    int f_len = fslope_bx.longside(dir);

    Vector<Real> strip((5+2)*f_len);

    Real* fstrip = strip.dataPtr();
    Real* foff   = fstrip + f_len;
    Real* fslope = foff + f_len;
    //
    // Get coarse and fine edge-centered volume coordinates.
    //
    Vector<Real> fvc[AMREX_SPACEDIM];
    Vector<Real> cvc[AMREX_SPACEDIM];
    for (dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        fine_geom.GetEdgeVolCoord(fvc[dir],target_fine_region,dir);
        crse_geom.GetEdgeVolCoord(cvc[dir],crse_bx,dir);
    }
    //
    // Alloc tmp space for slope calc and to allow for vectorization.
    //
    Real* fdat        = fine.dataPtr(fine_comp);
    const Real* cdat  = crse.dataPtr(crse_comp);
    const int* flo    = fine.loVect();
    const int* fhi    = fine.hiVect();
    const int* fblo   = target_fine_region.loVect();
    const int* fbhi   = target_fine_region.hiVect();
    const int* cblo   = crse_bx.loVect();
    const int* cbhi   = crse_bx.hiVect();
    const int* fslo   = fslope_bx.loVect();
    const int* fshi   = fslope_bx.hiVect();
    int slope_flag    = (do_limited_slope ? 1 : 0);
    Vector<int> bc     = GetBCArray(bcr);
    const int* ratioV = ratio.getVect();

#if (AMREX_SPACEDIM > 1)

    amrex_cqinterp (fdat,AMREX_ARLIM(flo),AMREX_ARLIM(fhi),
                   AMREX_ARLIM(fblo), AMREX_ARLIM(fbhi),
                   &ncomp,AMREX_D_DECL(&ratioV[0],&ratioV[1],&ratioV[2]),
                   cdat,&clo,&chi,
                   AMREX_ARLIM(cblo), AMREX_ARLIM(cbhi),
                   fslo,fshi,
                   cslope.dataPtr(),&c_len,fslope,fstrip,&f_len,foff,
                   bc.dataPtr(), &slope_flag,
                   AMREX_D_DECL(fvc[0].dataPtr(),fvc[1].dataPtr(),fvc[2].dataPtr()),
                   AMREX_D_DECL(cvc[0].dataPtr(),cvc[1].dataPtr(),cvc[2].dataPtr()),
                   &actual_comp,&actual_state);

#endif /*(AMREX_SPACEDIM > 1)*/
}

PCInterp::~PCInterp () {}

Box
PCInterp::CoarseBox (const Box& fine,
                     int        ratio)
{
    return amrex::coarsen(fine,ratio);
}

Box
PCInterp::CoarseBox (const Box&     fine,
                     const IntVect& ratio)
{
    return amrex::coarsen(fine,ratio);
}

void
PCInterp::interp (const FArrayBox& crse,
                  int              crse_comp,
                  FArrayBox&       fine,
                  int              fine_comp,
                  int              ncomp,
                  const Box&       fine_region,
                  const IntVect&   ratio,
                  const Geometry& /*crse_geom*/,
                  const Geometry& /*fine_geom*/,
                  Vector<BCRec>&   /*bcr*/,
                  int               /*actual_comp*/,
                  int               /*actual_state*/)
{
    BL_PROFILE("PCInterp::interp()");

    FArrayBox const* crsep = &crse;
    FArrayBox* finep = &fine;

    Gpu::LaunchSafeGuard lg(Gpu::isGpuPtr(crsep) && Gpu::isGpuPtr(finep));

    AMREX_LAUNCH_HOST_DEVICE_LAMBDA (fine_region, tbx,
    {
        amrex::pcinterp_interp(tbx,*finep,fine_comp,ncomp,*crsep,crse_comp,ratio);
    });
}

CellConservativeProtected::CellConservativeProtected () {}

CellConservativeProtected::~CellConservativeProtected () {}

Box
CellConservativeProtected::CoarseBox (const Box&     fine,
                                      const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

Box
CellConservativeProtected::CoarseBox (const Box& fine,
                                      int        ratio)
{
    Box crse(amrex::coarsen(fine,ratio));
    crse.grow(1);
    return crse;
}

void
CellConservativeProtected::interp (const FArrayBox& crse,
                                   int              crse_comp,
                                   FArrayBox&       fine,
                                   int              fine_comp,
                                   int              ncomp,
                                   const Box&       fine_region,
                                   const IntVect&   ratio,
                                   const Geometry&  crse_geom,
                                   const Geometry&  fine_geom,
                                   Vector<BCRec>&    bcr,
                                   int              actual_comp,
                                   int              actual_state)
{
    BL_PROFILE("CellConservativeProtected::interp()");
    BL_ASSERT(bcr.size() >= ncomp);

    AMREX_ASSERT(fine.box().contains(fine_region));

    FArrayBox const* crsep = &crse;
    FArrayBox* finep = &fine;

    Gpu::LaunchSafeGuard lg(Gpu::isGpuPtr(crsep) && Gpu::isGpuPtr(finep));

    const Box& crse_region = CoarseBox(fine_region,ratio);
    const Box& cslope_bx = amrex::grow(crse_region,-1);

    AsyncArray<BCRec> async_bcr(bcr.data(), ncomp);
    BCRec* bcrp = async_bcr.data();

    // component of ccfab : slopes for first compoent for x-direction
    //                      slopes for second component for x-direction
    //                      ...
    //                      slopes for last component for x-direction
    //                      slopes for y-direction
    //                      slopes for z-drction
    // then followed by
    //                      factors (one for all components) for x, y and z-direction
    const int ntmp = (ncomp+1)*AMREX_SPACEDIM;
    AsyncFab as_ccfab(cslope_bx, ntmp);
    FArrayBox* ccfab = as_ccfab.fabPtr();

    const Vector<Real>& vec_voff = amrex::ccinterp_compute_voff(cslope_bx, ratio, crse_geom, fine_geom);

    AsyncArray<Real> async_voff(vec_voff.data(), vec_voff.size());
    Real const* voff = async_voff.data();

    AMREX_LAUNCH_HOST_DEVICE_LAMBDA (cslope_bx, tbx,
    {
        amrex::cellconslin_slopes_linlim(tbx, *ccfab, *crsep, crse_comp, ncomp, bcrp);
    });

    AMREX_LAUNCH_HOST_DEVICE_LAMBDA (fine_region, tbx,
    {
        amrex::cellconslin_interp(tbx, *finep, fine_comp, ncomp, *ccfab, *crsep, crse_comp,
                                  voff, ratio);
    });
}

void
CellConservativeProtected::protect (const FArrayBox& crse,
                                    int              crse_comp,
                                    FArrayBox&       fine,
                                    int              fine_comp,
                                    FArrayBox&       fine_state,
                                    int              state_comp,
                                    int              ncomp,
                                    const Box&       fine_region,
                                    const IntVect&   ratio,
                                    const Geometry&  crse_geom,
                                    const Geometry&  fine_geom,
                                    Vector<BCRec>& bcr)
{
    BL_PROFILE("CellConservativeProtected::protect()");
    BL_ASSERT(bcr.size() >= ncomp);

    //
    // Make box which is intersection of fine_region and domain of fine.
    //
    Box target_fine_region = fine_region & fine.box();

    //
    // crse_bx is coarsening of target_fine_region, grown by 1.
    //
    Box crse_bx = CoarseBox(target_fine_region,ratio);

    //
    // cs_bx is coarsening of target_fine_region.
    //
    Box cs_bx(crse_bx);
    cs_bx.grow(-1);

    //
    // Get coarse and fine edge-centered volume coordinates.
    //
    int dir;
    Vector<Real> fvc[AMREX_SPACEDIM];
    Vector<Real> cvc[AMREX_SPACEDIM];
    for (dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        fine_geom.GetEdgeVolCoord(fvc[dir],target_fine_region,dir);
        crse_geom.GetEdgeVolCoord(cvc[dir],crse_bx,dir);
    }

#if (AMREX_SPACEDIM == 2)
    const int* cvcblo = crse_bx.loVect();
    const int* fvcblo = target_fine_region.loVect();

    int cvcbhi[AMREX_SPACEDIM];
    int fvcbhi[AMREX_SPACEDIM];

    for (dir=0; dir<AMREX_SPACEDIM; dir++)
    {
        cvcbhi[dir] = cvcblo[dir] + cvc[dir].size() - 1;
        fvcbhi[dir] = fvcblo[dir] + fvc[dir].size() - 1;
    }
#endif

    Real* fdat       = fine.dataPtr(fine_comp);
    Real* state_dat  = fine_state.dataPtr(state_comp);
    const Real* cdat = crse.dataPtr(crse_comp);

    const int* flo    = fine.loVect();
    const int* fhi    = fine.hiVect();
    const int* slo    = fine_state.loVect();
    const int* shi    = fine_state.hiVect();
    const int* clo    = crse.loVect();
    const int* chi    = crse.hiVect();
    const int* fblo   = target_fine_region.loVect();
    const int* fbhi   = target_fine_region.hiVect();
    const int* csbhi  = cs_bx.hiVect();
    const int* csblo  = cs_bx.loVect();

    Vector<int> bc     = GetBCArray(bcr);
    const int* ratioV = ratio.getVect();

#if (AMREX_SPACEDIM > 1)

    amrex_protect_interp (fdat,AMREX_ARLIM(flo),AMREX_ARLIM(fhi),
                         fblo, fbhi,
                         cdat,AMREX_ARLIM(clo),AMREX_ARLIM(chi),
                         csblo, csbhi,
#if (AMREX_SPACEDIM == 2)
                         fvc[0].dataPtr(),fvc[1].dataPtr(),
                         AMREX_ARLIM(fvcblo), AMREX_ARLIM(fvcbhi),
                         cvc[0].dataPtr(),cvc[1].dataPtr(),
                         AMREX_ARLIM(cvcblo), AMREX_ARLIM(cvcbhi),
#endif
                         state_dat, AMREX_ARLIM(slo), AMREX_ARLIM(shi),
                         &ncomp,AMREX_D_DECL(&ratioV[0],&ratioV[1],&ratioV[2]),
                         bc.dataPtr());

#endif /*(AMREX_SPACEDIM > 1)*/

}

CellConservativeQuartic::~CellConservativeQuartic () {}

Box
CellConservativeQuartic::CoarseBox (const Box& fine,
				    int        ratio)
{
    Box crse(amrex::coarsen(fine,ratio));
    crse.grow(2);
    return crse;
}

Box
CellConservativeQuartic::CoarseBox (const Box&     fine,
				    const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(2);
    return crse;
}

void
CellConservativeQuartic::interp (const FArrayBox&  crse,
				 int               crse_comp,
				 FArrayBox&        fine,
				 int               fine_comp,
				 int               ncomp,
				 const Box&        fine_region,
				 const IntVect&    ratio,
				 const Geometry&   /* crse_geom */,
				 const Geometry&   /* fine_geom */,
				 Vector<BCRec>&     bcr,
				 int               actual_comp,
				 int               actual_state)
{
    BL_PROFILE("CellConservativeQuartic::interp()");
    BL_ASSERT(bcr.size() >= ncomp);
    BL_ASSERT(ratio[0]==2);
#if (AMREX_SPACEDIM >= 2)
    BL_ASSERT(ratio[0] == ratio[1]);
#endif
#if (AMREX_SPACEDIM == 3)
    BL_ASSERT(ratio[1] == ratio[2]);
#endif

    //
    // Make box which is intersection of fine_region and domain of fine.
    //
    Box target_fine_region = fine_region & fine.box();
    //
    // crse_bx is coarsening of target_fine_region, grown by 2.
    //
    Box crse_bx = CoarseBox(target_fine_region,ratio);

    Box crse_bx2(crse_bx);
    crse_bx2.grow(-2);
    Box fine_bx2 = amrex::refine(crse_bx2,ratio);

    Real* fdat       = fine.dataPtr(fine_comp);
    const Real* cdat = crse.dataPtr(crse_comp);

    const int* flo    = fine.loVect();
    const int* fhi    = fine.hiVect();
    const int* clo    = crse.loVect();
    const int* chi    = crse.hiVect();
    const int* fblo   = target_fine_region.loVect();
    const int* fbhi   = target_fine_region.hiVect();
    const int* cblo   = crse_bx.loVect();
    const int* cbhi   = crse_bx.hiVect();
    const int* cb2lo  = crse_bx2.loVect();
    const int* cb2hi  = crse_bx2.hiVect();
    const int* fb2lo  = fine_bx2.loVect();
    const int* fb2hi  = fine_bx2.hiVect();

    Vector<int> bc     = GetBCArray(bcr);
    const int* ratioV = ratio.getVect();

    int ltmp = fb2hi[0]-fb2lo[0]+1;
    Vector<Real> ftmp(ltmp);

#if (AMREX_SPACEDIM >= 2)
    ltmp = (cbhi[0]-cblo[0]+1)*ratio[1];
    Vector<Real> ctmp(ltmp);
#endif

#if (AMREX_SPACEDIM == 3)
    ltmp = (cbhi[0]-cblo[0]+1)*(cbhi[1]-cblo[1]+1)*ratio[2];
    Vector<Real> ctmp2(ltmp);
#endif

    amrex_quartinterp (fdat,AMREX_ARLIM(flo),AMREX_ARLIM(fhi),
		      fblo, fbhi, fb2lo, fb2hi,
		      cdat,AMREX_ARLIM(clo),AMREX_ARLIM(chi),
		      cblo, cbhi, cb2lo, cb2hi,
		      &ncomp,
		      AMREX_D_DECL(&ratioV[0],&ratioV[1],&ratioV[2]),
		      AMREX_D_DECL(ftmp.dataPtr(), ctmp.dataPtr(), ctmp2.dataPtr()),
		      bc.dataPtr(),&actual_comp,&actual_state);
}

//Cell GP interp SR Dissertation Work 
CellGaussianProcess::~CellGaussianProcess () {}

Box
CellGaussianProcess::CoarseBox (const Box&     fine,
                          const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(2);
    return crse;
}

Box
CellGaussianProcess::CoarseBox (const Box& fine,
                          int        ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(2);
    return crse;
} 



AMREX_GPU_DEVICE
inline
void 
CellGaussianProcess::amrex_gpinterp(Box const& bx, FArrayBox& finefab,
                                    const int fcomp, const int ncomp, 
                                    FArrayBox const& crsefab, const int ccomp, 
                                    IntVect ratio,   
                                    const amrex::Real ks[16][5][5], 
                                    const amrex::Real V[5][5], 
                                    const amrex::Real lam[5], 
                                    const amrex::Real gam[16][5])
{
#if (AMREX_SPACEDIM!=2)
    Abort("Not completed for ND, only 2D!"); 
#endif   
 
    const auto len  = amrex::length(bx);
    const auto lo   = amrex::lbound(bx);
    const auto crse = crsefab.view(lo, ccomp); 
    
    const auto flo = amrex::refine(lo, ratio); 
    const auto fine = finefab.view(flo,fcomp);
    amrex::Real beta[5], ws[5];
    for(int n = 0; n < ncomp; ++n) { 
        for (int jc = 0; jc < len.y; ++jc){ 
            AMREX_PRAGMA_SIMD
            for(int ic = 0; ic < len.x; ++ic){
                amrex::Real sten_cen[5] = {crse(ic,jc-1,0,n), 
                                           crse(ic-1,jc,0,n), crse(ic,jc,0,n),
                                           crse(ic+1,jc,0,n), crse(ic,jc+1,0,n)};
    
                amrex::Real summ, test, sqrmean = 0.e0;
                for(int ii = 0; ii < 5; ii++){ 
                    beta[ii] = 0.e0; 
                } 
                amrex::Real vtemp[5]; 
                amrex::Real inn;  
                for(int ii = 0; ii < 5; ii++)
                {
                    for(int jj = 0; jj < 5; jj++){
                         vtemp[jj] = V[jj][ii];
                    }
                    
                    inn = GP::inner_prod<5>(vtemp, sten_cen); 

                    sqrmean += sten_cen[ii]; 
                } 
                sqrmean /= 5; 

                sqrmean *= sqrmean; 
                test = beta[2]/sqrmean;
//                test = 101; 
                if(test > 100){
                    
                    amrex::Real sten_im[5]  = {crse(ic-1,jc-1,0,n), crse(ic-2,jc,0,n),
                                               crse(ic-1,jc,0,n), crse(ic,jc,0,n), crse(ic-1,jc+1,0,n)};
                    amrex::Real sten_jm[5]  = {crse(ic,jc-2,0,n), crse(ic-1,jc-1,0,n), 
                                               crse(ic,jc-1,0,n), crse(ic+1,jc-1,0,n), crse(ic,jc,0,n)}; 
                    amrex::Real sten_ip[5]  = {crse(ic+1,jc-1,0,n), crse(ic,jc,0,n),
                                               crse(ic+1,jc,0,n), crse(ic+2,jc,0,n), crse(ic+1,jc+1,0,n)}; 
                    amrex::Real sten_jp[5]  = {crse(ic,jc,0,n), crse(ic-1,jc+1,0,n),
                                               crse(ic,jc+1,0,n), crse(ic+1,jc+1,0,n), crse(ic,jc+2,0,n)}; 
    
                    for(int ii = 0; ii < 5; ii++)
                    {
                        for(int jj = 0; jj < 5; jj++){
                         vtemp[jj] = V[jj][ii];                       
                        } 
                        inn = GP::inner_prod<5>(vtemp, sten_jm);
                        beta[0] += 1.e0/lam[ii]*(inn*inn); 

                        inn = GP::inner_prod<5>(vtemp, sten_im); 
                        beta[1] += 1.e0/lam[ii]*(inn*inn); 

                        inn = GP::inner_prod<5>(vtemp, sten_ip);
                        beta[3] += 1.e0/lam[ii]*(inn*inn); 

                        inn = GP::inner_prod<5>(vtemp, sten_jp); 
                        beta[4] += 1.e0/lam[ii]*(inn*inn); 

                   } 
                    for(int ry = 0; ry < ratio[1]; ry++){
                        const int j = jc*ratio[1] + ry; 
                        for(int rx = 0; rx< ratio[0]; rx++){ 
                            const int i = ic*ratio[0] + rx;
                            const int id = rx + ry*ratio[0]; 
                            summ = 0.e0;

                            for(int m = 0; m < 5; ++m){
                                amrex::Real denom = (1.e-36 + beta[m])*(1.e-36 + beta[m]); 
                                ws[m] = gam[id][m]/denom; 
                                summ += ws[m]; 
                            } 
                            fine(i,j,0,n) = (ws[0]/summ)*GP::inner_prod<5>(ks[id][0], sten_jm) 
                                          + (ws[1]/summ)*GP::inner_prod<5>(ks[id][1], sten_im) 
                                          + (ws[2]/summ)*GP::inner_prod<5>(ks[id][2], sten_cen) 
                                          + (ws[3]/summ)*GP::inner_prod<5>(ks[id][3], sten_ip) 
                                          + (ws[4]/summ)*GP::inner_prod<5>(ks[id][4], sten_jp);
/*                            if(fine(i,j,0,n)<0){
                                std::cout<< fine(i,j,0,n) << std::endl;
                                std::cout<< " id = " << id << std::endl; 
                                std::cout<< " i, j " << i << '\t' << j << std::endl;  
                                std::cout<< " terms = " << std::endl; 
                                std::cout<< (ws[0]/summ)*GP::inner_prod<5>(ks[id][0], sten_jm) << std::endl;  
                                std::cout<< (ws[1]/summ)*GP::inner_prod<5>(ks[id][1], sten_im) << std::endl; 
                                std::cout<< (ws[2]/summ)*GP::inner_prod<5>(ks[id][2], sten_cen)<< std::endl;  
                                std::cout<< (ws[3]/summ)*GP::inner_prod<5>(ks[id][3], sten_ip) << std::endl; 
                                std::cout<< (ws[4]/summ)*GP::inner_prod<5>(ks[id][4], sten_jp) << std::endl; 

                                std::cout<<" beta = " << std::endl; 
                                for(int k = 0; k < 5; ++k){ 
                                    std::cout<< beta[k] << '\t';                                 
                                }
                                std::cout<<std::endl; 

                                std::cout<<" gam = " << std::endl; 
                                for(int k = 0; k < 5; ++k){ 
                                    std::cout<< gam[id][k] << '\t';                                 
                                }
                                std::cout<<std::endl; 
                                std::cout << "sum = " << summ << std::endl;
                                std::cout << std::endl;  
                                std::cout<< "wm = " << std::endl; 
                                for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ws[k] << '\t';                                 
                                }
                                amrex::Real sum1 = 0e0; 
                                for(int k = 0; k < 5; ++k) sum1 += ws[k]/summ; 
                                std::cout<< " sum(ws) = " << sum1 << std::endl ;  
                                std::cout<< "ws = " << std::endl; 
                                for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ws[k]/summ << '\t';                                 
                                }
                                std::cout<<std::endl; 
                                std::cout<< "k*" << std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ks[id][0][k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 
                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ks[id][1][k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ks[id][2][k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ks[id][3][k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< ks[id][4][k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                std::cout<< "stencil " << std::endl; 
                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< sten_jm[k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< sten_im[k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< sten_cen[k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< sten_ip[k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 

                                 for(int k = 0; k < 5; ++k){ 
                                    std::cout<< sten_jp[k] << '\t';                                 
                                 }
                                 std::cout<<std::endl; 
                                std::cin.get();
                            }
//                            std::cin.get();                          */
                        }
                    }
                }
                else{
                    for(int ry = 0; ry < ratio[1]; ry++){
                        const int j = jc*ratio[1] + ry; 
                        for(int rx = 0; rx< ratio[0]; rx++){ 
                            const int i = ic*ratio[0] + rx; 
                            const int id = rx + ry*ratio[1]; 
/*                std::cout<< " id " <<  id << " i "  << i << " j " << j << " ic " << ic << " jc " << jc  << std::endl;
                 std::cin.get(); // */
                            fine(i,j,0,n) = GP::inner_prod<5>(ks[id][2], sten_cen);
                        }
                    }
                }
            }
        }
    }
}


void
CellGaussianProcess::interp (const FArrayBox& crse,
                             int              crse_comp,
                             FArrayBox&       fine,
                             int              fine_comp,
                             int              ncomp,
                             const Box&       fine_region,
                             const IntVect&   ratio,
                             const Geometry&  crse_geom,
                             const Geometry&  fine_geom,
                             Vector<BCRec>&    bcr,
                             int              actual_comp,
                             int              actual_state)
{
    BL_PROFILE("CellGaussianProcess::interp()");
    BL_ASSERT(bcr.size() >= ncomp);
    //
    // Make box which is intersection of fine_region and domain of fine.
    //
    Box target_fine_region = fine_region & fine.box();
    FArrayBox const* crsep = &crse; 
    auto finearr = fine.array();  
//    Gpu::LaunchSafeGuard lg(Gpu::isGpuPtr(crsep) && Gpu::isGpuPtr(fine.fabPtr())); 
    
    const Box& crse_region = CoarseBox(fine_region,ratio);
    Box cb = crse.box(); 
    const Box& cb1 = amrex::grow(crse_region,-2);
    const Box& fb  = cb.refine(ratio); 
    FArrayBox ftemp(fb, ncomp); 
    auto fparr = ftemp.array(); 
    FArrayBox *fp = &ftemp; 

    const amrex::Real *dx = crse_geom.CellSize();
    GP gp;  
    gp.l = 12*std::sqrt(dx[0]*dx[0] + dx[1]*dx[1]);
    gp.InitGP(ratio[0], ratio[1], dx); 
    Vector<int> bc = GetBCArray(bcr); //Assess if we need this. 

    AMREX_LAUNCH_HOST_DEVICE_LAMBDA (cb1, tbx,{
        amrex_gpinterp(tbx, *fp, fine_comp, ncomp, *crsep, crse_comp,
                      ratio, gp.ks, gp.V, gp.lam, gp.gam);
    });

    AMREX_PARALLEL_FOR_4D (target_fine_region, ncomp, i, j, k, n, {
        finearr(i,j,k,n) = fparr(i,j,k,n); 
    }); 
}

}

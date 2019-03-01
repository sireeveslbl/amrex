
#include <climits>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_INTERP_F.H>
#include <AMReX_Interp_C.H>

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
CellGaussianProcess::CellGaussianProcess (bool mult_sample)
{
    do_multi_sampled = mult_sample;
}

CellGaussianProcess::~CellGaussianProcess () {}

Box
CellGaussianProcess::CoarseBox (const Box&     fine,
                          const IntVect& ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
}

Box
CellGaussianProcess::CoarseBox (const Box& fine,
                          int        ratio)
{
    Box crse = amrex::coarsen(fine,ratio);
    crse.grow(1);
    return crse;
} 

//Perfroms Cholesky Decomposition on covariance matrix K
template<size_t n> 
void
CellGaussianProcess::CholeskyDecomp(amrex::Real K[n][n])
{
     for(int i = 0; i < n; ++i){
        for( int j = 0; j < i-1; ++j){
            K[i][i] -= K[i][j]*K[i][j]; 
        }
        K[i][i] = sqrt(K[i][i]); 
        for(int j = i+1; j < n; ++j){
            for(int k = 0; k < j; ++j){
                K[i][j] -= K[i][k]*K[j][k]; 
            }
        K[i][j] /= K[j][j]; 
        }
    }
}

//Performs Cholesky Backsubstitution
template<size_t n> 
void 
CellGaussianProcess::cholesky(amrex::Real const kstar[n], amrex::Real const K[n][n], 
                              amrex::Real &ks[n])
{
    /* Forward sub Ly = b */ 
    for(int i = 0; i < n; ++i){
        ks[i] = 0.0e0; 
        for(int j = 0; j < i; ++j) ks[i] += kstar[j]*K[i][j]; 
        ks[i] /= K[i][i]; 
    }
    /* Back sub Ux = y */ 
    for(int i = n-1; i >= 0; --i){
        for(int j = i; j < n; ++j) ks[i] -= K[j][i]*ks[j]; 
        ks[i] /= K[i][i]; 
    }
}

inline amrex::Real
CellGaussianProcess::sqrexp(const amrex::Real x[2],const amrex::Real y[2], const amrex::Real dx[2], const amrex::Real l)
{
    amrex::Real result = std::exp(-0.5*((x[0] - y[0])*(x[0] - y[0])*dx[0]*dx[0] + 
                                        (x[1] - y[1])*(x[1] - y[1])*dx[1]*dx[1])/(l*l));
    return result;    
} 

//Builds the Covariance matrix K if uninitialized --> if(!init) GetK, weights etc.
//Four K totals to make the gammas.  
void
CellGaussianProcess::GetK(amrex::Real &K[5][5], amrex::Real &Ktot[13][13],
                          const amrex::Real dx*, const amrex::Real l)
{

    int pnt[5][2];
/*    pnt[0][0] = 0 , pnt[0][1] = -1; 
    pnt[1][0] = -1, pnt[1][1] = 0; 
    pnt[2][0] = 0 , pnt[2][1] = 0; 
    pnt[3][0] = 1 , pnt[3][1] = 0; 
    pnt[4][0] = 0 , pnt[4][1] = 1; */ 
    pnt[0] = { 0, -1}; 
    pnt[1] = {-1,  0}; 
    pnt[2] = { 0,  0}; 
    pnt[3] = { 1,  0}; 
    pnt[4] = { 0,  1}; 

    for(int i = 0; i < 5; ++i) K[i][i] = 1.e0; 
//Small K
    for(int i = 1; i < 5; ++i)
        for(int j = i; j < 5; ++j){
            K[i][j] = sqrexp(pnt[i], pnt[j], dx, l); 
        }

    for(int i = 0; i < 13; ++i) Ktot[i][i] = 1.e0; 

    amrex::Real spnt[13][2]; 
    spnt[0]  = { 0, -2}; 
    spnt[1]  = {-1, -1}; 
    spnt[2]  = { 0, -1};
    spnt[3]  = { 1, -1}; 
    spnt[4]  = {-2,  0}; 
    spnt[5]  = {-1,  0}; 
    spnt[6]  = { 0,  0}; 
    spnt[7]  = { 1,  0}; 
    spnt[8]  = { 2,  0}; 
    spnt[9]  = {-1,  1}; 
    spnt[10] = { 0,  1}; 
    spnt[11] = { 1,  1}; 
    spnt[12] = { 0,  2}; 

    for(int i = 1; i < 13; ++i)
        for(int j = i; j <13; ++j){
            Ktot[i][j] = sqrexp(spnt[i], spnt[j], dx, l); 
        }
}

//Use a Cholesky Decomposition to solve for k*K^-1 
//Inputs: K, outputs w = k*K^-1. 
//We need weights for each stencil. Therefore we'll have 5 arrays of 16 X 5 each. 

void 
CellGaussianProcess::GetKs(amrex::Real ks[16][5][5], amrex::Real amrex::Real const K[5][5], amrex::Real const *dx,
                           const amrex::Real l)
{

    //Locations of new points relative to i,j 
    amrex::Real pnt[16][2]; 
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

    int spnt[5][2]; 
    spnt[0][0] = 0 , spnt[0][1] = -1; 
    spnt[1][0] = -1, spnt[1][1] = 0; 
    spnt[2][0] = 0 , spnt[2][1] = 0; 
    spnt[3][0] = 1 , spnt[3][1] = 0; 
    spnt[4][0] = 0 , spnt[4][1] = 1; 

    amrex::Real k1[16][5], k2[16][5], k3[16][5], k4[16][5], k5[16][5]; 
    amrex::Real temp[2]; 
    amrex::Real temp2[5]; 
    //Build covariance vector between interpolant points and stencil 
     for(int i = 0; i < 16; ++i){
        for(int j = i; j < 5; ++j){
            temp = {spnt[j][0], spnt[j][1] - 1.0}; 
            k1[i][j] = sqrexp(pnt[i], temp, dx, l);

            temp = {spnt[j][0] - 1.0,  spnt[j][1]};
            k2[i][j] = sqrexp(pnt[i], temp, dx, l);

            k3[i][j] = sqrexp(pnt[i], spnt[j], dx, l);
    
            temp = {spnt[j][0] + 1.0, spnt[j][1]};
            k4[i][j] = sqrexp(pnt[i], temp, dx, l); 

            temp = {spnt[j][0], spnt[j][1] + 1.0}; 
            k5[i][j] = sqrexp(pnt[i], temp, dx, l); 
        }
        cholesky<5>(k1[i], K, temp2); 
        for(int k = 0; k < 5; k++) ks[i][k][0] = temp2[k];
        cholesky<5>(k2[i], K, temp2); 
        for(int k = 0; k < 5; k++) ks[i][k][1] = temp2[k]; 
        cholesky<5>(k3[i], K, temp2); 
        for(int k = 0; k < 5; k++) ks[i][k][2] = temp2[k]; 
        cholesky<5>(k4[i], K, temp2); 
        for(int k = 0; k < 5; k++) ks[i][k][3] = temp2[k]; 
        cholesky<5>(k5[i], K, temp2); 
        for(int k = 0; k < 5; k++) ks[i][k][4] = temp2[k]; 
    }
}

// Here we are using Kt to get the weights for the overdetermined  
// In this case, we will have 16 new points
// Therefore, we will need 16 b =  k*^T Ktot^(-1)
// K1 is already Choleskied  
void 
CellGaussianProcess::GetKtotks(const amrex::Real K1[13][13], amrex::Real &ks[16][13], 
                               const amrex::Real *dx, const amrex::Real l)
{
    //Locations of new points relative to i,j 
    amrex::Real pnt[16][2]; 
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

    //Super K positions 
    amrex::Real spnt[13][2]; 
    spnt[0]  = { 0, -2}; 
    spnt[1]  = {-1, -1}; 
    spnt[2]  = { 0, -1};
    spnt[3]  = { 1, -1}; 
    spnt[4]  = {-2,  0}; 
    spnt[5]  = {-1,  0}; 
    spnt[6]  = { 0,  0}; 
    spnt[7]  = { 1,  0}; 
    spnt[8]  = { 2,  0}; 
    spnt[9]  = {-1,  1}; 
    spnt[10] = { 0,  1}; 
    spnt[11] = { 1,  1}; 
    spnt[12] = { 0,  2}; 
/*
    spnt[0][0] =  0, spnt[0][1] = -2; 
    spnt[1][0] = -1, spnt[1][1] = -1; 
    spnt[2][0] =  0, spnt[2][1] = -1; 
    spnt[3][0] =  1, spnt[3][1] = -1; 
    spnt[4][0] = -2, spnt[4][1] =  0; 
    spnt[5][0] = -1, spnt[5][1] =  0; 
    spnt[6][0] =  0, spnt[6][1] =  0; 
    spnt[7][0] =  1, spnt[7][1] =  0; 
    spnt[8][0] = -1, spnt[8][1] =  1;
    spnt[9][0] =  0, spnt[9][1] =  1;
*/ 

    amrex::Real temp[13];        
    for(int i = 0; i < 16; i++){
       for (int j = 0; j < 13; j++){
            temp[j] = sqrexp(pnt[i], spnt[j], dx, l); 
       }
       cholesky<13>(temp, K1, ks[i]); 
    } 
}

//Solves Ux = b where U is upper triangular
template<size_t n> 
void Ux_solve(const amrex::Real R[n][n], amrex::Real &x[n], const amrex::Real b[n])
{
        for(int k = n-1; k>=0; --k){
            x[k] = b[k]; 
            for( i = k+1; i < n; i++) x[k] -= x[i]*(U[i][k]); 
            x[k] /= U[i][i]; 
        }
}

// QR Decomposition routines! 
// In qr_decomp A is decomposed into R and V contains the Householder Vectors to construct Q. 
template <size_t n> 
void
CellGaussianProcess::qr_decomp(const amrex::Real A[n][n], amrex::Real &R[n][n], 
                                     amrex::Real &v[n][n])
{
    amrex::Real s, anorm, vnorm, innerprod; 
    for(int j = 0; j < n; j++){
        anorm = 0.e0; 
        vnorm = 0.e0;

        for(int i = j; i < n; i++) anorm += A[i][j]*A[i][j]; 
        anorm = std::sqrt(anorm); 

        s = std::copysign(1.e0, A[j][j])*anorm; 
        for(int i = j; i < n; i++){
            v[i][j] = A[i][j]; 
        }
        v[j][j] += s;

        for(int i = 0; i < n; i++) vnorm += v[i][j]*v[i][j]; 
        vnorm = std::sqrt(vnorm); 
        
        for(int i =0; i < n; i++) v[i][j] /= vnorm; 

        for(k = 0; k < n; k++){
            innerprod = 0.e0; 
            for(int i = 0; i < n; i++) innerprod += v[i][j]*A[i][k]; 
            
            for(int i = 0; i < n; i++) R[i][k] = A[i][k] - 2.e0*v[i][j]*innerprod; 
        }        
    }
}

//QR decomp for the non-square matrix 
void 
CellGaussianProcess::QR(const amrex::Real A[13][5], amrex::Real &R[5][5], amrex::Real &Q[13][13])
{
    //Q = I 
    amrex::Real v[13] = {}; 
    amrex::Real norm, inner, s; 
    for(int i = 0; i < 13; ++i){
        for(int j = 0; j < 13; ++j)
            Q[i][j] = 0.e0; 
        Q[i][i] = 1.e0; 
    }

    for(int j = 0; j < 5; ++j){
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
        if(std::abs(norm) > std::numeric_limits::<double>epsilon())
        {
            for(int i = j; i < 13; ++i) v[i] /= norm; //Normalize vector v
            for(int k = j; k < 5; ++k){
                inner = 0.e0; 
                for(int i = 0; i < 13; ++i) inner += v[i]*A[i][k]; 
                for(int i = 0; i < 13; ++i) A[i][k] -= 2.e0*inner*v[i];
            }
            for(int k = 0; k < 13; ++k){
                inner = 0.e0; 
                for(int i = 0; i < 13; ++i) inner += v[i]*Q[i][k]; 
                for(int i = 0; i < 13; ++i) Q[i][k] -= 2.e0*inner*v[i]; 
            }
        }
    } 
}

//Applies V onto A -> QA 
template <size_t rows> 
void 
CellGaussianProcess::q_appl(amrex::Real (&A)[rows][rows], 
                            amrex::Real (&V)[rows][rows])
{
    for(int i = 0; i < rows; ++i) 
        q_appl_vec(A[i], V); 
}

template <size_t rows> 
void 
CellGaussianProcess::q_appl_vec(amrex::Real (&A)[rows], 
                                amrex::Real (&V)[rows][rows])
{
    for(int i = 0; i < rows; ++i) 
        {
            amrex::Real temp = 0; 
            for(int j = 0; j < rows; ++j)
                temp += A[i]*V[j][i]; 
            for(int j = 0; j < rows; ++j) 
                A[j] -= 2*V[j][i]*temp; 
        }
}


//Each point will have its
//own set of gammas. 
//Use x = R^-1Q'b 
void
CellGaussianProcess::GetGamma(amrex::Real const ks[5][5],
                              amrex::Real const Kt[13], 
                              amrex::Real &gam[5])
{

//Extended matrix Each column contains the vector of coviarances corresponding 
//to each sample (weno-like stencil)
    amrex::Real A[13][5] = {{ks[0][0], 0.e0    , 0.e0    , 0.e0    , 0.e0    }, 
                            {0.e0    , ks[0][1], 0.e0    , 0.e0    , 0.e0    }, 
                            {ks[1][0], 0.e0    , ks[0][2], 0.e0    , 0.e0    }, 
                            {ks[2][0], ks[1][1], 0.e0    , ks[0][3], 0.e0    }, 
                            {ks[3][0], ks[2][1], ks[1][2], 0.e0    , ks[0][4]}, 
                            {0.e0    , ks[3][1], ks[2][2], ks[1][3], 0.e0    }, 
                            {0.e0    , 0.e0    , ks[3][2], ks[2][3], ks[1][4]}, 
                            {0.e0    , 0.e0    , 0.e0    , ks[3][3], ks[2][4]}, 
                            {ks[4][0], 0.e0    , 0.e0    , 0.e0    , ks[3][4]}, 
                            {0.e0    , ks[4][1], 0.e0    , 0.e0    , 0.e0    }, 
                            {0.e0    , 0.e0    , ks[4][2], 0.e0    , 0.e0    }, 
                            {0.e0    , 0.e0    , 0.e0    , ks[4][3], 0.e0    }, 
                            {0.e0    , 0.e0    , 0.e0    , 0.e0    , ks[4][4]}};


   amrex::Real Q[13][13]; 
   amrex::Real R[5][5]; 
   QR(A, Q, R); // This one is for non-square matrices
   amrex::Real temp[5] ={0};

   //Q'*Kt 
   for(int i = 0; i < 5; i++)
      for(int j = 0; j < 13; j++)
        temp[i] += Q[j][i]*Kt[j];
    //gam = R^-1 Q'Kt 
    Ux_solve<5>(R, gam, temp);         
}


//Use Shifted QR with deflation to get eigen pairs. 
template<size_t n>
void 
CellGaussianProcess::GetEigenPairs(const amrex::Real K[n][n])
{
// lam and V are part of class definition! 

    amrex::Real p[n][n], Q[n][n]; 
    amrex::Real V_iter[n][n]; 
    hessen(K, p); // Puts K into Hessenberg Form 
    for(int j = n-1; j > 0; j--){
        amrex::Real er = 1.e0; 
        while(er> 1.e-10){
            amrex::Real mu = K[j][j]; 
#pragma unroll 
            for(int i = 0; i < n; i++)
                p[i][i] -= mu;   
            qr_decomp<n>(B, Q); 
            qr_appl<n>(B, Q); 
            if(j < n){
                qr_appl(P, q, j);                 
            }
            for(int i = 0; i < j; i++)
#pragma unroll
                for(int k = 0; k < j; k++)
                    V_iter[i][k] = Q[i][k];
            
            q_appl<n>(V,V_iter); 
#pragma unroll
            for(int i = 0; i < n; i++)
                B[i + n*i] += mu; 

            er = fabs(B[j + n*(j-1)]); 
        }
    }
//Since K is symmetric the eigenvectors are converged. 
}


void 
CellGaussianProcess::amrex_cginterp(const int i, const int j, const int k, const int n,  
                              const int rx, const int ry, 
                              amrex::Array4<const amrex::Real> const& crse, 
                              amrex::Array4<amrex::Real> const& fine)
{
#if (AMREX_SPACEDIM==2)
    amrex::Real sten_cen[5] = {crse(i,j-1,k,n)  , crse(i-1,j,k,n)  , crse(i,j,k,n)  , crse(i+1,j,k,n)  , crse(i,j+1,k,n)  };
    amrex::Real sten_im[5]  = {crse(i-1,j-1,k,n), crse(i-2,j,k,n)  , crse(i-1,j,k,n), crse(i,j,k,n)    , crse(i-1,j+1,k,n)}; 
    amrex::Real sten_jm[5]  = {crse(i,j-2,k,n)  , crse(i-1,j-1,k,n), crse(i,j-1,k,n), crse(i+1,j-1,k,n), crse(i,j,k,n)    }; 
    amrex::Real sten_ip[5]  = {crse(i+1,j-1,k,n), crse(i,j,k,n)    , crse(i+1,j,k,n), crse(i+2,j,k,n)  , crse(i+1,j+1,k,n)}; 
    amrex::Real sten_jp[5]  = {crse(i,j,k,n)    , crse(i-1,j+1,k,n), crse(i,j+1,k,n), crse(i+1,j+1,;), crse(i,j+2,k,n)  }; 
    amrex::Real beta[5] = {0}, ws[5] = {0}, summ, test, sqrmean = 0;
    amrex::Real inn;  
    int idx, idy; 
    
    for(int ii = 0; ii < 4; ii++)
    {
        inn = inner_prod(V[n], sten_jm); 
        beta[0] += 1.e0/lam[ii]*inn*inn; 

        inn = inner_prod(V[n], sten_im); 
        beta[1] += 1.e0/lam[ii]*inn*inn; 

        inn = inner_prod(V[n], sten_cen); 
        beta[2] += 1.e0/lam[ii]*inn*inn; 

        inn = inner_prod(V[n], sten_ip); 
        beta[3] += 1.e0/lam[ii]*inn*inn; 

        inn = inner_prod(V[n], sten_jp); 
        beta[4] += 1.e0/lam[ii]*inn*inn; 

        sqrmean += sten_cen[ii]; 
    } 

    sqrmean /= 5; 
    sqrmean *= sqrmean; 
    test = beta[2]/sqrmean; 
    if(test > 100){
        for(int jj = 0; jj < ry; ++jj){
            idy = j*ry + jj; 
            for(int ii = 0; ii < rx; ++ii){
                idx = i*rx + ii; 
                id = ii + jj*rx; //TODO this assumes Rx = Ry. If this is not the case, we need to rethink. 
                summ = 0.e0; 
                for(int m = 0; m < 5; ++m){
                    ws[m] = gam[id][m]/((1e-32 + beta[m])*(1e-32 + beta[m])); 
                    summ += ws[m]; 
                }
                fine(idx,idy,k,n) = (ws[0]/summ)*inner_prod(ks[id][0], sten_jm) 
                                  + (ws[1]/summ)*inner_prod(ks[id][1], sten_im) 
                                  + (ws[2]/summ)*inner_prod(ks[id][2], sten_cen) 
                                  + (ws[3]/summ)*inner_prod(ks[id][3], sten_ip) 
                                  + (ws[4]/summ)*inner_prod(ks[id][4], sten_jp);
            }
        }
    }
    else{
        for(int jj = 0; jj < ry; ++jj){
            idy = j*ry + jj; 
            for(int ii = 0; ii < rx; ++ii){
                idx = i*rx + ii; 
                id = ii + jj*rx; //TODO this assumes Rx = Ry. If this is not the case, we need to rethink. 
                fine(idx,idy,k,n) = inner_prod(ks[id][2], sten_cen);
            }
        }
    }

#endif    
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

    Box crse_bx(amrex::coarsen(target_fine_region,ratio));
    Vector<int> bc     = GetBCArray(bcr); //Assess if we need this. 

    AMREX_PARALLEL_FOR_4D(crse_bx, fine_comp, i, j, k, n, {
        amrex_cgpinterp(i,j,k,n, ratio[0], ratio[1], crse.array(), fine.array());
    }); 


}

}

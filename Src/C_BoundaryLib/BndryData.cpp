
#include <winstd.H>
#include <BndryData.H>
#include <Utility.H>
#include <LO_BCTYPES.H>
#include <ParallelDescriptor.H>

//
// Mask info required for this many cells past grid edge
//  (here, e.g. ref=4, crse stencil width=3, and let stencil slide 2 past grid edge)
//
int BndryData::NTangHalfWidth = 5;

BndryData::BndryData()
    :
    BndryRegister () {}

BndryData::BndryData (const BoxArray& _grids,
                      int             _ncomp, 
                      const Geometry& _geom)
    :
    geom(_geom)
{
    define(_grids,_ncomp,_geom);
}

void
BndryData::init (const BndryData& src)
{
    //
    // Got to save the geometric info.
    //
    geom = src.geom;
    //
    // Redefine grids and bndry array.
    //
    const int ngrd  = grids.size();

    for (OrientationIter fi; fi; ++fi)
    {
        const Orientation face = fi();

        bcond[face] = src.bcond[face];

        bcloc[face] = src.bcloc[face];

        masks[face].resize(ngrd);

        for (FabSetIter bfsi(bndry[face]); bfsi.isValid(); ++bfsi)
        {
            const int grd        = bfsi.index();
            const Mask& src_mask = src.masks[face][grd];
            Mask* m = new Mask(src_mask.box(),src_mask.nComp());
            m->copy(src_mask);
            masks[face].set(grd,m);
        }
    }
}

BndryData::~BndryData ()
{
    //
    // Masks was not allocated with PArrayManage -- manually dealloc.
    //
    clear_masks();
}

void
BndryData::clear_masks ()
{
    for (OrientationIter oitr; oitr; oitr++)
    {
        const Orientation face = oitr();

        for (int k = 0, N = masks[face].size(); k < N; k++)
        {
            if (masks[face].defined(k))
            {
                delete masks[face].remove(k);
            }
        }
    }
}

void
BndryData::define (const BoxArray& _grids,
                   int             _ncomp,
                   const Geometry& _geom)
{
    geom = _geom;

    BndryRegister::setBoxes(_grids);

    const int ngrd = grids.size();

    BL_ASSERT(ngrd > 0);

    Array<IntVect> pshifts(27);

    std::vector< std::pair<int,Box> > isects;

    for (OrientationIter fi; fi; ++fi)
    {
        const Orientation face      = fi();
        const int         coord_dir = face.coordDir();

        masks[face].resize(ngrd);
        bcloc[face].resize(ngrd);
        bcond[face].resize(ngrd);

        for (int ig = 0; ig < ngrd; ++ig)
        {
            bcond[face][ig].resize(_ncomp);
        }
        BndryRegister::define(face,IndexType::TheCellType(),0,1,0,_ncomp);
        //
        // Alloc mask and set to quad_interp value.
        //
        for (FabSetIter bfsi(bndry[face]); bfsi.isValid(); ++bfsi)
        {
            Box face_box = BoxLib::adjCell(grids[bfsi.index()], face, 1);
            //
            // Extend box in directions orthogonal to face normal.
            //
            for (int dir = 0; dir < BL_SPACEDIM; dir++)
            {
                if (dir == coord_dir) continue;
                face_box.grow(dir,NTangHalfWidth);
            }
            Mask* m = new Mask(face_box);
            m->setVal(outside_domain,0);
            Box dbox = geom.Domain() & face_box;
            m->setVal(not_covered,dbox,0);
            //
            // Now have to set as not_covered the periodic translates as well.
            //
            if (geom.isAnyPeriodic())
            {
                geom.periodicShift(geom.Domain(), face_box, pshifts);

                for (int iiv = 0, N = pshifts.size(); iiv < N; iiv++)
                {
                    m->shift(pshifts[iiv]);
                    Box target = geom.Domain() & m->box();
                    m->setVal(not_covered,target,0);
                    m->shift(-pshifts[iiv]);
                }
            }
            masks[face].set(bfsi.index(),m);
            //
            // Turn mask off on intersection with grids at this level.
            //
            isects = grids.intersections(face_box);

            for (int ii = 0, N = isects.size(); ii < N; ii++)
                m->setVal(covered, isects[ii].second, 0);
            //
            // Handle special cases if is periodic.
            //
            if (geom.isAnyPeriodic() && !geom.Domain().contains(face_box))
            {
                geom.periodicShift(geom.Domain(), face_box, pshifts);

                for (int iiv = 0, M = pshifts.size(); iiv < M; iiv++)
                {
                    m->shift(pshifts[iiv]);

                    isects = grids.intersections(m->box());

                    for (int ii = 0, N = isects.size(); ii < N; ii++)
                        m->setVal(covered, isects[ii].second, 0);

                    m->shift(-pshifts[iiv]);
                }
            }
        }
    }
}

std::ostream&
operator<< (std::ostream&    os,
            const BndryData& bd)
{
    if (ParallelDescriptor::NProcs() != 1)
	BoxLib::Abort("BndryData::operator<<(): not implemented in parallel");

    const BoxArray& grds  = bd.boxes();
    const int       ngrds = grds.size();
    const int       ncomp = bd.bcond[0][0].size();

    os << "[BndryData with " << ngrds << " grids and " << ncomp << " comps:\n";

    for (int grd = 0; grd < ngrds; grd++)
    {
        for (OrientationIter face; face; ++face)
        {
            Orientation f = face();

            os << "::: face " << (int)(f) << " of grid " << grds[grd] << "\nBC = ";

            for (int i = 0; i < ncomp; ++i)
                os << bd.bcond[f][grd][i] << ' ';

            os << " LOC = " << bd.bcloc[f][grd] << '\n';
            os << bd.masks[f][grd];
            os << bd.bndry[f][grd];
        }
        os << "------------------------------------------------" << std::endl;
    }

    return os;
}

void 
BndryData::writeOn (std::ostream& os) const
{
    int ngrds = grids.size();
    int ncomp = bcond[0][0].size();

    os << ngrds << " " << ncomp << std::endl;

    for (int grd = 0; grd < ngrds; grd++)
    {
        os << grids[grd] << std::endl;
    }

    os << geom << std::endl;

    for (int grd = 0; grd < ngrds; grd++)
    {
        for (OrientationIter face; face; ++face)
        {
            Orientation f = face();

            for (int cmp = 0; cmp < ncomp; cmp++)
                os << bcond[f][grd][cmp] << " ";
            os << std::endl;

            os << bcloc[f][grd] << std::endl;
        }
    }

    for (OrientationIter face; face; ++face)
    {
        Orientation f = face();

        for (int grd = 0; grd < ngrds; grd++)
        {
            masks[f][grd].writeOn(os);
            bndry[f][grd].writeOn(os);
        }
    }
}

void 
BndryData::readFrom (std::istream& is)
{
    int tmpNgrids, tmpNcomp;
    is >> tmpNgrids >> tmpNcomp;

    BoxArray tmpBa(tmpNgrids);
    for (int grd = 0; grd < tmpNgrids; grd++)
    {
        Box readBox;
        is >> readBox;
        tmpBa.set(grd, readBox);
    }

    Geometry tmpGeom;
    is >> tmpGeom;

    define(tmpBa, tmpNcomp, tmpGeom);

    for (int grd = 0; grd < tmpNgrids; grd++)
    {
        for (OrientationIter face; face; ++face)
        {
            Orientation f = face();

            int intBc;
            for (int cmp = 0; cmp < tmpNcomp; cmp++)
            {
                is >> intBc;
                setBoundCond(f, grd, cmp, BoundCond(intBc));
            }

            Real realBcLoc;
            is >> realBcLoc;
            setBoundLoc(f, grd, realBcLoc);
        }
    }

    for (OrientationIter face; face; ++face)
    {
        Orientation f = face();

        for (int grd = 0; grd < tmpNgrids; grd++)
        {
            masks[f][grd].readFrom(is);
            bndry[f][grd].readFrom(is);
        }
    }
}

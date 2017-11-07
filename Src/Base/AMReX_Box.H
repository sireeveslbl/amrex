
#ifndef BL_BOX_H
#define BL_BOX_H

#include <iosfwd>

#include <AMReX_ccse-mpi.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Orientation.H>
#include <AMReX_SPACE.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>

namespace amrex
{
    class Box;
    class BoxCommHelper;

    /**
    * \brief Grow Box in all directions by given amount.
  
    * NOTE: n_cell negative shrinks the Box by that number of cells.
    */
    Box grow (const Box& b, int i);

    //! Grow Box in each direction by specified amount.
    Box grow (const Box& b, const IntVect& v);

    //! Grow Box in direction idir be n_cell cells
    Box grow (const Box& b, int idir, int n_cell);

    Box growLo (const Box& b, int idir, int n_cell);
    Box growHi (const Box& b, int idir, int n_cell);

    /**
    * \brief Coarsen Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
    * hi <- hi/ratio.
    * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
    * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
    * That is, refinement of coarsened Box must contain
    * the original Box.
    */
    Box coarsen (const Box& b, int refinement_ratio);

    /**
    * \brief Coarsen Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
    * hi <- hi/ratio.
    * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
    * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
    * That is, refinement of coarsened Box must contain
    * the original Box.
    */
    Box coarsen (const Box& b, const IntVect& refinement_ratio);

    /**
    * Refine Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
    * hi <- (hi+1)*ratio - 1.
    * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
    * hi <- hi*ratio.
    */
    Box refine (const Box& b, int refinement_ratio);

    //! Return a Box with indices shifted by nzones in dir direction.
    Box shift (const Box& b, int dir, int nzones);

    /**
    * \brief Refine Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
    * hi <- (hi+1)*ratio - 1.
    * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
    * hi <- hi*ratio.
    */
    Box refine (const Box& b, const IntVect& refinement_ratio);

    /**
    * \brief Returns a Box with NODE based coordinates in direction dir
    * that encloses Box b.  NOTE: equivalent to b.convert(dir,NODE)
    * NOTE: error if b.type(dir) == NODE.
    */
    Box surroundingNodes (const Box& b, int dir);

    /**
    * \brief Returns a Box with NODE based coordinates in all
    * directions that encloses Box b.
    */
    Box surroundingNodes (const Box& b);

    //! Returns a Box with different type
    Box convert (const Box& b, const IntVect& typ);
    Box convert (const Box& b, const IndexType& typ);

    /**
    * \brief Returns a Box with CELL based coordinates in 
    * direction dir that is enclosed by b.
    * NOTE: equivalent to b.convert(dir,CELL)
    * NOTE: error if b.type(dir) == CELL.
    */
    Box enclosedCells (const Box& b, int dir);

    /**
    * \brief Returns a Box with CELL based coordinates in all
    * directions that is enclosed by b.
    */
    Box enclosedCells (const Box& b);

    /**
    * \brief Returns the edge-centered Box (in direction dir) defining
    * the low side of Box b.
    */
    Box bdryLo (const Box& b, int dir, int len=1);

    /**
    * \brief Returns the edge-centered Box (in direction dir) defining
    * the high side of Box b.
    */
    Box bdryHi (const Box& b, int dir, int len=1);

    /**
    * \brief Similar to bdryLo and bdryHi except that it operates on the
    * given face of  box b.
    */
    Box bdryNode (const Box& b, Orientation face, int len=1);

    /**
    * \brief Returns the cell centered Box of length len adjacent
    * to b on the low end along the coordinate direction dir.
    * The return Box is identical to b in the other directions.
    * The return Box and b have an empty intersection.
    * NOTE:  len >= 1
    * NOTE:  Box retval = b.adjCellLo(b,dir,len)
    * is equivalent to the following set of operations:
    * Box retval(b);
    * retval.convert(dir,Box::CELL);
    * retval.setrange(dir,retval.smallEnd(dir)-len,len);
    */
    Box adjCellLo (const Box& b, int dir, int len=1);

    //! Similar to adjCellLo but builds an adjacent Box on the high end.
    Box adjCellHi (const Box& b, int dir, int len=1);

    //! Similar to adjCellLo and adjCellHi; operates on given face.
    Box adjCell (const Box& b, Orientation face, int len=1);

    /**
    * \brief Modify Box to that of the minimum Box containing both
    * the original Box and the argument.
    * Both Boxes must have identical type.
    */
    Box minBox (const Box& b1, const Box& b2);

    Vector<int> SerializeBox(const Box &b);
    int SerializeBoxSize();
    Box UnSerializeBox(const Vector<int> &serarray);


    //! Write an ASCII representation to the ostream.
    std::ostream& operator<< (std::ostream& os, const Box& bx);

    //! Read from istream.
    std::istream& operator>> (std::istream& os, Box& bx);


/**
* \brief A Rectangular Domain on an Integer Lattice
*
* A Box is an abstraction for defining discrete regions of
* SPACEDIM indexing space.  Boxes have an IndexType, which defines
* IndexType::CELL or IndexType::NODE based points for each direction
* and a low and high INTVECT which defines the lower and upper corners
* of the Box.  Boxes can exist in positive and negative indexing space.
*
* Box is a dimension dependent class, so SPACEDIM must be 
* defined as either 1, 2, or 3 when compiling.
*/
class Box
{
    friend MPI_Datatype ParallelDescriptor::Mpi_typemap<Box>::type();
    friend class BoxCommHelper;

public:
    /*
    * \brief The default constructor.  For safety, the constructed Box is  
    * invalid and may be tested for validity with ok(). 
    * DO NOT CHANGE THIS BEHAVIOR!
    */
    Box () : smallend(1), bigend(0), btype() {}

    //! Construct cell-centered type Box.
    Box (const IntVect& small, const IntVect& big)
        : smallend(small), bigend(big) {}

    //! Construct box with specified lengths.
    Box (const IntVect& small, const int* vec_len)
        : smallend(small),
          bigend(AMREX_D_DECL(small[0]+vec_len[0]-1,
                              small[1]+vec_len[1]-1,
                              small[2]+vec_len[2]-1))
        {}

    /**
    * \brief Construct Box with given type.  small and big are expected
    * to be consistent with given type.
    */
    Box (const IntVect& small, const IntVect& big, const IntVect& typ)
        : smallend(small), bigend(big), btype(typ) 
    {
        BL_ASSERT(typ.allGE(IntVect::TheZeroVector()) && typ.allLE(IntVect::TheUnitVector()));
    }

    //
    // Construct dimension specific Boxes.
    //
    Box (const IntVect& small, const IntVect& big, IndexType t)
        : smallend(small), bigend(big), btype(t)
        {}

    // dtor, copy-ctor, copy-op=, move-ctor, and move-op= are compiler generated.

    //! Get the smallend of the box.
    inline const IntVect& smallEnd () const& { return smallend; }
    const IntVect& smallEnd () && = delete;

    //! Returns the coordinate of the low end in the given direction.
    inline int smallEnd (int dir) const& { return smallend[dir]; }

    //! Get the bigend.
    inline const IntVect& bigEnd () const& { return bigend; }
    const IntVect& bigEnd () && = delete;

    //! Returns the coordinate of the high end in the given direction.
    inline int bigEnd (int dir) const { return bigend[dir]; }

    //! Returns the indexing type.
    inline IndexType ixType () const { return btype; }

    //! Returns the indexing type.
    inline IntVect type () const { return btype.ixType(); }

    //! Returns the indexing type in the specified direction.
    inline IndexType::CellIndex type (int dir) const { return btype.ixType(dir); }

    //! Return the length of the Box.
    inline IntVect size () const
    {
        return IntVect(AMREX_D_DECL(bigend[0]-smallend[0] + 1,
                                    bigend[1]-smallend[1] + 1,
                                    bigend[2]-smallend[2] + 1));
    }

    //! Return the length of the Box in given direction.
    inline int length (int dir) const { return bigend[dir] - smallend[dir] + 1; }

    //! Returns a constant pointer the array of low end coordinates. Useful for calls to FORTRAN.
    inline const int* loVect () const& { return smallend.getVect(); }
    const int* loVect () && = delete;
    //! Returns a constant pointer the array of high end coordinates. Useful for calls to FORTRAN.
    inline const int* hiVect () const& { return bigend.getVect(); }
    const int* hiVect () && = delete;

    //! Returns the coordinate normal to given face.
    inline int operator[] (Orientation face) const;

    //! Checks if it is an empty box.
    inline bool isEmpty () const { return !ok(); }

    //! Checks if it is a proper Box (including a valid type).
    inline bool ok () const { return bigend.allGE(smallend) && btype.ok(); }

    //! Returns true if argument is contained within Box.
    inline bool contains (const IntVect& p) const { return p.allGE(smallend) && p.allLE(bigend); }

    /** \brief Returns true if argument is contained within Box.
    * It is an error if the Boxes have different types.
    */
    inline bool contains (const Box& b) const
    { 
	BL_ASSERT(sameType(b)); 
	return b.smallend.allGE(smallend) && b.bigend.allLE(bigend);
    } 

    //! Returns true if argument is strictly contained within Box.
    inline bool strictly_contains (const IntVect& p) const { return p.allGT(smallend) && p.allLT(bigend); }

    /**
    * \brief Returns true if argument is strictly contained within Box.
    * It is an error if the Boxes have different types.
    */
    inline bool strictly_contains (const Box& b) const 
    { 
	BL_ASSERT(sameType(b));
	return b.smallend.allGT(smallend) && b.bigend.allLT(bigend);
    }
    /**
    * \brief Returns true if Boxes have non-null intersections.
    * It is an error if the Boxes have different types.
    */
    inline bool intersects (const Box& b) const { Box isect(*this); isect &= b; return isect.ok(); }

    /**
    * \brief Returns true is Boxes same size, ie translates of each other,.
    * It is an error if they have different types.
    */
    inline bool sameSize (const Box& b) const;

    //! Returns true if Boxes have same type.
    inline bool sameType (const Box &b) const { return btype == b.btype; }

    //! Returns true if Boxes are identical (including type).
    inline bool operator== (const Box& b) const { return smallend == b.smallend && bigend == b.bigend && b.btype == btype; }

    //! Returns true if Boxes differ (including type).
    inline bool operator!= (const Box& b) const { return !operator==(b); }

    //! Returns true if Box is cell-centered in all indexing directions.
    inline bool cellCentered () const { return !btype.any(); }

    /**
    * \brief Returns the number of points contained in the Box, else
    * abort()s if the number cannot be represented in a long.
    */
    inline long numPts () const { return AMREX_D_TERM( static_cast<long>(length(0)), 
                                                      *static_cast<long>(length(1)),
                                                      *static_cast<long>(length(2))); }

    /**
    * \brief Returns the number of points contained in the Box.
    * This is inteded for use only in diagnostic messages.
    */
    inline double d_numPts () const {
        BL_ASSERT(ok());
        return AMREX_D_TERM(double(length(0)), *double(length(1)), *double(length(2)));
    }

    /**
    * \brief Return the volume, in indexing space, of region enclosed by
    * this Box.   This is identical to numPts() for CELL centered
    * Box; otherwise, numPts() > volume().  Abort()s if the number
    * cannot be represented in a long.
    */
    inline long volume () const { return AMREX_D_TERM( static_cast<long>(length(0)-btype[0]), 
                                                      *static_cast<long>(length(1)-btype[1]),
                                                      *static_cast<long>(length(2)-btype[2])); }

    /**
    * \brief Returns length of longest side.  dir is modified to give
    * direction with longest side: 0...SPACEDIM-1. Ignores type.
    */
    inline int longside (int& dir) const {
        int maxlen = length(0);
        dir = 0;
        for (int i = 1; i < BL_SPACEDIM; i++)
        {
            if (length(i) > maxlen)
            {
                maxlen = length(i);
                dir = i;
            }
        }
        return maxlen;
    }

    //! Returns length of longest side.  Ignores type.
    inline int longside () const {
        int ignore = 0;
        return longside(ignore);
    }

    /**
    * \brief Returns length of shortest side.  dir is modified to give
    *  direction with shortest side: 0...SPACEDIM-1.  Ignores type.
    */
    inline int shortside (int& dir) const {
        int minlen = length(0);
        dir = 0;
        for (int i = 1; i < BL_SPACEDIM; i++)
        {
            if (length(i) < minlen)
            {
                minlen = length(i);
                dir = i;
            }
        }
        return minlen;
    }

    //! Returns length of shortest side.  Ignores type.
    inline int shortside () const {
        int ignore = 0;
        return shortside(ignore);
    }

    /**
    * \brief Returns offset of point from smallend; i.e.
    * index(smallend) -> 0, bigend would return numPts()-1.
    * Is used in accessing FArrayBox.
    */
    inline long index (const IntVect& v) const {
        long result = v[0]-smallend[0];
#if   BL_SPACEDIM==2
        result += length(0)*(v[1]-smallend[1]);
#elif BL_SPACEDIM==3
        result += length(0)*(v[1]-smallend[1]
                             +(v[2]-smallend[2])*length(1));
#endif
        return result;
    }

    //! Redefine the small end of the Box.
    inline Box& setSmall (const IntVect& sm) { smallend = sm; return *this; }

    //! Redefine the small end of the Box.
    inline Box& setSmall (int dir, int sm_index)  { smallend.setVal(dir,sm_index); return *this; }

    //! Redefine the big end of the Box.
    inline Box& setBig (const IntVect& bg) { bigend = bg; return *this; }

    //! Redefine the big end of the Box.
    inline Box& setBig (int dir, int bg_index) { bigend.setVal(dir,bg_index); return *this; }

    /**
    * \brief Set the entire range in a given direction, starting at
    * sm_index with length n_cells.  NOTE: This will yield an
    * illegal Box if n_cells <= 0.
    */
    inline Box& setRange (int dir, int sm_index, int n_cells = 1);

    //! Set indexing type
    inline Box& setType (const IndexType& t) { btype = t; return *this; }

    //! Shift this Box nzones indexing positions in coordinate direction dir.
    inline Box& shift (int dir, int nzones) { smallend.shift(dir,nzones); bigend.shift(dir,nzones); return *this; }

    //! Equivalent to b.shift(0,iv[0]).shift(1,iv[1]) ....
    inline Box& shift (const IntVect& iv) { smallend.shift(iv); bigend.shift(iv); return *this; }

    /**
    * \brief This member shifts the Box by "half" indicies, thereby
    * converting the Box from type CELL to NODE and visa-versa.
    * b.shiftHalf(0,1)  shifts b to the right by 1/2 cells.
    * b.shiftHalf(1,-3) shifts b in the -j direction by 3/2 cells.
    * NOTE: If num_halfs is EVEN the shift is num_halfs/2 full
    * zones and hence will not change the type.
    * This is: b.shifthalf(4) == b.shift(2).
    */
    inline  Box& shiftHalf (int dir, int num_halfs);

    //! Equivalent to b.shiftHalf(0,iv[0]).shiftHalf(1,iv[1]) ...
    inline  Box& shiftHalf (const IntVect& iv);

    /**
    * \brief Convert the Box from the current type into the
    * argument type.  This may change the Box coordinates:
    * type CELL -> NODE : increase coordinate by one on high end
    * type NODE -> CELL : reduce coordinate by one on high end
    * other type mappings make no change.
    */
    inline Box& convert (IndexType t) {
        for (int dir = 0; dir < BL_SPACEDIM; dir++)
        {
            const unsigned int typ = t[dir];
            const unsigned int bitval = btype[dir];
            const int off = typ - bitval;
            bigend.shift(dir,off);
            btype.setType(dir, (IndexType::CellIndex) typ);
        }
        return *this;
    }

    /**
    * \brief Convert the Box from the current type into the
    * argument type.  This may change the Box coordinates:
    * type CELL -> NODE : increase coordinate by one on high end
    * type NODE -> CELL : reduce coordinate by one on high end
    * other type mappings make no change.
    */
    inline Box& convert (const IntVect& typ) {
        BL_ASSERT(typ.allGE(IntVect::TheZeroVector()) && typ.allLE(IntVect::TheUnitVector()));
        IntVect shft(typ - btype.ixType());
        bigend += shft;
        btype = IndexType(typ);
        return *this;
    }

    //! Convert to NODE type in all directions.
    inline Box& surroundingNodes () {
        for (int i = 0; i < BL_SPACEDIM; ++i) {
            if ((btype[i] == 0)) {
                bigend.shift(i,1);
            }
        }
        btype.setall();
        return *this;
    }

    //! Convert to NODE type in given direction.
    inline Box& surroundingNodes (int dir) {
        if (!(btype[dir]))
        {
            bigend.shift(dir,1);
            //
            // Set dir'th bit to 1 = IndexType::NODE.
            //
            btype.set(dir);
        }
        return *this;
    }

    //! Convert to CELL type in all directions.
    inline Box& enclosedCells () {
        for (int i = 0 ; i < BL_SPACEDIM; ++i) {
            if (btype[i]) {
                bigend.shift(i,-1);
            }
        }
        btype.clear();
        return *this;
    }

    //! Convert to CELL type in given direction.
    inline Box& enclosedCells (int dir) {
        if (btype[dir])
        {
            bigend.shift(dir,-1);
            //
            // Set dir'th bit to 0 = IndexType::CELL.
            //
            btype.unset(dir);
        }
        return *this;
    }

    /**
    * \brief Return Box that is intersection of this Box
    * and argument.  The Boxes MUST be of same type.
    */
    inline Box operator& (const Box& rhs) const { Box lhs(*this); lhs &= rhs; return lhs; }

    //! Intersect this Box with its argument. The Boxes MUST be of the same type.
    inline Box& operator&= (const Box& rhs)
    {
        BL_ASSERT(sameType(rhs));
        smallend.max(rhs.smallend);
        bigend.min(rhs.bigend);
        return *this;
    }

    /// for serialization
    static inline size_t 
    linearSize()
    {
      size_t retval = 2*IntVect::linearSize();
      return retval;
    }

    /// for serialization
    inline void 
    linearOut(void* a_buffer ) const
    {
      unsigned char* buf = (unsigned char*) a_buffer;
      const IntVect& ivlo = this->smallEnd();
      const IntVect& ivhi = this->bigEnd();
      ivlo.linearOut(buf);
      buf += ivlo.linearSize();
      ivhi.linearOut(buf);
      buf += ivhi.linearSize();
    }

    /// for serialization
    inline void 
    linearIn(void* a_buffer )
    {
      unsigned char* buf = (unsigned char*) a_buffer;
      IntVect ivlo;
      IntVect ivhi;
      ivlo.linearIn(buf);
      buf += ivlo.linearSize();
      ivhi.linearIn(buf);
      buf += ivhi.linearSize();

      *this = Box(ivlo, ivhi);
    }

    /**
    * \brief Modify Box to that of the minimum Box containing both
    * the original Box and the argument.
    * Both Boxes must have identical type.
    */
    inline  Box& minBox (const Box&);

    //! Shift Box (relative) by given IntVect.
    inline Box& operator+= (const IntVect& v) { smallend += v; bigend   += v; return *this; }

    //! Shift Box (relative) by given IntVect.
    inline Box operator+ (const IntVect& v) const { Box r(*this); r += v; return r; }

    //! Shift Box (relative) by given IntVect.
    inline Box& operator-= (const IntVect& v) { smallend -= v; bigend -= v; return *this; }

    //! Shift Box (relative) by given IntVect.
    inline Box operator- (const IntVect& v) const { Box r(*this); r -= v; return r; }

    /**
    * \brief Chop the Box at the chop_pnt in the dir direction
    * returns one Box, modifies the object Box.
    * The union of the two is the original Box.
    * The modified Box is the low end, the returned Box
    * is the high end.  If type(dir) = CELL, the Boxes are disjoint
    * with the chop_pnt included in the high end (new Box).
    * It is an ERROR if chop_pnt is the low end of the orig Box.
    * If type(dir) = NODE, the chop_pnt is included in both Boxes
    * but is the only point in common.  It is also an error if the
    * chop_pnt is an end node of the Box.
    */
    inline Box chop (int dir, int chop_pnt);

    /*
    * \brief Grow Box in all directions by given amount.
    * NOTE: n_cell negative shrinks the Box by that number of cells.
    */
    inline Box& grow (int i) { smallend.diagShift(-i); bigend.diagShift(i); return *this; }

    //! Grow Box in each direction by specified amount.
    inline Box& grow (const IntVect& v) { smallend -= v; bigend += v; return *this;}

    /**
    * \brief Grow the Box on the low and high end by n_cell cells
    * in direction idir.
    */
    inline Box& grow (int idir, int n_cell) { smallend.shift(idir, -n_cell); bigend.shift(idir, n_cell); return *this; }

    /**
    * \brief Grow the Box on the low end by n_cell cells in direction idir.
    * NOTE: n_cell negative shrinks the Box by that number of cells.
    */
    inline Box& growLo (int idir, int n_cell = 1) { smallend.shift(idir, -n_cell); return *this; }

    /**
    * \brief Grow the Box on the high end by n_cell cells in
    * direction idir.  NOTE: n_cell negative shrinks the Box by that
    * number of cells.
    */
    inline Box& growHi (int idir, int n_cell = 1) { bigend.shift(idir,n_cell); return *this; }

    //! Grow in the direction of the given face.
    inline Box& grow (Orientation face, int n_cell = 1) {
        int idir = face.coordDir();
        if (face.isLow()) {
            smallend.shift(idir, -n_cell);
        } else {
            bigend.shift(idir,n_cell);
        }
        return *this;
    }

    /**
    * \brief Refine Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
    * hi <- (hi+1)*ratio - 1.
    * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
    * hi <- hi*ratio.
    */
    inline Box& refine (int refinement_ratio);

    /*
    * \brief Refine Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
    * hi <- (hi+1)*ratio - 1.
    * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
    * hi <- hi*ratio.
    */
    inline Box& refine (const IntVect& refinement_ratio);

    /**
    * \brief Coarsen Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
    * hi <- hi/ratio.
    * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
    * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
    * That is, refinement of coarsened Box must contain
    * the original Box.
    */
    inline Box& coarsen (int refinement_ratio);

    /**
    * \brief Coarsen Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
    * hi <- hi/ratio.
    * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
    * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
    * That is, refinement of coarsened Box must contain
    * the original Box.
    */
    inline Box& coarsen (const IntVect& refinement_ratio);

    /**
    * \brief Step through the rectangle.  It is a runtime error to give
    * a point not inside rectangle.  Iteration may not be efficient.
    */
    inline void next (IntVect &) const;

    /**
    * \brief This static member function returns a constant reference to 
    * an object of type Box representing the unit box in
    * BL_SPACEDIM-dimensional space.
    */
    static const Box& TheUnitBox ();

    inline bool isSquare() const;

    inline bool coarsenable(int refrat, int min_width=1) const {
        return coarsenable(IntVect(refrat),min_width);
    }

    inline bool coarsenable(const IntVect& refrat, int min_width=1) const
    {
        if (!size().allGE(refrat*min_width)) {
            return false;
        } else {
            Box testBox = *this;
            testBox.coarsen(refrat);
            testBox.refine (refrat);
            return (*this == testBox);
        }
    }

    inline void normalize ()
    {
        for (int idim=0; idim < BL_SPACEDIM; ++idim) {
            if (this->length(idim) == 0) {
                this->growHi(idim,1);
            }
        }
    }

private:
    IntVect         smallend;
    IntVect         bigend;
    IndexType       btype;
};

inline Box
Box::chop (int dir, int chop_pnt)
{
//
// Modified Box is low end, returned Box is high end.
// If CELL: chop_pnt included in high end.
// If NODE: chop_pnt included in both Boxes.
//

    //
    // Define new high end Box including chop_pnt.
    //
    IntVect sm(smallend);
    IntVect bg(bigend);
    sm.setVal(dir,chop_pnt);
    if (btype[dir])
    {
        //
        // NODE centered Box.
        //
        BL_ASSERT(chop_pnt > smallend[dir] && chop_pnt < bigend[dir]);
        //
        // Shrink original Box to just contain chop_pnt.
        //
        bigend.setVal(dir,chop_pnt);
    }
    else
    {
        //
        // CELL centered Box.
        //
        BL_ASSERT(chop_pnt > smallend[dir] && chop_pnt <= bigend[dir]);
        //
        // Shrink origional Box to one below chop_pnt.
        //
        bigend.setVal(dir,chop_pnt-1);
    }
    return Box(sm,bg,btype);
}

inline Box&
Box::shiftHalf (int dir,
                int nzones)
{
    const int nbit = (nzones<0 ? -nzones : nzones)%2;
    int nshift = nzones/2;
    //
    // Toggle btyp bit if nzones is odd.
    //
    const unsigned int bit_dir = btype[dir];
    if (nbit)
        btype.flip(dir);
    if (nzones < 0)
        nshift -= (bit_dir ? nbit : 0);
    else
        nshift += (bit_dir ? 0 : nbit);
    smallend.shift(dir,nshift);
    bigend.shift(dir,nshift);
    return *this;
}

inline Box&
Box::shiftHalf (const IntVect& nz)
{
    for (int i = 0; i < BL_SPACEDIM; i++)
        shiftHalf(i,nz[i]);
   return *this;
}

inline void
Box::next (IntVect& p) const
{
    BL_ASSERT(contains(p));

    ++p[0];

#if (BL_SPACEDIM >= 2)
    if (p[0] > bigend[0])
    {
	p[0] = smallend[0];
	++p[1];
#if (BL_SPACEDIM == 3)
	if (p[1] > bigend[1])
	{
	    p[1] = smallend[1];
	    ++p[2];
	}
#endif
    }
#endif
}

inline Box&
Box::refine (int ref_ratio)
{
    return this->refine(IntVect(AMREX_D_DECL(ref_ratio,ref_ratio,ref_ratio)));
}

inline Box&
Box::refine (const IntVect& ref_ratio)
{
    if (ref_ratio != 1) {
        IntVect shft(IntVect::TheUnitVector());
        shft -= btype.ixType();
        smallend *= ref_ratio;
        bigend += shft;
        bigend *= ref_ratio;
        bigend -= shft;
    }
    return *this;
}

inline Box&
Box::coarsen (int ref_ratio)
{
    return this->coarsen(IntVect(AMREX_D_DECL(ref_ratio,ref_ratio,ref_ratio)));
}

inline Box&
Box::coarsen (const IntVect& ref_ratio)
{
    smallend.coarsen(ref_ratio);

    if (btype.any())
    {
        IntVect off(IntVect::TheZeroVector());
        for (int dir = 0; dir < BL_SPACEDIM; dir++)
        {
            if (btype[dir])
                if (bigend[dir]%ref_ratio[dir])
                    off.setVal(dir,1);
        }
        bigend.coarsen(ref_ratio);
        bigend += off;
    }
    else
    {
        bigend.coarsen(ref_ratio);
    }

    return *this;
}

inline Box&
Box::minBox (const Box &b)
{
// BoxArray may call this with not ok boxes.  BL_ASSERT(b.ok() && ok());
    BL_ASSERT(sameType(b));
    smallend.min(b.smallend);
    bigend.max(b.bigend);
    return *this;
}

inline bool
Box::sameSize (const Box& b) const
{
    BL_ASSERT(sameType(b));
    return AMREX_D_TERM(length(0) == b.length(0),
                  && length(1)==b.length(1),
                  && length(2)==b.length(2));
}

inline int
Box::operator[] (Orientation face) const
{
    const int dir = face.coordDir();
    return face.isLow() ? smallend[dir] : bigend[dir];
}

inline Box&
Box::setRange (int dir,
               int sm_index,
               int n_cells)
{
    smallend.setVal(dir,sm_index);
    bigend.setVal(dir,sm_index+n_cells-1);
    return *this;
}

inline bool
Box::isSquare () const
{
    const IntVect& sz = this->size();
#if BL_SPACEDIM==1
    return false; // can't build a square in 1-D
#elif BL_SPACEDIM==2
    return (sz[0] == sz[1]);
#elif BL_SPACEDIM==3
    return (sz[0] == sz[1] && (sz[1] == sz[2]));
#endif
}

inline Box
grow (const Box& b, int i)
{
    Box result = b;
    result.grow(i);
    return result;
}

inline Box
grow (const Box& b, const IntVect& v)
{
    Box result = b;
    result.grow(v);
    return result;
}

inline Box
grow (const Box& b, int idir, int n_cell)
{
    Box result = b;
    result.grow(idir, n_cell);
    return result;
}

inline Box
growLo (const Box& b, int idir, int n_cell)
{
    Box result = b;
    result.growLo(idir, n_cell);
    return result;
}

inline Box
growHi (const Box& b, int idir, int n_cell)
{
    Box result = b;
    result.growHi(idir, n_cell);
    return result;
}

inline Box
coarsen (const Box& b, int ref_ratio)
{
    Box result = b;
    result.coarsen(IntVect(AMREX_D_DECL(ref_ratio,ref_ratio,ref_ratio)));
    return result;
}

inline Box
coarsen (const Box& b, const IntVect& ref_ratio)
{
    Box result = b;
    result.coarsen(ref_ratio);
    return result;
}

inline Box
refine (const Box& b, const IntVect& ref_ratio)
{
    Box result = b;
    result.refine(ref_ratio);
    return result;
}

inline Box
refine (const Box& b, int ref_ratio)
{
    Box result = b;
    result.refine(IntVect(AMREX_D_DECL(ref_ratio,ref_ratio,ref_ratio)));
    return result;
}

inline Box
shift (const Box& b, int dir, int nzones)
{
    Box result = b;
    result.shift(dir, nzones);
    return result;
}

inline Box
surroundingNodes (const Box& b, int dir)
{
    Box bx(b);
    bx.surroundingNodes(dir);
    return bx;
}

inline Box
surroundingNodes (const Box& b)
{
    Box bx(b);
    bx.surroundingNodes();
    return bx;
}

inline Box
convert (const Box& b, const IntVect& typ)
{
    Box bx(b);
    bx.convert(typ);
    return bx;
}

inline Box
convert (const Box& b, const IndexType& t)
{
    Box bx(b);
    bx.convert(t);
    return bx;
}

inline Box
enclosedCells (const Box& b, int dir)
{
    Box bx(b);
    bx.enclosedCells(dir);
    return bx;
}

inline Box enclosedCells (const Box& b)
{
    Box bx(b);
    bx.enclosedCells();
    return bx;
}

inline Box
bdryLo (const Box& b, int dir, int len)
{
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    int sm = low[dir];
    low.setVal(dir,sm-len+1);
    hi.setVal(dir,sm);
    //
    // set dir'th bit to 1 = IndexType::NODE.
    //
    IndexType typ(b.ixType());
    typ.set(dir);
    return Box(low,hi,typ);
}

inline Box
bdryHi (const Box& b, int dir, int len)
{
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    unsigned int bitval = b.type()[dir];
    int bg              = hi[dir]  + 1 - bitval%2;
    low.setVal(dir,bg);
    hi.setVal(dir,bg+len-1);
    //
    // Set dir'th bit to 1 = IndexType::NODE.
    //
    IndexType typ(b.ixType());
    typ.set(dir);
    return Box(low,hi,typ);
}

inline Box
bdryNode (const Box& b, Orientation face, int len)
{
    int dir = face.coordDir();
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    if (face.isLow())
    {
        int sm = low[dir];
	low.setVal(dir,sm-len+1);
        hi.setVal(dir,sm);
    }
    else
    {
        unsigned int bitval = b.type()[dir];
        int bg              = hi[dir]  + 1 - bitval%2;
        low.setVal(dir,bg);
        hi.setVal(dir,bg+len-1);
    }
    //
    // Set dir'th bit to 1 = IndexType::NODE.
    //
    IndexType typ(b.ixType());
    typ.set(dir);
    return Box(low,hi,typ);
}

inline Box
adjCellLo (const Box& b, int dir, int len)
{
    BL_ASSERT(len > 0);
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    int sm = low[dir];
    low.setVal(dir,sm - len);
    hi.setVal(dir,sm - 1);
    //
    // Set dir'th bit to 0 = IndexType::CELL.
    //
    IndexType typ(b.ixType());
    typ.unset(dir);
    return Box(low,hi,typ);
}

inline Box
adjCellHi (const Box& b, int dir, int len)
{
    BL_ASSERT(len > 0);
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    unsigned int bitval = b.type()[dir];
    int bg              = hi[dir]  + 1 - bitval%2;
    low.setVal(dir,bg);
    hi.setVal(dir,bg + len - 1);
    //
    // Set dir'th bit to 0 = IndexType::CELL.
    //
    IndexType typ(b.ixType());
    typ.unset(dir);
    return Box(low,hi,typ);
}

inline Box
adjCell (const Box& b, Orientation face, int len)
{
    BL_ASSERT(len > 0);
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    int dir = face.coordDir();
    if (face.isLow())
    {
        int sm = low[dir];
        low.setVal(dir,sm - len);
        hi.setVal(dir,sm - 1);
    }
    else
    {
        unsigned int bitval = b.type()[dir];
        int bg              = hi[dir]  + 1 - bitval%2;
        low.setVal(dir,bg);
        hi.setVal(dir,bg + len - 1);
    }
    //
    // Set dir'th bit to 0 = IndexType::CELL.
    //
    IndexType typ(b.ixType());
    typ.unset(dir);
    return Box(low,hi,typ);
}

inline Box
minBox (const Box& b, const Box& o)
{
    Box result = b;
    result.minBox(o);
    return result;
}

inline Vector<int> SerializeBox(const Box &b)
{
  int count(0);
  Vector<int> retArray(BL_SPACEDIM * 3);
  for(int i(0); i < BL_SPACEDIM; ++i) {
    retArray[count] = b.smallEnd(i);
    ++count;
  }
  for(int i(0); i < BL_SPACEDIM; ++i) {
    retArray[count] = b.bigEnd(i);
    ++count;
  }
  IntVect ivType(b.type());
  for(int i(0); i < BL_SPACEDIM; ++i) {
    retArray[count] = ivType[i];
    ++count;
  }
  return retArray;
}


inline int SerializeBoxSize() {
  return (BL_SPACEDIM * 3);
}


inline Box UnSerializeBox(const Vector<int> &serarray)
{
  BL_ASSERT(serarray.size() == (3 * BL_SPACEDIM));
  const int *iptr = serarray.dataPtr();
  return Box(IntVect(iptr),
             IntVect(iptr + BL_SPACEDIM),
	     IndexType(IntVect(iptr + (2 * BL_SPACEDIM))));
}

class BoxCommHelper
{
public:

    explicit BoxCommHelper (const Box& bx, int* p_ = 0);

    int* data () const& { return p; }
    int* data () && = delete;

    Box make_box () const {
	return Box(IntVect(p), IntVect(p+BL_SPACEDIM), IntVect(p+2*BL_SPACEDIM));
    }

    static int size () { return 3*BL_SPACEDIM; }

private:
    int* p;
    std::vector<int> v;
};

class BoxConverter {
public:
    virtual Box doit (const Box& fine) const = 0;
    virtual BoxConverter* clone () const = 0;
    virtual ~BoxConverter () = 0;
};

inline BoxConverter::~BoxConverter () { }

void AllGatherBoxes (Vector<Box>& bxs);

}

#endif /*BL_BOX_H*/

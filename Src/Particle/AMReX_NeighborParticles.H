#ifndef _NEIGHBORPARTICLES_H_
#define _NEIGHBORPARTICLES_H_

#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "AMReX_Particles.H"

namespace amrex {

/**

 * This is a container for particles that undergo short-range interations.
 * In addition to the normal ParticleContainer methods, each tile contains a "neighbor
 * buffer" that is filled with data corresponding to the particles within 1 neighbor cell of
 * the tile boundaries. This allows the N^2 search over each pair of particles to proceed
 * locally, instead of over the entire domain.
 *
 * Note that neighbor particles are different than "ghost" particles, which are used
 * in AMR subcycling to keep track of coarse level particles that may move on to fine
 * levels during a fine level time step.
 *
 * Note: For the neighbor particles, we don't communicate the integer components, only the
 * real data.
*/
template <int NStructReal, int NStructInt, int NNeighborReal>
class NeighborParticleContainer
    : public ParticleContainer<NStructReal, NStructInt, 0, 0>
{
    
    /**

     * This is used to keep track of which proc to send which particles to.
     */
    struct NeighborTag {
        int grid;              // the source grid of this particle
        int tile;              // the source tile of this particle
        int index;             // the source index of this particle
        int who;               // the dest proc for this particle
        int periodic_shift[3]; // whether to shift in each direction
        
        bool operator<(const NeighborTag& other) const {
            if (grid  != other.grid)  return grid  < other.grid;
            if (tile  != other.tile)  return tile  < other.tile;
            if (index != other.index) return index < other.index;
            if (who   != other.who)   return who   < other.who;
            if (periodic_shift[0] != other.periodic_shift[0]) 
                return periodic_shift[0] < other.periodic_shift[0];
            if (periodic_shift[1] != other.periodic_shift[1]) 
                return periodic_shift[1] < other.periodic_shift[1];
            if (periodic_shift[2] != other.periodic_shift[2]) 
                return periodic_shift[2] < other.periodic_shift[2];
            return false;
        }
        
        bool operator==(const NeighborTag& other) const {
            return (grid == other.grid   ) and (tile == other.tile) 
                and (index == other.index) and (who == other.who  )
                and (periodic_shift[0] == other.periodic_shift[0])
                and (periodic_shift[1] == other.periodic_shift[1])
                and (periodic_shift[2] == other.periodic_shift[2]);
        }
    };

public:

    using ParticleType = typename ParticleContainer<NStructReal,
                                                    NStructInt, 0, 0>::ParticleType;
    using MyParIter = ParIter<NStructReal, NStructInt, 0, 0>;
    using PairIndex = std::pair<int, int>;

    NeighborParticleContainer(ParGDBBase* gdb, int ncells);

    NeighborParticleContainer(const Geometry            & geom,
                              const DistributionMapping & dmap,
                              const BoxArray            & ba,
                              int                         nneighbor);

    ///
    /// This resets the particle container to use the given BoxArray
    /// and DistributionMapping
    ///
    void Regrid(const DistributionMapping &dmap, const BoxArray &ba );

    ///
    /// This builds the internal data structure used for looking up neighbors
    ///
    void BuildLevelMask(int lev, 
                        const Geometry &geom, 
                        const DistributionMapping &dmap, 
                        const BoxArray &ba );

    ///
    /// This fills the neighbor buffers for each tile with the proper data
    ///
    void fillNeighbors(int lev);

    ///
    /// This updates the neighbors with their current particle data.
    ///
    void updateNeighbors(int lev);

    ///
    /// Each tile clears its neighbors, freeing the memory
    ///
    void clearNeighbors(int lev);

    ///
    /// Build a Neighbor List for each tile
    ///
    void buildNeighborList(int lev, bool sort=false);

    Vector<ParticleType>              remote_neighbors;
    std::map<PairIndex, Vector<int> > neighbor_list;
    const size_t pdata_size = (NNeighborReal+BL_SPACEDIM)*
        sizeof(typename ParticleType::RealType);

protected:
    
    NeighborTag getNeighborTag(int lev,
                               const IntVect& neighbor_cell,
                               const BaseFab<int>& mask);

    /// Apply periodic shift to particle position, so naive distance calculation
    /// between neighbors and regular particles will be correct.
    void applyPeriodicShift(int lev, ParticleType& p, const NeighborTag& tag);

    /// Perform the MPI communication neccesary to fill neighbor buffers
    void communicateNeighborsMPI(std::map<int, Vector<char> >& neighbors_to_comm, 
                                 bool reuse_rcv_counts=false);

    /// Perform handshake to figure out how many bytes each proc should receive
    void getRcvCountsMPI(const std::map<int, Vector<char> >& send_data);
    
    /// This function returns whether two particles are close enough to 
    /// potentially interact or not. It is used to construct the neighbor
    /// list.
    virtual bool check_pair(const ParticleType& p1, const ParticleType& p2) {
        return false;
    };

    /// The number of cells out which to grab neighbors.
    int num_neighbor_cells;

    /// A mask used to look up the grids and tiles of neighboring cells.
    std::unique_ptr<FabArray<BaseFab<int> > > mask_ptr;

    /// For each of the neighbor particles, we cache which proc it should go to
    /// and where it is in the container.
    Vector<NeighborTag> neighbor_tag_cache;
    
    /// How many bytes this proc will send when in updateNeighbors
    long num_snds;

    /// How many bytes this proc will receive from other procs
    Vector<long> rcvs;

    /// The other procs this one can talk to.
    Vector<int>  neighbor_procs;
};

#include "AMReX_NeighborParticlesI.H"

}

#endif // _NEIGHBORPARTICLES_H_

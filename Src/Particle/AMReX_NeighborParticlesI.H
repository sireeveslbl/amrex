#include "AMReX_NeighborParticles.H"

#include "AMReX_Particles_F.H"

using namespace amrex;

template <int NStructReal, int NStructInt, int NNeighborReal>
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::NeighborParticleContainer(ParGDBBase* gdb, int ncells)
    : ParticleContainer<NStructReal, NStructInt, 0, 0> (gdb),
      num_neighbor_cells(ncells)
{
}


template <int NStructReal, int NStructInt, int NNeighborReal>
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::NeighborParticleContainer(const Geometry            & geom,
                            const DistributionMapping & dmap,
                            const BoxArray            & ba,
                            int                         ncells)
    : ParticleContainer<NStructReal, NStructInt, 0, 0> (geom, dmap, ba),
      num_neighbor_cells(ncells)
{
    BuildLevelMask(0, geom, dmap, ba);
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::Regrid(const DistributionMapping &dmap, const BoxArray &ba ) {
    const int lev = 0;
    this->SetParticleBoxArray(lev, ba);
    this->SetParticleDistributionMap(lev, dmap);
    BuildLevelMask(lev, this->Geom(lev), dmap, ba);
    this->Redistribute();
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::BuildLevelMask(int lev,
                 const Geometry &geom, 
                 const DistributionMapping &dmap, 
                 const BoxArray &ba ) {
    
    BL_PROFILE("NeighborParticleContainer::BuildLevelMask");
    BL_ASSERT(lev == 0);
    BL_ASSERT(dmap == this->ParticleDistributionMap(lev));
    BL_ASSERT(ba == this->ParticleBoxArray(lev));

    if (mask_ptr == nullptr ||
        ! BoxArray::SameRefs(mask_ptr->boxArray(), this->ParticleBoxArray(lev)) ||
        ! DistributionMapping::SameRefs(mask_ptr->DistributionMap(), this->ParticleDistributionMap(lev))) {        
        mask_ptr.reset(new FabArray<BaseFab<int> > (ba, dmap, 2, num_neighbor_cells));
        mask_ptr->setVal(-1, num_neighbor_cells);

#ifdef _OPENMP
#pragma omp parallel
#endif
        for (MFIter mfi(*mask_ptr, this->tile_size); mfi.isValid(); ++mfi) {
            const Box& box = mfi.tilebox();
            const int grid_id = mfi.index();
            const int tile_id = mfi.LocalTileIndex();
            mask_ptr->setVal(grid_id, box, 0, 1);
            mask_ptr->setVal(tile_id, box, 1, 1);
        }
        
        mask_ptr->FillBoundary(geom.periodicity());

        neighbor_procs.clear();
        for (MFIter mfi(*mask_ptr, this->tile_size); mfi.isValid(); ++mfi) {
            const Box& box = mfi.growntilebox();
            for (IntVect iv = box.smallEnd(); iv <= box.bigEnd(); box.next(iv)) {
                const int grid = (*mask_ptr)[mfi](iv, 0);
                const int proc = this->ParticleDistributionMap(lev)[grid];
                if (proc != ParallelDescriptor::MyProc() and grid >= 0) {
                    neighbor_procs.push_back(proc);
                }
            }
        }

        RemoveDuplicates(neighbor_procs);
    }
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void 
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::fillNeighbors(int lev) {
    
    BL_PROFILE("NeighborParticleContainer::fillNeighbors");
    BL_ASSERT(lev == 0);
    
    const int MyProc = ParallelDescriptor::MyProc();
    const DistributionMapping& dmap = this->ParticleDistributionMap(lev);
    const Periodicity& periodicity  = this->Geom(lev).periodicity();
    const Box& domain = this->Geom(lev).Domain();
    const IntVect& lo = domain.smallEnd();
    const IntVect& hi = domain.bigEnd();
    
    remote_neighbors.clear();
    neighbor_tag_cache.clear();
    
    std::map<int, Vector<char> > neighbors_to_comm;
    
    int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    num_threads = omp_get_num_threads();
#endif
    
    // tmp data structures used for OMP reduction
    std::map<int, Vector<Vector<char> > > tmp_remote;
    Vector<Vector<NeighborTag> >          tmp_tag_cache(num_threads);
    for (int i = 0; i < static_cast<int>(neighbor_procs.size()); ++i) {
        const int neighbor_proc = neighbor_procs[i];
        tmp_remote[neighbor_proc].resize(num_threads);
        neighbors_to_comm[neighbor_proc];
    }
    
    // First pass - each thread collects the neighbor particles it will send
    // into a private buffer
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        Vector<NeighborTag> tags;
        tags.reserve(AMREX_D_TERM(3, *3, *3));            
        for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
#ifdef _OPENMP
            int thread_num = omp_get_thread_num();
#else
            int thread_num = 0;
#endif
            const int& grid = pti.index();
            const int& tile = pti.LocalTileIndex(); 
            PairIndex src_index(grid, tile);

            Box shrink_box = pti.tilebox();
            shrink_box.grow(-num_neighbor_cells);
            
            auto& particles = pti.GetArrayOfStructs();
            for (unsigned i = 0; i < pti.numParticles(); ++i) {
                const ParticleType& p = particles[i];
                const IntVect& iv = this->Index(p, lev);
                const int nc = num_neighbor_cells;
                
                // if the particle is more than one cell away from 
                // the tile boundary, it's not anybody's neighbor
                if (shrink_box.contains(iv)) continue;
                
                // Figure out all our neighbors, removing duplicates
                AMREX_D_TERM(
                for (int ii = -nc; ii < nc + 1; ii += nc) {,
                    for (int jj = -nc; jj < nc + 1; jj += nc) {,
                        for (int kk = -nc; kk < nc + 1; kk += nc) {)
                            if (AMREX_D_TERM((ii == 0), and (jj == 0), and (kk == 0))) continue;
                            IntVect shift(AMREX_D_DECL(ii, jj, kk));
                            IntVect neighbor_cell = iv + shift;

                            if ((*mask_ptr)[pti](neighbor_cell, 0) < 0) continue;

                            NeighborTag tag;
                            tag.grid  = grid;
                            tag.tile  = tile;
                            tag.index = i;
                            tag.who   = dmap[(*mask_ptr)[pti](neighbor_cell, 0)];

                            if (tag.who == MyProc) continue;

                            for (int dim = 0; dim < 3; ++dim)
                                tag.periodic_shift[dim] = 0;

                            if (periodicity.isAnyPeriodic()) {    
                                for (int dir = 0; dir < BL_SPACEDIM; ++dir) {
                                    if (not periodicity.isPeriodic(dir)) continue;
                                    if (neighbor_cell[dir] < lo[dir]) {
                                        tag.periodic_shift[dir] = -1;
                                    } else if (neighbor_cell[dir] > hi[dir]) {
                                        tag.periodic_shift[dir] =  1;
                                    }
                                }
                            }

                            tags.push_back(tag);
                AMREX_D_TERM(
                         },
                    },
                })
                    
                RemoveDuplicates(tags);

                for (int j = 0; j < static_cast<int>(tags.size()); ++j) {
                    const NeighborTag& tag = tags[j];

                    ParticleType particle = p; // yes make a copy
                    applyPeriodicShift(lev, particle, tag);
                        
                    tmp_tag_cache[thread_num].push_back(tag);

                    Vector<char>& buffer = tmp_remote[tag.who][thread_num];
                    size_t old_size = buffer.size();
                    size_t new_size = buffer.size() + pdata_size;
                    buffer.resize(new_size);
                    std::memcpy(&buffer[old_size], &particle, pdata_size);
                }
                tags.clear();
            }
        }
    }

    // collect the tag cache across all threads
    for (int i = 0; i < num_threads; ++i) {
        neighbor_tag_cache.insert(neighbor_tag_cache.begin(),
                                  tmp_tag_cache[i].begin(), 
                                  tmp_tag_cache[i].end());
        tmp_tag_cache[i].erase(tmp_tag_cache[i].begin(), 
                               tmp_tag_cache[i].end());        
    }

    // collect the neighbors owed from all threads
    typename std::map<int, Vector<Vector<char> > >::iterator it;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single nowait
#endif
    for (it=tmp_remote.begin(); it != tmp_remote.end(); it++) {
#ifdef _OPENMP
#pragma omp task firstprivate(it)
#endif
        {
            int who = it->first;
            for (int i = 0; i < num_threads; ++i) {
                neighbors_to_comm[who].insert(neighbors_to_comm[who].begin(),
                                              tmp_remote[who][i].begin(), 
                                              tmp_remote[who][i].end());
                tmp_remote[who][i].erase(tmp_remote[who][i].begin(), 
                                         tmp_remote[who][i].end());
            }
        }
    }

    // remove any empty map entries from neighbors_to_comm
    for (auto pmap_it = neighbors_to_comm.begin(); pmap_it != neighbors_to_comm.end(); /* no ++ */)
    {
        if (pmap_it->second.empty()) {
            neighbors_to_comm.erase(pmap_it++);
        }
        else {
            ++pmap_it;
        }
    }

    communicateNeighborsMPI(neighbors_to_comm);
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void 
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::updateNeighbors(int lev) {

    BL_PROFILE("NeighborParticleContainer::updateNeighbors");
    BL_ASSERT(lev == 0);

    const int MyProc = ParallelDescriptor::MyProc();
 
    remote_neighbors.clear();
    
    std::map<int, Vector<char> > neighbors_to_comm;

    int num_threads = 1;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single
    num_threads = omp_get_num_threads();
#endif

    // tmp data structures used for OMP reduction
    std::map<int, Vector<Vector<char> > > tmp_remote;
    for (int i = 0; i < static_cast<int>(neighbor_procs.size()); ++i) {
        const int neighbor_proc = neighbor_procs[i];
        tmp_remote[neighbor_proc].resize(num_threads);
        neighbors_to_comm[neighbor_proc];
    }

    // First pass - each thread collects the neighbor particles it will send
    // into a private buffer
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < static_cast<int>(neighbor_tag_cache.size()); ++i) {
#ifdef _OPENMP
        int thread_num = omp_get_thread_num();
#else
        int thread_num = 0;
#endif
        const NeighborTag& tag = neighbor_tag_cache[i];
        auto& particles = this->ParticlesAt(lev, tag.grid, tag.tile).GetArrayOfStructs();
        BL_ASSERT(tag.grid >= 0 and tag.who != MyProc);

        ParticleType particle = particles[tag.index]; // yes make a copy
        applyPeriodicShift(lev, particle, tag);

        Vector<char>& buffer = tmp_remote[tag.who][thread_num];
        size_t old_size = buffer.size();
        size_t new_size = buffer.size() + pdata_size;
        buffer.resize(new_size);
        std::memcpy(&buffer[old_size], &particle, pdata_size);
    }

    // collect the neighbors owed from all threads
    typename std::map<int, Vector<Vector<char> > >::iterator it;
#ifdef _OPENMP
#pragma omp parallel
#pragma omp single nowait
#endif
    for (it=tmp_remote.begin(); it != tmp_remote.end(); it++) {
#ifdef _OPENMP
#pragma omp task firstprivate(it)
#endif
        {
            int who = it->first;
            for (int i = 0; i < num_threads; ++i) {
                neighbors_to_comm[who].insert(neighbors_to_comm[who].begin(),
                                              tmp_remote[who][i].begin(), 
                                              tmp_remote[who][i].end());
                tmp_remote[who][i].erase(tmp_remote[who][i].begin(), 
                                         tmp_remote[who][i].end());
            }
        }
    }
    
    // remove any empty map entries from neighbors_to_comm
    for (auto pmap_it = neighbors_to_comm.begin(); pmap_it != neighbors_to_comm.end(); /* no ++ */)
    {
        if (pmap_it->second.empty()) {
            neighbors_to_comm.erase(pmap_it++);
        }
        else {
            ++pmap_it;
        }
    }

    bool reuse_rcv_counts = true;
    communicateNeighborsMPI(neighbors_to_comm, reuse_rcv_counts);
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>
::clearNeighbors(int lev) 
{

    BL_PROFILE("NeighborParticleContainer::clearNeighbors");
    BL_ASSERT(lev == 0);

    remote_neighbors.clear();
    neighbor_tag_cache.clear();
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>::
applyPeriodicShift(int lev, ParticleType& p, const NeighborTag& tag) {
    
    BL_ASSERT(lev == 0);
    
    const Periodicity& periodicity = this->Geom(lev).periodicity();
    if (not periodicity.isAnyPeriodic()) return;
    
    const RealBox& prob_domain = this->Geom(lev).ProbDomain();
    for (int dir = 0; dir < BL_SPACEDIM; ++dir) {
        if (not periodicity.isPeriodic(dir)) continue;
        if      (tag.periodic_shift[dir] == -1) p.pos(dir) += prob_domain.length(dir);
        else if (tag.periodic_shift[dir] ==  1) p.pos(dir) -= prob_domain.length(dir);
    }
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>::
getRcvCountsMPI(const std::map<int, Vector<char> >& send_data) {

#ifdef BL_USE_MPI
    const int NProcs = ParallelDescriptor::NProcs();

    BL_ASSERT(send_data.size() == neighbor_procs.size());

    // each proc figures out how many bytes it will send, and how
    // many it will receive
    Vector<long> snds(NProcs, 0);
    rcvs.resize(NProcs, 0);

    num_snds = 0;
    for (const auto& kv : send_data) {
        num_snds      += kv.second.size();
        snds[kv.first] = kv.second.size();
    }
    ParallelDescriptor::ReduceLongMax(num_snds);
    if (num_snds == 0) return;

    const int num_rcvs = neighbor_procs.size();
    Vector<MPI_Status>  stats(num_rcvs);
    Vector<MPI_Request> rreqs(num_rcvs);

    const int SeqNum = ParallelDescriptor::SeqNum();
    
    // Post receives
    for (int i = 0; i < num_rcvs; ++i) {
        const int Who = neighbor_procs[i];
        const long Cnt = 1;

        BL_ASSERT(Who >= 0 && Who < NProcs);

        rreqs[i] = ParallelDescriptor::Arecv(&rcvs[Who], Cnt, Who, SeqNum).req();
    }
    
    // Send.
    for (int i = 0; i < num_rcvs; ++i) {
        const int Who = neighbor_procs[i];
        const long Cnt = 1;

        BL_ASSERT(Who >= 0 && Who < NProcs);

        ParallelDescriptor::Send(&snds[Who], Cnt, Who, SeqNum);        
    }

    if (num_rcvs > 0) {
        BL_MPI_REQUIRE( MPI_Waitall(num_rcvs, rreqs.data(), stats.data()) );
    }

#endif // BL_USE_MPI
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>::
communicateNeighborsMPI(std::map<int, Vector<char> >& neighbors_to_comm, bool reuse_rcv_counts) {
    
    BL_PROFILE("NeighborParticleContainer::communicateNeighborsMPI");
    
#ifdef BL_USE_MPI
    const int NProcs = ParallelDescriptor::NProcs();
    
    // each proc figures out how many bytes it will send, and how
    // many it will receive
    if (!reuse_rcv_counts) getRcvCountsMPI(neighbors_to_comm);
    if (num_snds == 0) return;
    
    Vector<int> RcvProc;
    Vector<std::size_t> rOffset; // Offset (in bytes) in the receive buffer
    
    std::size_t TotRcvBytes = 0;
    for (int i = 0; i < NProcs; ++i) {
        if (rcvs[i] > 0) {
            RcvProc.push_back(i);
            rOffset.push_back(TotRcvBytes);
            TotRcvBytes += rcvs[i];
        }
    }
    
    const int nrcvs = RcvProc.size();
    Vector<MPI_Status>  stats(nrcvs);
    Vector<MPI_Request> rreqs(nrcvs);
    
    const int SeqNum = ParallelDescriptor::SeqNum();
    
    // Allocate data for rcvs as one big chunk.
    Vector<char> recvdata(TotRcvBytes);
    
    // Post receives.
    for (int i = 0; i < nrcvs; ++i) {
        const auto Who    = RcvProc[i];
        const auto offset = rOffset[i];
        const auto Cnt    = rcvs[Who];
        
        BL_ASSERT(Cnt > 0);
        BL_ASSERT(Cnt < std::numeric_limits<int>::max());
        BL_ASSERT(Who >= 0 && Who < NProcs);
        
        rreqs[i] = ParallelDescriptor::Arecv(&recvdata[offset], Cnt, Who, SeqNum).req();
    }
    
    // Send.
    for (const auto& kv : neighbors_to_comm) {
        const auto Who = kv.first;
        const auto Cnt = kv.second.size();
        
        BL_ASSERT(Cnt > 0);
        BL_ASSERT(Who >= 0 && Who < NProcs);
        BL_ASSERT(Cnt < std::numeric_limits<int>::max());
        
        ParallelDescriptor::Send(kv.second.data(), Cnt, Who, SeqNum);
    }
    
    // unpack the received data and put them into the proper neighbor buffers
    if (nrcvs > 0) {
        BL_MPI_REQUIRE( MPI_Waitall(nrcvs, rreqs.data(), stats.data()) );

        if (recvdata.size() % pdata_size != 0) {
            amrex::AllPrint() << "ParticleContainer::communicateNeighborsMPI: sizes = "
                              << recvdata.size() << ", " << pdata_size << "\n";
            amrex::Abort("ParticleContainer::communicateNeighborsMPI: How did this happen?");
        }
            
        char* pbuf = recvdata.data();
        int npart = recvdata.size() / pdata_size;
        for (int i = 0; i < npart; ++i) {           
            size_t old_size = remote_neighbors.size();
            size_t new_size = old_size + 1;
            remote_neighbors.resize(new_size);
            std::memcpy(&remote_neighbors[old_size], pbuf, pdata_size); 
            pbuf += pdata_size;
        }
    }
#endif
}

template <int NStructReal, int NStructInt, int NNeighborReal>
void
NeighborParticleContainer<NStructReal, NStructInt, NNeighborReal>::
buildNeighborList(int lev, bool sort) {
    
    BL_PROFILE("NeighborParticleContainer::buildNeighborList");
    BL_ASSERT(lev == 0);

    const int MyProc = ParallelDescriptor::MyProc();
    const DistributionMapping& dmap = this->ParticleDistributionMap(lev);
    const Periodicity& periodicity  = this->Geom(lev).periodicity();
    const Box& domain = this->Geom(lev).Domain();
    const IntVect& lo = domain.smallEnd();
    const IntVect& hi = domain.bigEnd();

    neighbor_list.clear();
    std::map<PairIndex, Vector<int> > list;
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
        PairIndex index(pti.index(), pti.LocalTileIndex());
        neighbor_list[index] = Vector<int>();
        list[index].resize(pti.numParticles(), -1);
    }
    
    // For each cell, we build linked lists storing the indices of
    // the particles belonging to it.
    FabArray<BaseFab<int> > head(this->ParticleBoxArray(lev),
                                 this->ParticleDistributionMap(lev),
                                 1, num_neighbor_cells);
    head.setVal(-1);
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
        PairIndex index(pti.index(), pti.LocalTileIndex());
        auto& particles = pti.GetArrayOfStructs();
        int Np = particles.size();        
        for (int i = 0; i < Np; ++i) {
            const ParticleType& p = particles[i];
            const IntVect& cell = this->Index(p, lev);
            list[index][i] = head[pti](cell);
            head[pti](cell) = i;
        }
    }
    head.FillBoundary(periodicity);
    
    // using these lists, we build a neighbor list containing both
    // real and ghost particles.
#ifdef _OPENMP
#pragma omp parallel
#endif
    for (MyParIter pti(*this, lev); pti.isValid(); ++pti) {
        auto& particles = pti.GetArrayOfStructs();
        auto& nl = neighbor_list[std::make_pair(pti.index(), pti.LocalTileIndex())];
        int Np = particles.size();

        int p_start_index = 0;
        for (unsigned i = 0; i < Np; ++i) {
            const ParticleType& p = particles[i];            
            int num_neighbors = 0;
            nl.push_back(0);
            const IntVect& cell = this->Index(p, lev);
            Box bx(cell, cell);
            bx.grow(num_neighbor_cells);
            bool check_remote = false;
            for (IntVect iv = bx.smallEnd(); iv <= bx.bigEnd(); bx.next(iv)) {
                const BaseFab<int>& mask = (*mask_ptr)[pti];
                const int grid = mask(iv, 0);
                if (grid < 0) continue;
                const int tile = mask(iv, 1);
                const int who = dmap[grid];
                if (who == MyProc) {
                    auto& nbors = this->ParticlesAt(lev, grid, tile).GetArrayOfStructs();
                    int j = head[pti](iv);
                    PairIndex index(grid, tile);

                    NeighborTag tag;
                    for (int dim = 0; dim < 3; ++dim)
                        tag.periodic_shift[dim] = 0;
                    
                    if (periodicity.isAnyPeriodic()) {    
                        for (int dir = 0; dir < BL_SPACEDIM; ++dir) {
                            if (not periodicity.isPeriodic(dir)) continue;
                            if (iv[dir] < lo[dir]) {
                                tag.periodic_shift[dir] = -1;
                            } else if (iv[dir] > hi[dir]) {
                                tag.periodic_shift[dir] =  1;
                            }
                        }
                    }

                    while (j >= 0) {
                        if ((i == j) and 
                            (grid == pti.index()) and
                            (tile == pti.LocalTileIndex())) {
                            j = list[index][j];
                            continue;
                        }

                        if (periodicity.isAnyPeriodic()) {
                            ParticleType particle = p; // yes make a copy
                            applyPeriodicShift(lev, particle, tag);
                            if ( check_pair(particle, nbors[j]) ) {
                                nl.push_back(grid);
                                nl.push_back(tile);
                                nl.push_back(j+1);
                                num_neighbors += 1;
                            } 
                            j = list[index][j];
                        } else {
                            if ( check_pair(p, nbors[j]) ) {
                                nl.push_back(grid);
                                nl.push_back(tile);
                                nl.push_back(j+1);
                                num_neighbors += 1;
                            }
                            j = list[index][j];
                        }
                    }
                } else {
                    check_remote = true;
                }
            }
            if (check_remote) { 
                for (int j = 0; j < static_cast<int>(remote_neighbors.size()); ++j) {
                    if ( check_pair(p, remote_neighbors[j]) ) {
                        nl.push_back(-1);
                        nl.push_back(-1);
                        nl.push_back(j+1);
                        num_neighbors += 1;
                    }
                }
            }
            nl[p_start_index] = num_neighbors;
            p_start_index += 3*num_neighbors + 1;
        }
    }
}

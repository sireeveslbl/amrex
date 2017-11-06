#include <iostream>

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Particles.H>
#include "AMReX_NeighborParticles.H"
#include "AMReX_PlotFileUtil.H"

using namespace amrex;

struct TestParams {
  int nx;
  int ny;
  int nz;
  int max_grid_size;
  int nppc;
  bool verbose;
};

class MyParticleContainer 
    : public NeighborParticleContainer<1, 0, 2>
{
 public:
    
    MyParticleContainer (const Geometry            & geom, 
                         const DistributionMapping & dmap,
                         const BoxArray            & ba,
                         int                         nn)
        : NeighborParticleContainer<1, 0, 2> (geom, dmap, ba, nn) 
        {
        }

    void printNL() {
        const int lev = 0;
        for (MyParIter pti(*this, lev, MFItInfo().SetDynamic(false)); pti.isValid(); ++pti) {
            PairIndex index(pti.index(), pti.LocalTileIndex());
            const auto& nl = neighbor_list[index];
            const int size = nl.size();
            // if (pti.index() == 3 and pti.LocalTileIndex() == 511) {
            //     int ind = 0;
            //     while (ind < size) {
            //         amrex::AllPrintToFile("out") << nl[ind] << "\n" << "\t";                    
                    
            //         for (int i = ind+1; i < ind + 3*nl[ind]+1; ++i) {
            //             amrex::AllPrintToFile("out") << nl[i] << " ";
            //         }
            //         amrex::AllPrintToFile("out") << "\n";                    
            //         ind += 3*nl[ind] + 1;
            //     }
            // }
            int ind = 0;
            while (ind < size) {
                amrex::AllPrintToFile("nl") << "(" << pti.index() << ", " << pti.LocalTileIndex() << "): " << nl[ind] << std::endl;
                ind += 3*nl[ind] + 1;
            }
        }
    }
    
private:

    inline virtual bool check_pair(const ParticleType& p1, const ParticleType& p2) final {
        return (std::abs(p1.pos(0) - p2.pos(0)) < 0.01171875) and
               (std::abs(p1.pos(1) - p2.pos(1)) < 0.01171875) and
               (std::abs(p1.pos(2) - p2.pos(2)) < 0.01171875);
    }
};

void test_neighbor_list(TestParams& parms)
{

  RealBox real_box;
  for (int n = 0; n < BL_SPACEDIM; n++) {
    real_box.setLo(n, 0.0);
    real_box.setHi(n, 1.0);
  }

  IntVect domain_lo(AMREX_D_DECL(0, 0, 0)); 
  IntVect domain_hi(AMREX_D_DECL(parms.nx - 1, parms.ny - 1, parms.nz-1)); 
  const Box domain(domain_lo, domain_hi);

  // This sets the boundary conditions to be doubly or triply periodic
  int is_per[BL_SPACEDIM];
  for (int i = 0; i < BL_SPACEDIM; i++) 
    is_per[i] = 0;
  Geometry geom(domain, &real_box, CoordSys::cartesian, is_per);

  BoxArray ba(domain);
  ba.maxSize(parms.max_grid_size);
  if (parms.verbose && ParallelDescriptor::IOProcessor()) {
    std::cout << "Number of boxes              : " << ba.size() << '\n' << '\n';
  }

  DistributionMapping dmap(ba);
  MyParticleContainer myPC(geom, dmap, ba, 1);

  MyParticleContainer::ParticleInitData pdata = {10.0};
  myPC.InitOnePerCell(0.5, 0.5, 0.5, pdata);

  myPC.Redistribute();
  myPC.fillNeighbors(0);
  myPC.buildNeighborList(0);

  for (int i = 0; i < 25; ++i) {
      myPC.updateNeighbors(0);
  }

  //  myPC.printNL();

}

int main(int argc, char* argv[])
{
  amrex::Initialize(argc,argv);
  
  ParmParse pp;  
  TestParams parms;  
  pp.get("nx", parms.nx);
  pp.get("ny", parms.ny);
  pp.get("nz", parms.nz);
  pp.get("max_grid_size", parms.max_grid_size);
  pp.get("nppc", parms.nppc);
  if (parms.nppc < 1 && ParallelDescriptor::IOProcessor())
      amrex::Abort("Must specify at least one particle per cell");
  
  parms.verbose = false;
  pp.query("verbose", parms.verbose);
  
  if (parms.verbose && ParallelDescriptor::IOProcessor()) {
      std::cout << std::endl;
      std::cout << "Number of particles per cell : ";
      std::cout << parms.nppc  << std::endl;
      std::cout << "Size of domain               : ";
      std::cout << parms.nx << " " << parms.ny << " " << parms.nz << std::endl;
  }
  
  test_neighbor_list(parms);
  
  amrex::Finalize();
}

# AMREX_HOME defines the directory in which we will find all the AMReX code.
# If you set AMREX_HOME as an environment variable, this line will be ignored
AMREX_HOME ?= ../../../

TINY_PROFILE = TRUE
# DEBUG     = TRUE
DEBUG     = FALSE
USE_MPI   = FALSE
USE_OMP   = TRUE
# USE_OMP   = FALSE
USE_DEVICE = TRUE
COMP      = gnu
DIM       = 2

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

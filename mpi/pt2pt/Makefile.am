AUTOMAKE_OPTIONS = subdir-objects

NVCC = nvcc
NVCFLAGS = -cuda -maxrregcount 32 -ccbin $(CXX)
SUFFIXES = .cu .cpp
.cu.cpp:
	$(NVCC) $(NVCFLAGS) $(INCLUDES) $(CPPFLAGS) --output-file $@ $<

pt2ptdir = $(pkglibexecdir)/mpi/pt2pt
pt2pt_PROGRAMS = osu_bibw osu_bw osu_bw_mt osu_latency osu_mbw_mr osu_multi_lat

if MPI4_LIBRARY
pt2pt_PROGRAMS += osu_mbw_mr_sessions osu_latency_sessions osu_bw_sessions osu_bibw_sessions
endif

AM_CFLAGS = -I../../util

UTILITIES = ../../util/osu_util.c ../../util/osu_util.h ../../util/osu_util_mpi.c ../../util/osu_util_mpi.h
if CUDA_KERNELS
UTILITIES += ../../util/kernel.cu
CLEANFILES = ../../util/kernel.cpp
if BUILD_USE_PGI
AM_CXXFLAGS = --nvcchost --no_preincludes
endif
endif

osu_bw_SOURCES = osu_bw.c $(UTILITIES)
osu_bw_sessions_SOURCES = osu_bw_sessions.c $(UTILITIES)
osu_bw_mt_SOURCES = osu_bw_mt.c $(UTILITIES)
osu_bibw_SOURCES = osu_bibw.c $(UTILITIES)
osu_bibw_sessions_SOURCES = osu_bibw_sessions.c $(UTILITIES)
osu_latency_SOURCES = osu_latency.c $(UTILITIES)
osu_latency_sessions_SOURCES = osu_latency_sessions.c $(UTILITIES)
osu_mbw_mr_SOURCES = osu_mbw_mr.c $(UTILITIES)
osu_mbw_mr_sessions_SOURCES = osu_mbw_mr_sessions.c $(UTILITIES)
osu_multi_lat_SOURCES = osu_multi_lat.c $(UTILITIES)
osu_latency_mt_SOURCES = osu_latency_mt.c $(UTILITIES)

if MPI2_LIBRARY
    pt2pt_PROGRAMS += osu_latency_mt 
endif

if EMBEDDED_BUILD
    AM_LDFLAGS =
    AM_CPPFLAGS = -I$(top_builddir)/../src/include \
		  -I${top_srcdir}/util \
		  -I${top_srcdir}/../src/include
if BUILD_PROFILING_LIB
    AM_LDFLAGS += $(top_builddir)/../lib/lib@PMPILIBNAME@.la
endif
    AM_LDFLAGS += $(top_builddir)/../lib/lib@MPILIBNAME@.la
endif

if OPENACC
    AM_CFLAGS += -acc -ta=tesla:nordc
    AM_CXXFLAGS = -acc -ta=tesla:nordc
endif


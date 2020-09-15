startupdir = $(pkglibexecdir)/mpi/startup
startup_PROGRAMS = osu_init osu_hello osu_comm_dup

if MPI4_LIBRARY
    startup_PROGRAMS += osu_sessions_init osu_sessions_dup
endif


if EMBEDDED_BUILD
    AM_LDFLAGS =
    AM_CPPFLAGS = -I$(top_builddir)/../src/include \
		  -I${top_srcdir}/../src/include
if BUILD_PROFILING_LIB
    AM_LDFLAGS += $(top_builddir)/../lib/lib@PMPILIBNAME@.la
endif
    AM_LDFLAGS += $(top_builddir)/../lib/lib@MPILIBNAME@.la
endif


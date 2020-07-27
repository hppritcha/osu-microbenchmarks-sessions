#define BENCHMARK "OSU MPI Init Test"
#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif
/*
 * Copyright (C) 2002-2019 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static MPI_Session lib_shandle = MPI_SESSION_NULL;
static MPI_Comm lib_comm = MPI_COMM_NULL;

int
main (int argc, char *argv[])
{
    int myid, numprocs, rc, i;
    const char pset_name[] = "mpi://world";
    struct timespec tp_before, tp_after;
    long duration = 0, min, max, avg;
    MPI_Group wgroup = MPI_GROUP_NULL;
    MPI_Comm comm_array[1000];

    rc = MPI_Session_init(MPI_INFO_NULL, MPI_ERRORS_RETURN,
                          &lib_shandle);
    if (rc != MPI_SUCCESS) {
         return -1;
    }

    /*
     * create a group from the WORLD process set
     */
    rc = MPI_Group_from_session_pset(lib_shandle,
                                     pset_name,
                                        &wgroup);
    if (rc != MPI_SUCCESS) {
        MPI_Session_finalize(&lib_shandle);
        return -1;
    }

   /*
    * get a communicator
    */
    rc = MPI_Comm_create_from_group(wgroup, "mpi.forum.example",
                                    MPI_INFO_NULL,
                                    MPI_ERRORS_RETURN,
                                    &lib_comm);
    if (rc != MPI_SUCCESS) {
        MPI_Group_free(&wgroup);
        MPI_Session_finalize(&lib_shandle);
        return -1;
    }

    clock_gettime(CLOCK_REALTIME, &tp_before);
    for (i=0;i<1000;i++)
        MPI_Comm_dup(lib_comm, &comm_array[i]);
    clock_gettime(CLOCK_REALTIME, &tp_after);
    for (i=0;i<1000;i++)
        MPI_Comm_free(&comm_array[i]);

    duration = (tp_after.tv_sec - tp_before.tv_sec) * 1e3;
    duration += (tp_after.tv_nsec - tp_before.tv_nsec) / 1e6;

    MPI_Comm_size(lib_comm, &numprocs);
    MPI_Comm_rank(lib_comm, &myid);

    MPI_Reduce(&duration, &min, 1, MPI_LONG, MPI_MIN, 0, lib_comm);
    MPI_Reduce(&duration, &max, 1, MPI_LONG, MPI_MAX, 0, lib_comm);
    MPI_Reduce(&duration, &avg, 1, MPI_LONG, MPI_SUM, 0, lib_comm);
    avg = avg/numprocs;

    if(myid == 0) {
        fprintf(stdout, HEADER);
        fprintf(stdout, "nprocs: %d, min: %ld ms, max: %ld ms, avg: %ld ms\n",
                numprocs, min, max, avg);
        fflush(stdout);
    }

    MPI_Comm_free(&lib_comm);
    MPI_Group_free(&wgroup);
    MPI_Session_finalize(&lib_shandle);

    return EXIT_SUCCESS;
}


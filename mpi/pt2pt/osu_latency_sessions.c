#define BENCHMARK "OSU MPI%s Latency Test"
/*
 * Copyright (C) 2002-2019 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University. 
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <osu_util_mpi.h>

static MPI_Session lib_shandle = MPI_SESSION_NULL;
static MPI_Comm lib_comm = MPI_COMM_NULL;


int
main (int argc, char *argv[])
{
    int myid, numprocs, i, rc;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = LAT;
    const char pset_name[] = "mpi://world";
    MPI_Flags flags = MPI_FLAG_THREAD_NONCONCURRENT_SINGLE;
    MPI_Group wgroup = MPI_GROUP_NULL;


    set_header(HEADER);
    set_benchmark_name("osu_latency");

    po_ret = process_options(argc, argv);

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

    rc = MPI_Session_init(&flags, MPI_INFO_NULL, MPI_ERRORS_RETURN,
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


    MPI_CHECK(MPI_Comm_size(lib_comm, &numprocs));
    MPI_CHECK(MPI_Comm_rank(lib_comm, &myid));

    if (0 == myid) {
        switch (po_ret) {
            case PO_CUDA_NOT_AVAIL:
                fprintf(stderr, "CUDA support not enabled.  Please recompile "
                        "benchmark with CUDA support.\n");
                break;
            case PO_OPENACC_NOT_AVAIL:
                fprintf(stderr, "OPENACC support not enabled.  Please "
                        "recompile benchmark with OPENACC support.\n");
                break;
            case PO_BAD_USAGE:
                print_bad_usage_message(myid);
                break;
            case PO_HELP_MESSAGE:
                print_help_message(myid);
                break;
            case PO_VERSION_MESSAGE:
                print_version_message(myid);
                MPI_CHECK(MPI_Finalize());
                exit(EXIT_SUCCESS);
            case PO_OKAY:
                break;
        }
    }

    switch (po_ret) {
        case PO_CUDA_NOT_AVAIL:
        case PO_OPENACC_NOT_AVAIL:
        case PO_BAD_USAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_FAILURE);
        case PO_HELP_MESSAGE:
        case PO_VERSION_MESSAGE:
            MPI_CHECK(MPI_Finalize());
            exit(EXIT_SUCCESS);
        case PO_OKAY:
            break;
    }

    if(numprocs != 2) {
        if(myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    if (allocate_memory_pt2pt(&s_buf, &r_buf, myid)) {
        /* Error allocating memory */
        MPI_CHECK(MPI_Finalize());
        exit(EXIT_FAILURE);
    }

    print_header(myid, LAT);

    
    /* Latency test */
    for(size = options.min_message_size; size <= options.max_message_size; size = (size ? size * 2 : 1)) {
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        MPI_CHECK(MPI_Barrier(lib_comm));

        if(myid == 0) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                if(i == options.skip) {
                    t_start = MPI_Wtime();
                }

                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 1, 1, lib_comm));
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 1, 1, lib_comm, &reqstat));
            }

            t_end = MPI_Wtime();
        }

        else if(myid == 1) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                MPI_CHECK(MPI_Recv(r_buf, size, MPI_CHAR, 0, 1, lib_comm, &reqstat));
                MPI_CHECK(MPI_Send(s_buf, size, MPI_CHAR, 0, 1, lib_comm));
            }
        }

        if(myid == 0) {
            double latency = (t_end - t_start) * 1e6 / (2.0 * options.iterations);

            fprintf(stdout, "%-*d%*.*f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, latency);
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);
    MPI_Comm_free(&lib_comm);
    MPI_Group_free(&wgroup);
    MPI_Session_finalize(&lib_shandle);

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}


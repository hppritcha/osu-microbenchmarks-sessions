#define BENCHMARK "OSU MPI%s Bandwidth Test"
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

int
main (int argc, char *argv[])
{
    int myid, numprocs, i, j;
    int size;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t = 0.0, time_sum = 0.0;
    int window_size = 64;
    int po_ret = 0;
    options.bench = PT2PT;
    options.subtype = BW;
    int threads_num,thread;
    char *threads_str;
    int provided;
    MPI_Comm dup_comm[24];//hardcoded for simplicity

    set_header(HEADER);
    set_benchmark_name("osu_bw");

    po_ret = process_options(argc, argv);
    window_size = options.window_size;

    if (PO_OKAY == po_ret && NONE != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }
    
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided));
    if (MPI_THREAD_MULTIPLE != provided){
            if(myid == 0) {
                fprintf(stderr, "MPI threading support is not MPI_THREAD_MULTIPLE:%d\n "
                        , provided);
            }
            MPI_Finalize();
            exit(EXIT_FAILURE);
    }
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myid));

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

    threads_str = getenv("OMP_NUM_THREADS");
    if (NULL == threads_str ||
                    (threads_num = strtol(threads_str, NULL, 0)) <= 0) {
           if(myid == 0) {
                   fprintf(stderr, "Cannot find the number of OMP threads to use. "
                       "Set OMP_NUM_THREADS accordingly .\n");
           }
           MPI_Finalize();
           exit(EXIT_FAILURE);
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

    for (i = 0; i < threads_num; i++) {
        if (0 != MPI_Comm_dup(MPI_COMM_WORLD, &dup_comm[i])) {
            fprintf(stderr, "Fail MPI_Comm_dup\n");
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

    }

    print_header(myid, BW);

    /* Bandwidth test */
    for(size = options.min_message_size; size <= options.max_message_size; size *= 2) {
        set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
        set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);


        if(size > LARGE_MESSAGE_SIZE) {
            options.iterations = options.iterations_large;
            options.skip = options.skip_large;
        }

        if(myid == 0) {
#pragma omp parallel for private(t_start, t_end, time_diff, request, reqstat) reduction(+:time_sum) num_threads(threads_num)
        for (thread = 0 ;thread < threads_num; thread ++) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                if(i == options.skip) {
                    t_start = MPI_Wtime();
                }

                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Isend(s_buf, size, MPI_CHAR, 1, 1, dup_comm[thread],
                            request + j));
                }

                MPI_CHECK(MPI_Waitall(window_size, request, reqstat));
                MPI_CHECK(MPI_Recv(r_buf, 4, MPI_CHAR, 1, 1, dup_comm[thread],
                        &reqstat[0]));
            }

            t_end = MPI_Wtime();
            t = t_end - t_start;
	    time_sum += t;
	} //for thread_num
        } //if my_id = 0

        else if(myid == 1) {
#pragma omp parallel for  private(request, reqstat) num_threads(threads_num)
       for (thread = 0 ;thread < threads_num; thread ++) {
            for(i = 0; i < options.iterations + options.skip; i++) {
                for(j = 0; j < window_size; j++) {
                    MPI_CHECK(MPI_Irecv(r_buf, size, MPI_CHAR, 0, 1, dup_comm[thread],
                            request + j));
                }

                MPI_CHECK(MPI_Waitall(window_size, request, reqstat));
                MPI_CHECK(MPI_Send(s_buf, 4, MPI_CHAR, 0, 1, dup_comm[thread]));
            }
        } //for num threads
	} // if my_id = 1

        if(myid == 0) {
            double tmp = size / 1e6 * options.iterations * window_size;

            fprintf(stdout, "%-*d%*.*f time %f\n", 10, size, FIELD_WIDTH,
                    FLOAT_PRECISION, tmp/ time_sum / threads_num, time_sum/threads_num);
            fflush(stdout);
        }
    }

    free_memory(s_buf, r_buf, myid);
    MPI_CHECK(MPI_Finalize());

    if (NONE != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}

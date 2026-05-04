/*
============================================================================
Filename    : progD.c
Author      : Jaime Oliver Pastor & Roméo Maignal
SCIPER		: 356574 & 360568
============================================================================
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "function.h"
#include "utility.h"

int main(int argc, char *argv[]) {
    int nrounds, size;

    /* Parse input arguments */
    if(argc != 3) {
        printf("Invalid input! Usage: ./progD <nrounds> <size>\n");
        return 1;
    } else {
        nrounds = atoi(argv[1]);
        size = atoi(argv[2]);
    }

    int rank, nprocs;

    /* Initialise model */
    rand_gen generator = init_rand(0);
    int *model = (int*) malloc(size * sizeof(int));
    for(int i = 0; i < size; i++) {
        model[i] = next_rand(generator) * MAX_VAL;
    }
    free_rand(generator);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int amount_per_process = size / (nprocs - 1);
    int remainder = size % (nprocs - 1);
    int amount = 0;
    if(rank != 0) {
        amount = amount_per_process;
        if (rank == nprocs - 1)
            amount += remainder;
    }

    int *chunk = (int*) malloc(amount * sizeof(int));
    int *chunk_result = (int*) calloc(size, sizeof(int));
    int *result = (int*) calloc(size, sizeof(int));

    int *sendcounts = (int*) calloc(nprocs, sizeof(int));
    int *indexes = (int*) calloc(nprocs, sizeof(int));

    if(rank == 0) {
        sendcounts[0] = 0;
        indexes[0] = 0;
        for(int process = 1; process < nprocs; process++) {
            sendcounts[process] = amount_per_process;
            if (process == nprocs - 1)
                sendcounts[process] += remainder;
            indexes[process] = (process - 1) * amount_per_process;
        }
    }

    set_clock();
    for(int round = 0; round < nrounds; round++) {
        for(int i = 0; i < size; i++) {
            chunk_result[i] = 0;
            result[i] = 0;
        }

        MPI_Scatterv(model, sendcounts, indexes, MPI_INT, chunk, amount, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0)
            compute(chunk, result, amount, size);

        MPI_Reduce(result, chunk_result, size, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
            for(int i = 0; i < size; i++)
                model[i] += chunk_result[i];
    }

    MPI_Finalize();

    if(rank == 0) {
        /* Output stats */
        double totaltime = elapsed_time();
        printf("- ProgD: Using %d procs for %d iterations on %d size: %.3gs.\n", nprocs, nrounds, size, totaltime);
        write_csv(&model, 1, size, "model.csv");
    }

    return 0;
}

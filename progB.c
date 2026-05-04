/*
============================================================================
Filename    : progB.c
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
    int nrounds, size, B1, B2;

    /* Parse input arguments */
    if(argc != 5) {
        printf("Invalid input! Usage: ./progB <nrounds> <size>\n");
        return 1;
    } else {
        nrounds = atoi(argv[1]);
        size = atoi(argv[2]);
        B1 = atoi(argv[3]);
        B2 = atoi(argv[4]);
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

    set_clock();
    for(int round = 0; round < nrounds; round++) {
        if(rank == 0) {
            int chunk_size_per_process = size / (nprocs - 1);
            int remainder = size % (nprocs - 1);

            for(int process = 1; process < nprocs; process++) {
                int chunk_size = chunk_size_per_process;
                int index = (process - 1) * chunk_size_per_process;
                if (process == nprocs - 1)
                    chunk_size += remainder;
                for(int i = 0; i < chunk_size; i += B1) {
                    int batch_size = B1;
                    if (i + batch_size > chunk_size)
                        batch_size = chunk_size - i;
                    MPI_Send(&model[index + i], batch_size, MPI_INT, process, 0, MPI_COMM_WORLD);
                }
            }

            int *buffer = (int*) malloc(B2 * sizeof(int));

            for(int process = 1; process < nprocs; process++) {
                for(int i = 0; i < size; i += B2) {
                    int recieve = B2;
                    if(i + recieve > size)
                        recieve = size - i;
                    MPI_Recv(buffer, recieve, MPI_INT, process, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for(int j = 0; j < recieve; j++)
                        model[i + j] += buffer[j];
                }
            }

            free(buffer);
        } else {
            int chunk_size_per_process = size / (nprocs - 1);
            int remainder = size % (nprocs - 1);
            int chunk_size = chunk_size_per_process;
            if (rank == nprocs - 1)
                chunk_size += remainder;
            int *chunk = (int*) malloc(chunk_size * sizeof(int));
            int *result = (int*) calloc(size, sizeof(int));

            for(int i = 0; i < chunk_size; i += B1) {
                int recieve = B1;
                if (i + recieve > chunk_size)
                    recieve = chunk_size - i;
                MPI_Recv(&chunk[i], recieve, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            compute(chunk, result, chunk_size, size);

            for(int i = 0; i < size; i += B2) {
                int batch_size = B2;
                if(i + batch_size > size)
                    batch_size = size - i;
                MPI_Send(&result[i], batch_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
            }

            free(chunk);
            free(result);
        }
    }

    MPI_Finalize();

    if(rank == 0) {
        /* Output stats */
        double totaltime = elapsed_time();
        printf("- ProgB: Using %d procs for %d iterations on %d size with batches of size B1=%d and B2=%d: %.3gs.\n", nprocs, nrounds, size, B1, B2, totaltime);
        write_csv(&model, 1, size, "model.csv");
    }

    return 0;
}

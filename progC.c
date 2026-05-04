/*
============================================================================
Filename    : progC.c
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
		printf("Invalid input! Usage: ./progC <nrounds> <size>\n");
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

	int chunk_size_per_process = size / (nprocs - 1);
	int remainder = size % (nprocs - 1);

	/* Pre-compute max number of send/recv requests needed */
	int max_chunk_size = chunk_size_per_process + remainder;
	int max_send_reqs = (max_chunk_size + B1 - 1); /* upper bound: ceil(max_chunk/B1) * (nprocs-1) */
	int max_recv_reqs = (size + B2 - 1);            /* upper bound: ceil(size/B2) * (nprocs-1) */

	set_clock();
	for(int round = 0; round < nrounds; round++) {
		if(rank == 0) {
			/* Issue all Isends to all workers before waiting */
			int total_sends = 0;
			for(int p = 1; p < nprocs; p++) {
				int cs = chunk_size_per_process;
				if (p == nprocs - 1) cs += remainder;
				total_sends += (cs + B1 - 1) / B1;
			}
			MPI_Request *send_reqs = (MPI_Request*) malloc(total_sends * sizeof(MPI_Request));
			int n = 0;
			for(int process = 1; process < nprocs; process++) {
				int chunk_size = chunk_size_per_process;
				int index = (process - 1) * chunk_size_per_process;
				if (process == nprocs - 1)
					chunk_size += remainder;
				for(int i = 0; i < chunk_size; i += B1) {
					int send = B1;
					if(i + send > chunk_size)
						send = chunk_size - i;
					MPI_Isend(&model[index + i], send, MPI_INT, process, 0, MPI_COMM_WORLD, &send_reqs[n++]);
				}
			}
			MPI_Waitall(n, send_reqs, MPI_STATUS_IGNORE);
			free(send_reqs);

			int *buffer = (int*) malloc(size * sizeof(int));

			/* Issue all Irecvs from all workers before waiting */
			int total_recvs = (nprocs - 1) * ((size + B2 - 1) / B2);
			MPI_Request *recv_reqs = (MPI_Request*) malloc(total_recvs * sizeof(MPI_Request));

			/* We need per-process buffers to avoid overwriting during concurrent receives */
			int **proc_buffers = (int**) malloc((nprocs - 1) * sizeof(int*));
			for(int p = 0; p < nprocs - 1; p++)
				proc_buffers[p] = (int*) malloc(size * sizeof(int));

			int rn = 0;
			for(int process = 1; process < nprocs; process++) {
				int *pbuf = proc_buffers[process - 1];
				for(int i = 0; i < size; i += B2) {
					int recieve = B2;
					if(i + recieve > size)
						recieve = size - i;
					MPI_Irecv(&pbuf[i], recieve, MPI_INT, process, 1, MPI_COMM_WORLD, &recv_reqs[rn++]);
				}
			}
			MPI_Waitall(rn, recv_reqs, MPI_STATUS_IGNORE);
			free(recv_reqs);

			/* Accumulate all results into model */
			for(int p = 0; p < nprocs - 1; p++) {
				for(int i = 0; i < size; i++)
					model[i] += proc_buffers[p][i];
				free(proc_buffers[p]);
			}
			free(proc_buffers);
			free(buffer);

		} else {
			int chunk_size = chunk_size_per_process;
			if (rank == nprocs - 1)
				chunk_size += remainder;
			int *chunk = (int*) malloc(chunk_size * sizeof(int));
			int *result = (int*) calloc(size, sizeof(int));

			int n_recv = (chunk_size + B1 - 1) / B1;
			MPI_Request *req_recv = (MPI_Request*) malloc(n_recv * sizeof(MPI_Request));
			int rn = 0;
			for(int i = 0; i < chunk_size; i += B1) {
				int recieve = B1;
				if(i + recieve > chunk_size)
					recieve = chunk_size - i;
				MPI_Irecv(&chunk[i], recieve, MPI_INT, 0, 0, MPI_COMM_WORLD, &req_recv[rn++]);
			}
			MPI_Waitall(rn, req_recv, MPI_STATUS_IGNORE);
			free(req_recv);

			compute(chunk, result, chunk_size, size);

			int n_send = (size + B2 - 1) / B2;
			MPI_Request *req_send = (MPI_Request*) malloc(n_send * sizeof(MPI_Request));
			int sn = 0;
			for(int i = 0; i < size; i += B2) {
				int send = B2;
				if(i + send > size)
					send = size - i;
				MPI_Isend(&result[i], send, MPI_INT, 0, 1, MPI_COMM_WORLD, &req_send[sn++]);
			}
			MPI_Waitall(sn, req_send, MPI_STATUS_IGNORE);
			free(req_send);

			free(chunk);
			free(result);
		}
	}

	MPI_Finalize();

	if(rank == 0) {
		/* Output stats */
		double totaltime = elapsed_time();
		printf("- ProgC: Using %d procs for %d iterations on %d size with batches of size B1=%d and B2=%d: %.3gs.\n", nprocs, nrounds, size, B1, B2, totaltime);
		write_csv(&model, 1, size, "model.csv");
	}

	return 0;
}

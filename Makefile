CC = mpicc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: progD progC progB progA prog0 

progD: progD.c utility.h function.h
	$(CC) $(CFLAGS) $< -o $@

progC: progC.c utility.h function.h
	$(CC) $(CFLAGS) $< -o $@

progB: progB.c utility.h function.h
	$(CC) $(CFLAGS) $< -o $@

progA: progA.c utility.h function.h
	$(CC) $(CFLAGS) $< -o $@

prog0: prog0.c utility.h function.h
	$(CC) $(CFLAGS) $< -o $@

#rmm: rmm.c utility.h
#	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f progD progC progB progA prog0 rmm function.o matC.csv

# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = gcc 
CFLAGS = -O3 -fopenmp -fomit-frame-pointer -march=armv8-a -ffast-math -mtune=tsv110
# OPT = -no-multibyte-chars

LDLIBS = -lpthread -lm -llapack -lopenblas -lrt -lcblas -I/usr/include/openblas -O -fopenmp
targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o gemm-naive.o gemm-blocked.o gemm-blas.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o gemm-naive.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o gemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o gemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)

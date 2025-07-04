#all: diffusionc diffusionf diffusionc-mpi diffusionf-mpi diffusionc-mpi-nonblocking
#
#include ../Makefile.inc
#
MPICC=mpicc
MPIF77=mpif90
#
#diffusionf: diffusion.f90
#	${F77} ${FFLAGS} -o $@ $< ${USEPGPLOT} ${PGPLIBS}
#
diffusionc: diffusion.c
	${CC} ${CFLAGS} -g -c -o diffusionc.o $< ${USEPGPLOT} ${PGPLIBS}
	${MPIF77} ${FFLAGS} -g -o $@ diffusionc.o ${USEPGPLOT} ${PGPLIBS}
#
#diffusionf-mpi: diffusionf-mpi.f90
#	${MPIF77} ${FFLAGS} -o $@ $< ${PGPLIBS}
#
diffusionc-mpi: diffusionc-mpi.c 
	${MPICC} ${CFLAGS} -c -o diffusionc-mpi.o $< ${PGPLIBS}
	${MPIF77} ${FFLAGS} -o $@ diffusionc-mpi.o ${USEPGPLOT} ${PGPLIBS}

diffusionc-mpi-nonblocking: diffusionc-mpi-nonblocking.c
	${MPICC} ${CFLAGS} -c -o diffusionc-mpi-nonblocking.o $< ${PGPLIBS}
	${MPIF77} ${FFLAGS} -o $@ diffusionc-mpi-nonblocking.o ${USEPGPLOT} ${PGPLIBS}

clean:
	rm -rf diffusionf
	rm -rf diffusionc
	rm -rf diffusionf-mpi
	rm -rf diffusionc-mpi
	rm -rf diffusionc-mpi-nonblocking
	rm -rf *.o
	rm -rf *~

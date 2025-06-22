#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

#ifdef PGPLOT
#include "cpgplot.h"
#endif

void calc_temperature(float **temperature, int i, int old, int new, 
		              float dt, float kappa, float dx) {
    temperature[new][i] = temperature[old][i] + dt*kappa/(dx*dx) *
        (temperature[old][i+1] - 2.*temperature[old][i] + 
         temperature[old][i-1]) ;
}

int main(int argc, char **argv) {
    /* simulation parameters */
    const int totpoints=1000;
    const float xleft = -12., xright = +12.;
    const float kappa = 1.;

    const int nsteps=100000;

#ifdef PGPLOT
    const int plotsteps=50;
    int red, grey,white;
#endif

    /* data structures */
    float *x;
    float **temperature;
    float *theory;

    /* parameters of the original temperature distribution */
    const float ao=1., sigmao=1.;
    float a, sigma;

    float fixedlefttemp, fixedrighttemp;

    int old, new;
    int step, i;
    float time;
    float dt, dx;
    float l_error, g_error;

	// Set up MPI
	MPI_Init(&argc, &argv);
    int rank, size;
	MPI_Request request1 = MPI_REQUEST_NULL;
	MPI_Request request2 = MPI_REQUEST_NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);    	

    /* set parameters */

    dx = (xright-xleft)/(totpoints-1);
    dt = dx*dx * kappa/10.;

    /*
     * allocate data, including ghost cells: old and new timestep
     * theory doesn't need ghost cells, but we include it for simplicity
     */

	// Setup local arraysizes, so that they span the whole domain
	int l_arraysize = (totpoints + 2) / size;
	int arraysize_diff = totpoints + 2 - l_arraysize * size;
	if (rank < arraysize_diff) {
		l_arraysize +=1;
	}

	// Starting index for each process assuming equal distribution of array sizes
	int l_starting_idx = rank * l_arraysize;
	// Displace the index for later processes in case of unequal distribution of sizes
	if (rank >= arraysize_diff) l_starting_idx += arraysize_diff;

	// Add padding: 1 cells from each of neighbouring cells for d/dx calculations.
	if (rank != 0 && rank != size-1) {
		l_arraysize += 2; // For processes of neither first nor last rank
	} else if ( size != 0 ) {
		l_arraysize += 1; // For first or last rank processes when run on several tasks
	}
	
	if (rank != 0) {
	    l_starting_idx -= 1; // Adjust starting index to include padding
	}

    theory = (float *)malloc((l_arraysize)*sizeof(float));
    x      = (float *)malloc((l_arraysize)*sizeof(float));
    temperature = (float **)malloc(2 * sizeof(float *));
    temperature[0] = (float *)malloc((l_arraysize)*sizeof(float));
    temperature[1] = (float *)malloc((l_arraysize)*sizeof(float));
    old = 0;
    new = 1;

	printf("Rank %d process got %d cells allocated, with starting index at %d\n", rank, l_arraysize, l_starting_idx);

    /* setup initial conditions */

    time = 0.;
    for (i=0; i<l_arraysize; i++) {
        x[i] = xleft + (l_starting_idx + i-1+0.5)*dx;
        temperature[old][i] = ao*exp(-(x[i]*x[i]) / (2.*sigmao*sigmao));
        theory[i]           = ao*exp(-(x[i]*x[i]) / (2.*sigmao*sigmao));
    }
    fixedlefttemp = ao*exp(-(xleft-0.5*dx)*(xleft-0.5*dx) / (2.*sigmao*sigmao));
    fixedrighttemp= ao*exp(-(xright+0.5*dx)*(xright+0.5*dx)/(2.*sigmao*sigmao));

#ifdef PGPLOT
    cpgbeg(0, "/xwindow", 1, 1);
    cpgask(0);
    cpgenv(xleft, xright, 0., 1.5*ao, 0, 0);
    cpglab("x", "Temperature", "Diffusion Test");
    red = 2;  cpgscr(red,1.,0.,0.);
    grey = 3; cpgscr(grey,.2,.2,.2);
    white=4;cpgscr(white,1.0,1.0,1.0);

    cpgsls(1); cpgslw(1); cpgsci(grey);
    cpgline(totpoints+2, x, theory);
    cpgsls(2); cpgslw(3); cpgsci(red);
    cpgline(totpoints+2, x, temperature[old]);
#endif

    /* evolve */
    for (step=0; step < nsteps; step++) {
        /* boundary conditions: keep endpoint temperatures fixed. */
		if (rank == 0) {
		    temperature[old][0] = fixedlefttemp;
		}
		if (rank == size - 1) {
            temperature[old][l_arraysize - 1] = fixedrighttemp;
		}
		MPI_Barrier(MPI_COMM_WORLD);

        // Exchange ghost cells
        if (rank > 0) {
            MPI_Isendrecv(&temperature[old][1], 1, MPI_FLOAT, rank-1, 0,
                         &temperature[old][0], 1, MPI_FLOAT, rank-1, 1,
                         MPI_COMM_WORLD, &request1);
        }
        if (rank < size-1) {
            MPI_Isendrecv(&temperature[old][l_arraysize-2], 1, MPI_FLOAT, rank+1, 1,
                         &temperature[old][l_arraysize-1], 1, MPI_FLOAT, rank+1, 0,
                         MPI_COMM_WORLD, &request2);
        }

        for (i=2; i<l_arraysize - 2; i++) {
            calc_temperature(temperature, i, old, new, dt, kappa, dx);
        }

		MPI_Wait(&request1, MPI_STATUS_IGNORE);
        calc_temperature(temperature, 1, old, new, dt, kappa, dx);

		MPI_Wait(&request2, MPI_STATUS_IGNORE);
        calc_temperature(temperature, l_arraysize-2, old, new, dt, kappa, dx);


        time += dt;

#ifdef PGPLOT
        if (step % plotsteps == 0) {
            cpgbbuf();
            cpgeras();
            cpgsls(2); cpgslw(12); cpgsci(red);
            cpgline(totpoints+2, x, temperature[new]);
        }
#endif

        /* update correct solution */

        sigma = sqrt(2.*kappa*time + sigmao*sigmao);
        a = ao*sigmao/sigma;
        for (i=0; i<l_arraysize; i++) {
            theory[i] = a*exp(-(x[i]*x[i]) / (2.*sigma*sigma));
        }

#ifdef PGPLOT
        if (step % plotsteps == 0) {
            cpgsls(1); cpgslw(6); cpgsci(white);
            cpgline(totpoints+2, x, theory);
            cpgebuf();
        }
#endif
		// Error calculation
        l_error = 0.;
        for (i=1;i<l_arraysize-1;i++) {
            l_error += (theory[i] - temperature[new][i])*(theory[i] - temperature[new][i]);
        }
        l_error = sqrt(l_error);
        MPI_Reduce(&l_error, &g_error, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

		if (rank == 0) {
		    printf("Step = %d, Time = %g, Error = %g\n", step, time, g_error);
		}

        old = new;
        new = 1 - old;
		MPI_Barrier(MPI_COMM_WORLD);
    }


    /*
     * free data
     */

	free(temperature[1]);
	free(temperature[0]);
	free(temperature);
	free(x);
	free(theory);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize();

    return 0;
}

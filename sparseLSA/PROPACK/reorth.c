/* Stephen Becker, 3/7/09
 * Re-writing the fortran reorth.c
 * so that I can compile it on computers without
 * a fortran compiler
 *
 * This calls some BLAS functions
 * There is a different naming convention on windows,
 * so when compiling this on windows, define the symbol "WINDOWS"
 * (In the mex compiler, this can be done with the -DWINDOWS flag)
 *
 * See reorth.f for details on calling this
 * reorth.f written by R.M. Larsen for PROPACK
 *
 * Not working!!!
 *
 * */

#include <stdio.h>
#ifdef WINDOWS
  extern void dgemv(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA,
          double *X, int *INCX, double *BETA, double *Y, int *INCY);
  extern double dnrm2(int *N, double *X, int *INCX);
#else
  extern void dgemv_(char *TRANS, int *M, int *N, double *ALPHA, double *A, int *LDA,
          double *X, int *INCX, double *BETA, double *Y, int *INCY);
  extern double dnrm2_(int *N, double *X, int *INCX);
#endif




/* Modified Gram Schmidt re-orthogonalization */
void MGS( int *n, int *k, double *V, int *ldv, double *vnew, double *index ) {
    int i, j, idx;
    int LDV = *ldv;
    double s;
    for (i=0;i<*k;i++){

        idx = (int) index[i]-1;  /* -1 because Fortran uses 1-based indices */
        s = 0.0;
        
        for (j=0;j<*n;j++)
            
            /*s += V[j,idx]*vnew[j]; */  /* Fortran is row-indexed */
            s += V[ idx*LDV + j ] * vnew[j];

        for (j=0;j<*n;j++)
        
            /* vnew[j] -= s*V[j,idx]; */ /* Fortran is row-indexed */
            vnew[j] -= s*V[ idx*LDV + j ];

    }
}



  

void reorth( int *n, int *k, double *V, int *ldv, double *vnew, double *normv, double *index,
        double *alpha, double *work, int *iflag, int *nre ) {

    int i, one = 1;
    char Transpose = 'T', Normal = 'N';
    double *normv_old;
    const int MAXTRY = 4;

    double oneD = 1.0;
    double nOneD = -1.0;
    double zero = 0.0;

#ifdef WINDOWS
    void (*dgemvPtr)(char *, int *, int *, double *, double *, int *,
          double *, int *, double *, double *, int *) = dgemv;
    double (*dnrm2Ptr)(int *, double *, int *) = dnrm2;
#else
    void (*dgemvPtr)(char *, int *, int *, double *, double *, int *,
          double *, int *, double *, double *, int *) = dgemv_;
    double (*dnrm2Ptr)(int *, double *, int *) = dnrm2_;
#endif

    /* Hack: if index != 1:k, we do MGS to avoid reshuffling */
    if ( *iflag == 1 ) {
        for ( i=0; i< *k; i++ ){
            if ( index[i] != (i+1) ) {
                *iflag = 0;
                break;
            }
        }
    }
    *normv_old = 0;
    *nre = 0;
    *normv = dnrm2Ptr( n, vnew, &one );


    while ( ( *normv < *alpha* *normv_old) || ( *nre == 0 ) ) {
        if ( *iflag == 1 ) {
            /* CGS */
            dgemvPtr(&Transpose, n, k, &oneD, V, ldv, vnew, &one, &zero, work, &one);
            dgemvPtr(&Normal, n, k, &nOneD, V, ldv, work, &one, &oneD, vnew, &one);
        } else {
            /* MGS */
            MGS( n, k, V, ldv, vnew, index );
        }
        *normv_old = *normv;
        *normv = dnrm2Ptr( n, vnew, &one );
        *nre = *nre + 1;

        if ( *nre > MAXTRY ) {
            /* vnew is numerically in span(V) --> return vnew as all zeros */
            *normv = 0.0;
            for ( i=0; i< *n ; i++ )
                vnew[i] = 0.0;
            return;
        }

    }
}



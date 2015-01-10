/* 
 * Stephen Becker, 11/10/08
 * Computes A(omega), where A = U*V' is never explicitly computed 
 * srbecker@caltech.edu
 *
 * Partial Changelog:
 *
 *  3/24/09
 *      adding in "mxIsComplex" and "mxGetPi" routines to handle complex data seamlessly
 *      Not fully implemented yet, though; need to define .' vs ' convention (i.e. transpose vs adjoint)
 *  5/13/09
 *      Finished implementing the complex stuff.
 *      The convention is A = U*V', not A = U*V.'
 *      Having problems with complex data on 64-bit linux though: zdot isn't working
 *      see
 *      http://matlab.izmiran.ru/help/techdoc/matlab_external/ch04cr17.html
 *      and the fort.h file in this directory, for alternative methods
 *  5/19/09
 *      try mex -largeArrayDims -lblas -DWINDOWS XonOmega.c  ??
 *      it only affects zdotu, not the level-3 blas
 *      tried -fPIC (as opposed to -fpic) in LDFLAGS, no luck
 *      So, my current solution is to implement the blas function myself for 64-bit processors
 *
 * 
 * Note to the user:
 *      to really take advantage of this function, you should recompile it on
 *      your computer so that you can take advantage of any good BLAS libraries you might have
 *      However, it is unlikely that this will be the speed bottleneck for most applications...
 *
 *      There are 3 ways to call this program:
 *
 *      (1) b = XonOmega(U,V,omega)
 *              implicity forms the matrix A = U*V', then returns b = A(omega)
 *              where omega are linear indices
 *
 *      (2) b = XonOmega(U,V,omegaX,omegaY)
 *              does the same, but omegaX and omegaY are subscripts
 *              (i.e. omega = ind2sub(... OmegaX, OmegaY), or [OmegaX,OmegaY] = sub2ind(...,omega) )
 *
 *      (3) b = XonOmega(U,V,Y)
 *              gets information about omega from the sparsity pattern of the sparse matrix Y
 *              This is probably a bit slower that the other methods
 *
 * */

#include "mex.h"
/* #include "limits.h"  in mex.h already */

/* for Windows, BLAS functions have no underscore, but for linux/unix/darwin, they do 
 * So, for Windows, either uncomment the line that says #define WINDOWs, or,
 * when compiling, use mex -L(location) -lmwblas -DWINDOWS  for Windows 
 *  ( where (location) is fullfile(matlabroot,'extern','lib','win32',cc)
 *    where cc is usually 'lcc' (default) or 'microsoft' (for visual studio)
 *    
 * )
 * For linux, etc., you can specifically undefine the WINDOWS symbol
 * by passing in the -UWINDOWS option to mex, but that's not
 * necessary unless you uncommented the #define WINDOWS line below.
 * You can use the mathwork's blas, -lmwblas, or, recommended,
 * use a good blas that's already installed, and probably available
 * with just -lblas.  So, try this:
 *    mex -O -lblas updateSparse.c
 * the -O is for optimized
 * the -lblas may actually not even be necessary
 */

/* #define WINDOWS */

/* to use "mwSize", need matrix.h, but doesn't work for R2006a */
#include "matrix.h" 
/* So, use the following definitions instead: */
#ifndef mwSize
    #define mwSize size_t
#endif
#ifndef mwIndex
    #define mwIndex size_t  /* should make it compatible w/ 64-bit systems */
#endif

#ifndef true
    #define true 1
#endif
#ifndef false
    #define false 0
#endif

typedef struct{ double re; double im; } complex16; /* a hack */

#ifdef WINDOWS
  /* make a level-1 BLAS call */
  extern double ddot( int *K, double *x, int *incx, double *y, int *incy );
  /* if omega = 1:M*N, then use a level-3 BLAS call, so include dgemm */
  extern void dgemm( char *transA, char *transB, int *M, int *N, int *K, 
          double *alpha, double *A, int *LDA, double *R, int *LDB, 
          double *beta, double *C, int *LDC );
  extern void zgemm( char *transA, char *transB, int *M, int *N, int *K,
          complex16 *alpha, complex16 *A, int *LDA, complex16 *R, int *LDB, 
          complex16 *beta, complex16 *C, int *LDC );
  extern complex16 zdotu( int *K, complex16 *x, int *incx, complex16 *y, int *incy );
#else
  extern double ddot_( int *K, double *x, int *incx, double *y, int *incy );
  extern void dgemm_( char *transA, char *transB, int *M, int *N, int *K,
          double *alpha, double *A, int *LDA, double *R, int *LDB, 
          double *beta, double *C, int *LDC );
  extern void zgemm_( char *transA, char *transB, int *M, int *N, int *K,
          complex16 *alpha, complex16 *A, int *LDA, complex16 *R, int *LDB, 
          complex16 *beta, complex16 *C, int *LDC );
  extern complex16 zdotu_( int *K, complex16 *x, int *incx, complex16 *y, int *incy );
#endif


void printUsage() {
    mexPrintf("XonOmega.c: usage is\n\t b = XonOmega(U,V,omega)\n");
    mexPrintf("where A = U*V' and b = A(omega)\n");
    mexPrintf("\nAlternative usage is\n\t b = XonOmega(U,V,omegaI,omegaJ),\n where [omegaI,omegaJ] = ind2sub(...,omega)\n");
    mexPrintf("\nAnother alternative usage is\n\t b = XonOmega(U,V, OMEGA),\n where OMEGA is a sparse matrix with nonzeros on omega.\nThis will agree with the other forms of the command if omega is sorted\n\n");
    mexPrintf("If U and V are complex, then make sure A=U*V' and not A=U*V.'\n\n");
}

void mexFunction(
         int nlhs,       mxArray *plhs[],
         int nrhs, const mxArray *prhs[]
         )
{

/* deal with the underscores now so that they can be ignored after this point */
#ifdef WINDOWS
    double (*my_ddot)( int *, double *, int *, double *, int *) = ddot;
    complex16 (*my_zdot)( int *, complex16 *, int *, complex16 *, int *) = zdotu;
    void (*my_dgemm)(char *, char *, int *, int *, int *, double *, double *,
            int *, double *, int *, double *, double *, int *) = dgemm;
    void (*my_zgemm)(char *, char *, int *, int *, int *, complex16 *, complex16 *,
        int *, complex16 *, int *, complex16 *, complex16 *, int *) = zgemm;
#else
    double (*my_ddot)( int *, double *, int *, double *, int *) = ddot_;
    complex16 (*my_zdot)( int *, complex16 *, int *, complex16 *, int *) = zdotu_;
    void (*my_dgemm)(char *, char *, int *, int *, int *, double *, double *,
            int *, double *, int *, double *, double *, int *) = dgemm_;
    void (*my_zgemm)(char *, char *, int *, int *, int *, complex16 *, complex16 *,
            int *, complex16 *, int *, complex16 *, complex16 *, int *) = zgemm_;
#endif

    mwSize M, N, K, K2;
    mwSize nOmega1, nOmega2, nOmega;
    mwIndex i,j,k,m;
    double *U, *Vt, *output, *omega;
    double *U_imag, *Vt_imag, *output_imag; /* for complex data */
    complex16 *U_cplx, *Vt_cplx, temp_cplx; /* for complex data */
    double *omegaX, *omegaY;
    mwIndex *omegaI, *omegaJ;
    int SPARSE = false;
    int COMPLEX = false;
    int USE_BLAS = false;
    int LARGE_BIT = false;
    complex16 alpha_cplx, beta_cplx;
    complex16 *output_cplx;


    char transA = 'N', transB = 'T';
    mwSize LDA, LDB;
    double alpha, beta;
    
    /* Check for proper number of input and output arguments */    
    if ( (nrhs < 3) || (nrhs > 4) ) {
        printUsage();
    mexErrMsgTxt("Three (or four) input argument required.");
    } 
    if(nlhs > 1){
        printUsage();
    mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument  */
    if (!(mxIsDouble(prhs[0])) || !((mxIsDouble(prhs[1]))) ){
        printUsage();
    mexErrMsgTxt("Input arguments wrong data-type (must be doubles).");
    }   

    /* Get the size and pointers to input data */
    M  = mxGetM(prhs[0]);
    K  = mxGetN(prhs[0]);
    N  = mxGetM(prhs[1]);
    K2  = mxGetN(prhs[1]);
    if ( K != K2 ) {
        printUsage();
        mexErrMsgTxt("Inner dimension of U and V' must agree.");
    }
    COMPLEX = (( (mxIsComplex(prhs[0])) ) || (mxIsComplex(prhs[1])) );
    nOmega1 = mxGetM( prhs[2] );
    nOmega2 = mxGetN( prhs[2] );


    /* on 64-bit systems, these may be longs, but I really want
     * them to be ints so that they work with the BLAS calls */
    mxAssert(M<INT_MAX,"Matrix is too large for 32-bit FORTRAN");
    mxAssert(N<INT_MAX,"Matrix is too large for 32-bit FORTRAN");
    mxAssert(K<INT_MAX,"Matrix is too large for 32-bit FORTRAN");
    
    
    if ( (nOmega1 != 1) && (nOmega2 != 1) ) {
/*         printUsage(); */
/*         mexErrMsgTxt("Omega must be a vector"); */
        /* Update:
         * if this happens, we assume Omega is really a sparse matrix
         * and everything is OK */
        if ( ( nOmega1 != M ) || ( nOmega2 != N ) || (!mxIsSparse( prhs[2] )) ){
            printUsage();
            mexErrMsgTxt("Omega must be a vector or a sparse matrix");
        }
        nOmega = mxGetNzmax( prhs[2] );
        SPARSE = true;
    } else {
        nOmega = nOmega1 < nOmega2 ? nOmega2 : nOmega1;
        if ( nOmega > N*M ) {
            printUsage();
            mexErrMsgTxt("Omega must have M*N or fewer entries");
        }
    }

    U = mxGetPr(prhs[0]);
    Vt = mxGetPr(prhs[1]);
    if (COMPLEX) {
        U_imag = mxGetPi(prhs[0]);
        Vt_imag = mxGetPi(prhs[1]);
        plhs[0] = mxCreateDoubleMatrix(nOmega, 1, mxCOMPLEX);
        output = mxGetPr(plhs[0]);
        output_imag = mxGetPi(plhs[0]);

        U_cplx = (complex16 *)malloc( M*K*sizeof(complex16) );
        Vt_cplx = (complex16 *)malloc( N*K*sizeof(complex16) );

        for ( i=0 ; i < M*K ; i++ ){
            U_cplx[i].re = (double)U[i];
            U_cplx[i].im = (double)U_imag[i];
        }
        for ( i=0 ; i < N*K ; i++ ){
            Vt_cplx[i].re = (double)Vt[i];
            Vt_cplx[i].im = -(double)Vt_imag[i]; 
            /* minus sign, since adjoint, not transpose */
        }
        
    } else {
        plhs[0] = mxCreateDoubleMatrix(nOmega, 1, mxREAL);
        /* mxCreateDoubleMatrix automatically zeros out the entries */
        output = mxGetPr(plhs[0]);
    }

    /* 
     * We have 4 "modes":
     * Modes 1 and 2 work when "omega" is a vector if linear indices
     *      If omega is in fact all linear indices, then
     *      we really want all of U*V', so we'll call a level-3 BLAS
     *      function (dgemm) for efficiency (call this "mode 2"),
     *      but otherwise we'll use a "for" loop (call this "mode 1");
     * Mode 3
     *      if omega is given only by the nonzero elements of an input 
     *      sparse matrix Y
     * Mode 4
     *      if omega is given as a set of subscripts, i.e. omegaX, omegaY
     *      then the "for" loop is slightly different
     *      
     */


    /* determine if we're on a 64-bit processor */
    LARGE_BIT = ( sizeof( size_t ) > 4 );

    if (( nrhs < 4 ) && (!SPARSE) ){
        /* ----------- MODES 1 and 2 ---------------------------- */
         /* by default, make output the same shape (i.e. row- or column-
          * vector) as the input "omega" */
        mxSetM( plhs[0], mxGetM( prhs[2] ) );
        mxSetN( plhs[0], mxGetN( prhs[2] ) );
        
        
        
        /* omega is a vector of linear indices */
        USE_BLAS = false;
        omega = mxGetPr(prhs[2]);
        if ( nOmega == (M*N) ) {
            /* in this case, we want to use level-3 BLAS, unless
             * omega isn't sorted (in which case, BLAS would give wrong
             * answer, since it assumes omega is sorted and in column-major
             * order.  So, find out if omega is sorted: */
            USE_BLAS = true;
            for ( k = 0 ; k < nOmega-1 ; k++ ) {
                if (omega[k] > omega[k+1] ) {
                    USE_BLAS = false;
                    break;
                }
            }
        }
        /* USE_BLAS refers to whether I use level-3 BLAS or not;
         * if not, then I use level-1 BLAS */

        if ( !USE_BLAS ) {
            /* ----------- MODE 1 ------------------------------- */
            
            if ( (COMPLEX) && (LARGE_BIT) ) {
              for ( k=0 ; k < nOmega ; k++ ){
                i = (mwIndex) ( (mwIndex)(omega[k]-1) % M);
                j = (mwIndex) ( (mwIndex)(omega[k]-1)/ M);
                
                /* The following zdot call doesn't work for me on my 64-bit linux system */
                /*  This was a type: N, instead of i
                temp_cplx = my_zdot( (int*) &K, U_cplx+N, (int*)&M, Vt_cplx+j, (int*)&N ); 
                temp_cplx = my_zdot( (int*) &K, U_cplx+i, (int*)&M, Vt_cplx+j, (int*)&N ); 
                */

                /* So, implement the BLAS call myself
                 * ZDOTU(N,ZX,INCX,ZY,INCY)  
                 * */
                temp_cplx.re = 0.0;
                temp_cplx.im = 0.0;
                for ( m=0 ; m < K ; m++ ){
                    temp_cplx.re += U_cplx[i+m*M].re * Vt_cplx[j+m*N].re
                        - U_cplx[i+m*M].im * Vt_cplx[j+m*N].im;
                    temp_cplx.im += U_cplx[i+m*M].im * Vt_cplx[j+m*N].re
                        + U_cplx[i+m*M].re * Vt_cplx[j+m*N].im;
                }
                output[k] = temp_cplx.re;
                output_imag[k] = temp_cplx.im;
                
              }
            } else if (COMPLEX) {
              for ( k=0 ; k < nOmega ; k++ ){
                i = (mwIndex) ( (mwIndex)(omega[k]-1) % M);
                j = (mwIndex) ( (mwIndex)(omega[k]-1)/ M);
                temp_cplx = my_zdot( (int*) &K, U_cplx+i, (int*)&M, Vt_cplx+j, (int*)&N ); 
                output[k] = temp_cplx.re;
                output_imag[k] = temp_cplx.im;
              }

            } else {
              for ( k=0 ; k < nOmega ; k++ ){
                /* don't forget that Matlab is 1-indexed, C is 0-indexed */
                i = (mwIndex) ( (mwIndex)(omega[k]-1) % M);
                j = (mwIndex) ( (mwIndex)(omega[k]-1)/ M);
    /*             mexPrintf("%2d %2d\n",i+1,j+1); */
                output[k] = my_ddot( (int*)&K, U+i, (int*)&M, Vt+j, (int*)&N );
              }
            }
        } else {

            /* ----------- MODE 2 ------------------------------- */
            /* here, we have no problem with complex numbers and blas on 64-bit systems */

            /* we need to compute A itself, so use level-3 BLAS */
            transA = 'N';
            transB = 'T';
            LDA = M;
            LDB = N;
            
            if (COMPLEX) {
                alpha_cplx.re = 1.0;  alpha_cplx.im = 0.0;
                beta_cplx.re = 0.0;   beta_cplx.im  = 0.0;
                /* need to make a new complex data structure */
                output_cplx = (complex16 *) malloc( M*N*sizeof(complex16) );
                my_zgemm(&transA,&transB,(int*)&M,(int*)&N,(int*)&K,
                    &alpha_cplx,U_cplx,(int*)&LDA,Vt_cplx,(int*)&LDB,&beta_cplx,output_cplx,(int *)&M );
                /* now, copy the data to the output */
                for ( i=0 ; i < M*N ; i++ ) {
                    output[i] = output_cplx[i].re;
                    output_imag[i] = output_cplx[i].im;
                }
            } else {
                alpha = 1.0;
                beta = 0.0;
                my_dgemm(&transA,&transB,(int*)&M,(int*)&N,(int*)&K,
                    &alpha,U,(int*)&LDA,Vt,(int*)&LDB,&beta,output,(int*)&M );
            }
        }

    } else {

        /* ----------- MODE 3 ------------------------------- */

        if (SPARSE) {
            /* sparse array indices in Matlab are rather confusing;
             * see mxSetJc help file to get started.  The Ir index
             * is straightforward: it contains rows indices of nonzeros,
             * in column-major order.  But the Jc index is tricky... 
             * Basically, Jc (which has N+1 entries, not nnz entries like Ir)
             * tells you which Ir entries correspond to the jth row, thus fully 
             * specifying the indices.  Ir[ Jc[j]:Jc[J+1] ] are the rows
             * that correspond to column j. This works because Ir is
             * in column-major order.   For this to work (and match A(omega)),
             * we need omega to be sorted!  */
            omegaI = mxGetIr( prhs[2] );
            omegaJ = mxGetJc( prhs[2] );
            
            if ((COMPLEX)&&(LARGE_BIT)) {
                for ( j=0 ; j < N ; j++ ){
                    for ( k = omegaJ[j] ; k < omegaJ[j+1] ; k++ ) {
                        i = (int) omegaI[k];

                        temp_cplx.re = 0.0;
                        temp_cplx.im = 0.0;
                        for ( m=0 ; m < K ; m++ ){
                            temp_cplx.re += U_cplx[i+m*M].re * Vt_cplx[j+m*N].re
                                - U_cplx[i+m*M].im * Vt_cplx[j+m*N].im;
                            temp_cplx.im += U_cplx[i+m*M].im * Vt_cplx[j+m*N].re
                                + U_cplx[i+m*M].re * Vt_cplx[j+m*N].im;
                        }
                        output[k] = temp_cplx.re;
                        output_imag[k] = temp_cplx.im;
                    }
                }
            } else if (COMPLEX) {
                for ( j=0 ; j < N ; j++ ){
                    for ( k = omegaJ[j] ; k < omegaJ[j+1] ; k++ ) {
                        i = (int) omegaI[k];
                        temp_cplx = my_zdot((int*) &K, U_cplx+i, (int*)&M, Vt_cplx+j, (int*)&N );
                        output[k] = temp_cplx.re;
                        output_imag[k] = temp_cplx.im;
                    }
                }
            } else {
                for ( j=0 ; j < N ; j++ ){
                    for ( k = omegaJ[j] ; k < omegaJ[j+1] ; k++ ) {
                        i = (int) omegaI[k];
                        /* mexPrintf("%2d %2d\n",i+1,j+1);  */
                        output[k] = my_ddot( (int*)&K, U+i, (int*)&M, Vt+j, (int*)&N );
                    }
                }
            }
        /* ----------- MODE 4 ------------------------------- */
        } else {
            /* we have omegaX and omegaY, the row and column indices */
            nOmega1 = mxGetM( prhs[3] );
            nOmega2 = mxGetN( prhs[3] );
            if ( (nOmega1 != 1) && (nOmega2 != 1) ) {
                printUsage();
                mexErrMsgTxt("OmegaY must be a vector");
            }
            nOmega1 = nOmega1 < nOmega2 ? nOmega2 : nOmega1;
            if ( nOmega1 != nOmega ) {
                printUsage();
    mexErrMsgTxt("In subscript index format, subscript vectors must be same length.");
            }
            omegaX = mxGetPr(prhs[2]);
            omegaY = mxGetPr(prhs[3]);

            if ((COMPLEX)&&(LARGE_BIT)) {
                for ( k=0 ; k < nOmega ; k++ ){
                    i = (int) omegaX[k] - 1;
                    j = (int) omegaY[k] - 1;
                    temp_cplx.re = 0.0;
                    temp_cplx.im = 0.0;
                    for ( m=0 ; m < K ; m++ ){
                        temp_cplx.re += U_cplx[i+m*M].re * Vt_cplx[j+m*N].re
                            - U_cplx[i+m*M].im * Vt_cplx[j+m*N].im;
                        temp_cplx.im += U_cplx[i+m*M].im * Vt_cplx[j+m*N].re
                            + U_cplx[i+m*M].re * Vt_cplx[j+m*N].im;
                    }
                    output[k] = temp_cplx.re;
                    output_imag[k] = temp_cplx.im;
                }
            } else if (COMPLEX) {
                for ( k=0 ; k < nOmega ; k++ ){
                    i = (int) omegaX[k] - 1;
                    j = (int) omegaY[k] - 1;
                    temp_cplx = my_zdot( (int*)&K, U_cplx+i, (int*)&M, Vt_cplx+j, (int*)&N );
                    output[k] = temp_cplx.re;
                    output_imag[k] = temp_cplx.im;
                }
            } else {
                for ( k=0 ; k < nOmega ; k++ ){
                    i = (int) omegaX[k] - 1;
                    j = (int) omegaY[k] - 1;
                    output[k] = my_ddot( (int*)&K, U+i, (int*)&M, Vt+j, (int*)&N );
                }
            }

        }
    }

}

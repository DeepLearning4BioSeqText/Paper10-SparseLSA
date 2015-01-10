/* 
MEX interface for LAPACK routine bdsqr.
Matlab calling sequence:
  [sigma,bnd] = bdsqr(alpha,beta)   
*/

/* Stephen Becker, 11/12/08
 * Now, dbdqr is in a C file.  So, no underscore for the name.
 * Also, for Windows, the Lapack libraries don't need underscores
 * either.  These are controlled by pre-processor definitions
 * that I put in; either edit this source file, or, better, pass
 * in a -D[name of variable].
 * e.g. mex ... -DWINDOWS ...
 * */

/* #define WINDOWS */

/* #define DBDQR_IN_C */


#include <stdio.h>
#include <string.h>
#include "mex.h"

/* Templates for FORTRAN routines: */
#ifdef DBDQR_IN_C
void dbdqr(int *n, double *d, double *e, double *c1, double *c2);
#else
void dbdqr_(int *n, double *d, double *e, double *c1, double *c2);
#endif

#ifdef WINDOWS
void dbdsqr(char *uplo, int *n, int *ncvt, int *nru, int *ncc,
	     double *d, double *e, double *vt, int *ldt, double *u,
	     int *ldu, double *c, int *ldc, double *work, int *info);
#else
void dbdsqr_(char *uplo, int *n, int *ncvt, int *nru, int *ncc,
	     double *d, double *e, double *vt, int *ldt, double *u,
	     int *ldu, double *c, int *ldc, double *work, int *info);
#endif

/* Here comes the gateway function to be called by Matlab: */
void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{
  int m, n, i, info, zero=0, one=1;
  double *d,*e,dummy, *wrk, *bnd;

  if (nrhs != 2)
     mexErrMsgTxt("bdsqr requires two input arguments");
  else if  (nlhs != 2)
     mexErrMsgTxt("bdsqr requires two output arguments");

  m = mxGetM(prhs[0]); /* get the dimensions of the input */
  n = mxGetN(prhs[0]);
  /* make sure input input vectors are same length */
  if (m != mxGetM(prhs[1]) )
    mexErrMsgTxt("alpha and beta must have the same size");
  /* make sure input is m x 1 */
  if ( n != 1 || mxGetN(prhs[1]) != 1 || n != mxGetN(prhs[1])) 
    mexErrMsgTxt("alpha and beta must be a m x 1 vectors");
    
  /* Create/allocate return arguments */
  for (i=0; i<2; i++) { 
    plhs[i]=mxCreateDoubleMatrix(m,1,mxREAL); 
  }

  e = mxCalloc(m,sizeof(double));
  wrk = mxCalloc(4*m-4,sizeof(double));
  d = mxGetPr(plhs[0]);
  memcpy(d,mxGetPr(prhs[0]), m*sizeof(double));
  memcpy(e,mxGetPr(prhs[1]), m*sizeof(double));
  bnd = mxGetPr(plhs[1]);
  for (i=0; i<m; i++)
    bnd[i] = 0;

  /* Reduce to upper m-by-m upper bidiagonal */
#ifdef DBDQR_IN_C
  dbdqr(&m, d, e, &bnd[m-1],&dummy);
#else
  dbdqr_(&m, d, e, &bnd[m-1],&dummy);
#endif

  /* Compute singular values and last row of U */
#ifdef WINDOWS
  dbdsqr("u", &m, &zero, &one, &zero, d, e, &dummy, &one,
       bnd, &one, &dummy, &one, wrk, &info); 
#else
  dbdsqr_("u", &m, &zero, &one, &zero, d, e, &dummy, &one,
       bnd, &one, &dummy, &one, wrk, &info); 
#endif

  /* Check exit status of dbdsqr */
  if ( info < 0 )
    mexErrMsgTxt("DBDSQR was called with illegal arguments");
  else if ( info > 0)
    mexWarnMsgTxt("DBDSQR: singular values did not converge");

  /* Free work arrays */
  mxFree(e);
  mxFree(wrk);
}




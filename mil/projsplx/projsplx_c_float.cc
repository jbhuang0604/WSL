#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"

/* 
(c) Xiaojing Ye
xye@ufl.edu
Department of Mathematics
University of Florida
http://www.math.ufl.edu/~xye
Jan. 15, 2011.
Paper link: http://ufdc.ufl.edu/IR00000353/
*/

typedef float ElementType;
     
     
void Swap( ElementType *Lhs, ElementType *Rhs )
{
            ElementType Tmp = *Lhs;
            *Lhs = *Rhs;
            *Rhs = Tmp;
}
ElementType Median3( ElementType A[ ], int Left, int Right )
{
            int Center = ( Left + Right ) / 2;

            if( A[ Left ] > A[ Center ] )
                Swap( &A[ Left ], &A[ Center ] );
            if( A[ Left ] > A[ Right ] )
                Swap( &A[ Left ], &A[ Right ] );
            if( A[ Center ] > A[ Right ] )
                Swap( &A[ Center ], &A[ Right ] );

            /* Invariant: A[ Left ] <= A[ Center ] <= A[ Right ] */

            Swap( &A[ Center ], &A[ Right - 1 ] );  /* Hide pivot */
            return A[ Right - 1 ];                /* Return pivot */
}
void InsertionSort( ElementType A[ ], int N )
{
            int j, P;
            ElementType Tmp;

/* 1*/      for( P = 1; P < N; P++ )
            {
/* 2*/          Tmp = A[ P ];
/* 3*/          for( j = P; j > 0 && A[ j - 1 ] > Tmp; j-- )
/* 4*/              A[ j ] = A[ j - 1 ];
/* 5*/          A[ j ] = Tmp;
            }
}
        
#define Cutoff ( 3 )

void Qsort( ElementType A[ ], int Left, int Right )
{
            int i, j;
            ElementType Pivot;
            if( Left + Cutoff <= Right )
            {
	        Pivot = Median3( A, Left, Right );
                i = Left; j = Right - 1;
	        for( ; ; )
                {
	            while( A[ ++i ] < Pivot ){ }
	            while( A[ --j ] > Pivot ){ }
                    if( i < j )
                       Swap( &A[ i ], &A[ j ] );
                    else
                       break;
                }
                Swap( &A[ i ], &A[ Right - 1 ] );  /* Restore pivot */

                Qsort( A, Left, i - 1 );
                Qsort( A, i + 1, Right );
            }
            else  /* Do an insertion sort on the subarray */
	        InsertionSort( A + Left, Right - Left + 1 );
}
      
          
void Quicksort( ElementType A[ ], int N )
{
            Qsort( A, 0, N - 1 );
}
    
        
void mexFunction
(
    int nargout,
    mxArray *plhs[],
    int nargin,
    const mxArray *prhs [ ]
)
{
    /* get the size and pointers for input Data */
    /* int m  = mxGetM(prhs[0]);
    int n  = mxGetN(prhs[0]);
    m = (m>n)? m:n;*/

    int numDims, m, d, j;
    const int *dims;
    float *y, *s, *x;
    float sumResult = -1, tmpValue, tmax; 
    bool bget = false;
    
    if (nargin < 1)
    {
        mexErrMsgTxt ("One input argument is required for function x = projsplx(y)") ;
    }

    dims = mxGetDimensions(prhs[0]);
    numDims = mxGetNumberOfDimensions(prhs[0]);  

    /*m = dims[0]; n=dims[1];*/
    m = 0;
    for (d=0; d<numDims; d++) {
       m= (m > dims[d])? m:dims[d];
    }

    y  = (float*)mxGetPr(prhs[0]); 
     
     
    /*  set the output pointer to the output matrix */
    plhs[0] = mxCreateNumericMatrix(m,1,mxSINGLE_CLASS, mxREAL);
    
    /* s = sort(y,'ascend'); */
    s = (float*) calloc (m,sizeof(float));
    for(j = 0; j < m; j++ ){
    	s[j] = y[j]; 
    }
    Quicksort(s,m);
    
    x = (float*)mxGetPr(plhs[0]);

    /* if t is not less than s[0] */
    for(j = m-1; j >= 1; j--){    	
    	sumResult = sumResult + s[j];
    	tmax = sumResult/(m-j);
	if(tmax >= s[j-1]){
		bget = true;
		break;
	}
    }   

    /* if t is less than s[0] */
    if(!bget){
	sumResult = sumResult + s[0];
	tmax = sumResult/m;
    }
    free(s);

    /* x = max(y-tmax, 0); */
    for(j = 0; j <= m-1; j++){
	tmpValue = y[j] - tmax;
	x[j] = (tmpValue > 0)? tmpValue:0;
    }
}


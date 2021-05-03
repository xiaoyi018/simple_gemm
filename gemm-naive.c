const char* gemm_desc = "Naive, three-loop gemm.";

/* This routine performs a gemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void square_gemm (int n, float* A, float* B, float* C)
{
  /* For each row i of A */
  for (int i = 0; i < n; ++i)
    /* For each column j of B */
    for (int j = 0; j < n; ++j) 
    {
      /* Compute C(i,j) */
      float cij = C[i+j*n];
      for( int k = 0; k < n; k++ )
	cij += A[i+k*n] * B[k+j*n];
      C[i+j*n] = cij;
    }
}

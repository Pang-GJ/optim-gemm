/* Routine for computing C = A * B */

void AddDot4x4(int, float*, float*, int, float*);

void MY_MMult(int m, int n, int k, float* a, int lda, float* b, int ldb,
              float* c, int ldc) {
  int i, j;

#define A(i, j) a[(i) * k + (j)]
#define B(i, j) b[(i) * n + (j)]
#define C(i, j) c[(i) * n + (j)]

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      AddDot4x4(k, &A(i, 0), &B(0, j), ldb, &C(i, j));
    }
  }

#undef A
#undef B
#undef C
}

void AddDot4x4(int k, float* x, float* y, int ldb, float* gamma) {
  for (int p = 0; p < k; p++) {
    *gamma += x[p] * y[p * ldb];
  }
}
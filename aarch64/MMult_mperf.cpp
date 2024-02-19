
#include <arm_neon.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>

#include "megengine.h"

#define ENABLE_TMA 1

/* Block sizes */
#define kc 256
#define nc 252

/* Create macros so that the matrices are stored in row-major order */

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

/* Routine for computing C = A * B + C */
void InnerKernel(int, int, int, float*, int, float*, int, float*, int, bool);

void PackMatrixB_12(int, float*, int, float*);
void PackMatrixB_4(int, float*, int, float*);
void PackMatrixA_8(int, float*, int, float*);
void PackMatrixA_4(int, float*, int, float*);

void PackMatrixA(int, float*, int, float*);

// void my_matmul_asm(int m, int n, int k, float* a, int lda, float* b, int ldb,
//                    float* c, int ldc) {
// void MY_MMult(int m, int n, int k, float* a, int lda, float* b, int ldb,
//               float* c, int ldc) {
//   int j, p, pb, ib;
//   for (p = 0; p < k; p += kc) {
//     pb = min(k - p, kc);
//     bool is_first_k = (p == 0) ? 1 : 0;
//     for (j = 0; j < n; j += nc) {
//       ib = min(n - j, nc);
//       InnerKernel(m, ib, pb, &A(0, p), lda, &B(p, j), ldb, &C(0, j), ldc,
//                   is_first_k);
//     }
//   }
// }
void MY_MMult(int m, int n, int k, float* a, int lda, float* b, int ldb,
              float* c, int ldc) {
  sgemm_8x12 strategy(m, n, k);
  bool trA = false, trB = false;

  GemmInterleaved<sgemm_8x12> gemm(m, n, k, trA, trB, strategy);
  float workspace[gemm.get_workspace_size()];
  gemm.execute(a, lda, b, ldb, c, ldc, &workspace);
}

void PackMatrixB_12(int k, float* b, int ldb, float* b_to) {
  int j;
  for (j = 0; j < k; ++j) {
    float* b_ij_pntr = &B(j, 0);
    *b_to++ = b_ij_pntr[0];
    *b_to++ = b_ij_pntr[1];
    *b_to++ = b_ij_pntr[2];
    *b_to++ = b_ij_pntr[3];
    *b_to++ = b_ij_pntr[4];
    *b_to++ = b_ij_pntr[5];
    *b_to++ = b_ij_pntr[6];
    *b_to++ = b_ij_pntr[7];
    *b_to++ = b_ij_pntr[8];
    *b_to++ = b_ij_pntr[9];
    *b_to++ = b_ij_pntr[10];
    *b_to++ = b_ij_pntr[11];
  }
}
void PackMatrixB_4(int k, float* b, int ldb, float* b_to) {
  int j;
  for (j = 0; j < k; ++j) {
    float* b_ij_pntr = &B(j, 0);
    *b_to++ = b_ij_pntr[0];
    *b_to++ = b_ij_pntr[1];
    *b_to++ = b_ij_pntr[2];
    *b_to++ = b_ij_pntr[3];
  }
}
void PackMatrixA_8(int k, float* a, int lda, float* a_to) {
  int i;
  float *a_0i_pntr = a, *a_1i_pntr = a + lda, *a_2i_pntr = a + (lda << 1),
        *a_3i_pntr = a + (3 * lda), *a_4i_pntr = a + (lda << 2),
        *a_5i_pntr = a + (lda * 5), *a_6i_pntr = a + (lda * 6),
        *a_7i_pntr = a + (lda * 7);

  for (i = 0; i < k; ++i) {
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
    *a_to++ = *a_4i_pntr++;
    *a_to++ = *a_5i_pntr++;
    *a_to++ = *a_6i_pntr++;
    *a_to++ = *a_7i_pntr++;
  }
}
void PackMatrixA_4(int k, float* a, int lda, float* a_to) {
  int i;
  float *a_0i_pntr = a, *a_1i_pntr = a + lda, *a_2i_pntr = a + (lda << 1),
        *a_3i_pntr = a + (3 * lda);

  for (i = 0; i < k; ++i) {
    *a_to++ = *a_0i_pntr++;
    *a_to++ = *a_1i_pntr++;
    *a_to++ = *a_2i_pntr++;
    *a_to++ = *a_3i_pntr++;
  }
}
void InnerKernel(int m, int n, int k, float* a, int lda, float* b, int ldb,
                 float* c, int ldc, bool is_first_k) {
  int i, j;
  float packedA[m * k];
  float packedB[k * n];
  for (j = 0; j < n; j += 12) {
    if (j + 12 > n) break;
    PackMatrixB_12(k, &B(0, j), ldb, packedB + j * k);
    for (i = 0; i < m; i += 8) {
      if (i + 8 > m) break;
      if (0 == j) {
        PackMatrixA_8(k, &A(i, 0), lda, packedA + i * k);
      }
      // Kern_8x12(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
      // is_first_k);
      matmul_general_8x12_a53::kern_8x12(packedA + i * k, packedB + j * k, k,
                                         &C(i, j), ldc, is_first_k);
    }
    if (i != m) {
      if (0 == j) PackMatrixA_4(k, &A(i, 0), lda, packedA + i * k);
      // Kern_4x12(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
      // is_first_k,
      //           4);
      matmul_general_8x12_a53::kern_4x12(packedA + i * k, packedB + j * k, k,
                                         &C(i, j), ldc, is_first_k, 4);
    }
  }
  if (j != n) {
    for (; j < n; j += 4) {
      PackMatrixB_4(k, &B(0, j), ldb, packedB + j * k);
      for (i = 0; i < m; i += 8) {
        if (i + 8 > m) break;
        // Kern_8x4(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
        // is_first_k,
        //          n - j);
        matmul_general_8x12_a53::kern_8x4(packedA + i * k, packedB + j * k, k,
                                          &C(i, j), ldc, is_first_k, n - j);
      }
      if (i != m) {
        // Kern_4x4(packedA + i * k, packedB + j * k, k, &C(i, j), ldc,
        // is_first_k,
        //          m - i, n - j);

        matmul_general_8x12_a53::kern_4x4(packedA + i * k, packedB + j * k, k,
                                          &C(i, j), ldc, is_first_k, m - i,
                                          n - j);
      }
    }
  }
}

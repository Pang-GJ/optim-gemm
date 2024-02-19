#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#error("arm neon not supported")
#endif

/* Block sizes */
#define mc 64
#define kc 64

/* Routine for computing C = A * B */
#include <sys/types.h>

#define A(i, j) a[(i) * lda + (j)]
#define B(i, j) b[(i) * ldb + (j)]
#define C(i, j) c[(i) * ldc + (j)]

#define min(i, j) ((i) < (j) ? (i) : (j))

void AddDot4x4(int, float*, int, float*, int, float*, int);

void InnerKernel(int, int, int, float*, int, float*, int, float*, int);

void PackMatrixB(int, float*, int, float*);
void PackMatrixA(int, float*, int, float*);

void MY_MMult(int m, int n, int k, float* a, int lda, float* b, int ldb,
              float* c, int ldc) {
  int i, p, pb, ib;
  for (p = 0; p < k; p += kc) {
    pb = min(k - p, kc);
    for (i = 0; i < m; i += mc) {
      ib = min(m - i, mc);
      InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc);
    }
  }
}

void PackMatrixB(int k, float* b, int ldb, float* packed_b) {
  int j;
  for (j = 0; j < k; ++j) {
    float* b_ij_pntr = &B(j, 0);
    *packed_b++ = b_ij_pntr[0];
    *packed_b++ = b_ij_pntr[1];
    *packed_b++ = b_ij_pntr[2];
    *packed_b++ = b_ij_pntr[3];
  }
}

void PackMatrixA(int k, float* a, int lda, float* packed_a) {
  int i;
  float *a_0i_pntr = a, *a_1i_pntr = a + lda, *a_2i_pntr = a + 2 * lda,
        *a_3i_pntr = a + 3 * lda;

  for (i = 0; i < k; ++i) {
    *packed_a++ = *a_0i_pntr++;
    *packed_a++ = *a_1i_pntr++;
    *packed_a++ = *a_2i_pntr++;
    *packed_a++ = *a_3i_pntr++;
  }
}

void InnerKernel(int m, int n, int k, float* a, int lda, float* b, int ldb,
                 float* c, int ldc) {
  int i, j;
  float packed_a[m * k];
  float packed_b[k * n];

  // unrolling the loop 4 times
  for (int j = 0; j < n; j += 4) {
    PackMatrixB(k, &B(0, j), ldb, &packed_b[j * k]);
    for (int i = 0; i < m; i += 4) {
      if (0 == j) {
        PackMatrixA(k, &A(i, 0), lda, packed_a + i * k);
      }
      AddDot4x4(k, packed_a + i + k, lda, packed_b + j * k, ldb, &C(i, j), ldc);
    }
  }
}

void AddDot4x4(int k, float* a, int lda, float* b, int ldb, float* c, int ldc) {
  /* So, this routine computes a 4x4 block of matrix A

             C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).
             C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).
             C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).
             C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).

       Notice that this routine is called with c = C( i, j ) in the
       previous routine, so these are actually the elements

             C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 )
             C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 )
             C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 )
             C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 )

       in the original matrix C

       In this version, we use registers for elements in the current row
       of B as well */
  int p;
  register float  // clangd-disable-line
      /* hold
           A( 0, p )
           A( 1, p )
           A( 2, p )
           A( 3, p ) */
      a_0p_reg,
      a_1p_reg, a_2p_reg, a_3p_reg;

  float32x4_t c_p0_sum = {0};
  float32x4_t c_p1_sum = {0};
  float32x4_t c_p2_sum = {0};
  float32x4_t c_p3_sum = {0};

  /**
    vld1q_f32：这个指令的命名可以分解为：
      vld：基本操作是向量加载。
      1：表示每次加载一个元素到每个lane。
      q：表示操作的是128位的向量寄存器。
      f32：表示操作的数据类型是32位浮点数。
    vmlaq_n_f32：这个指令的命名可以分解为：
      vmla：基本操作是向量乘加，即先进行乘法然后将结果加到累加器。
      q：表示操作的是128位的向量寄存器。
      n：表示使用一个标量值进行乘法。
      f32：表示操作的数据类型是32位浮点数。
   */

  for (p = 0; p < k; p++) {
    float32x4_t b_reg = vld1q_f32(b);
    b += 4;

    a_0p_reg = a[0];
    a_1p_reg = a[1];
    a_2p_reg = a[2];
    a_3p_reg = a[3];
    a += 4;

    // first row of C
    c_p0_sum = vmlaq_n_f32(c_p0_sum, b_reg, a_0p_reg);

    // second row of C
    c_p1_sum = vmlaq_n_f32(c_p1_sum, b_reg, a_1p_reg);

    // third row of C
    c_p2_sum = vmlaq_n_f32(c_p2_sum, b_reg, a_2p_reg);

    // fourth row of C
    c_p3_sum = vmlaq_n_f32(c_p3_sum, b_reg, a_3p_reg);
  }
  // vaddq_f32 指令的作用是将两个 128
  // 位寄存器中的四个单精度浮点数相加，并将结果存储在另一个 128
  // 位寄存器中。每个浮点数的加法操作是独立进行的，

  // vst1q_f32 指令的作用是将一个 128 位的 NEON
  // 寄存器中的四个单精度浮点数存储到连续的内存地址中。

  // write back results
  float* c_pntr = &C(0, 0);
  float32x4_t c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p0_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(1, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p1_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(2, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p2_sum);
  vst1q_f32(c_pntr, c_reg);

  c_pntr = &C(3, 0);
  c_reg = vld1q_f32(c_pntr);
  c_reg = vaddq_f32(c_reg, c_p3_sum);
  vst1q_f32(c_pntr, c_reg);
}
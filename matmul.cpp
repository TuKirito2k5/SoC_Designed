#include "blockmatmul.h"

// ============================================================================
// HÀM NHÂN MA TRẬN THÔNG THƯỜNG (BASELINE MATRIX MULTIPLICATION)
// Tính AB[N][N] = A[N][N] * B[N][N]
// ============================================================================
void matmul_baseline(DTYPE A[N][N], DTYPE B[N][N], DTYPE AB[N][N]) {
#ifdef USE_DIRECTIVES
    #pragma HLS ARRAY_RESHAPE variable=A complete dim=2
    #pragma HLS ARRAY_RESHAPE variable=B complete dim=1
#endif

    ROW_LOOP: for (int i = 0; i < N; i++) {
        COL_LOOP: for (int j = 0; j < N; j++) {
#ifdef USE_DIRECTIVES
            #pragma HLS PIPELINE II=1
#endif
            DTYPE sum = 0;
            PRODUCT_LOOP: for (int k = 0; k < N; k++) {
#ifdef USE_DIRECTIVES
                #pragma HLS UNROLL
#endif
                sum += A[i][k] * B[k][j];
            }
            AB[i][j] = sum;
        }
    }
}
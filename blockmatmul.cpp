#include "blockmatmul.h"

void blockmatmul(hls::stream<blockvec, 0> &Arows,
                 hls::stream<blockvec, 0> &Bcols,
                 hls::stream<blockvec, 0> &ABpartial,
                 int it) {
#pragma HLS INTERFACE mode=s_axilite port=return bundle=BUS_A
#pragma HLS INTERFACE mode=s_axilite port=it bundle=BUS_A
#pragma HLS INTERFACE mode=axis port=Arows
#pragma HLS INTERFACE mode=axis port=Bcols
#pragma HLS INTERFACE mode=axis port=ABpartial

    int counter = it % (N / BLOCK_SIZE);

    static DTYPE A[BLOCK_SIZE][N];
#ifdef USE_DIRECTIVES
    #pragma HLS ARRAY_RESHAPE variable=A complete dim=1
#endif

    DTYPE AB[BLOCK_SIZE][BLOCK_SIZE];
#ifdef USE_DIRECTIVES
    #pragma HLS ARRAY_PARTITION variable=AB complete dim=0
#endif

    INIT_AB_ROW: for (int i = 0; i < BLOCK_SIZE; i++) {
        INIT_AB_COL: for (int j = 0; j < BLOCK_SIZE; j++) {
            #pragma HLS UNROLL
            AB[i][j] = 0;
        }
    }

    // 1. Nạp Arows
    if (counter == 0) {
        LOAD_A_COL: for (int k = 0; k < N; k++) {
            #pragma HLS PIPELINE II=1
            blockvec tempA = Arows.read();
            // Trích xuất 4 giá trị int (mỗi cái 32-bit) từ 128-bit data
            LOAD_A_ROW: for (int i = 0; i < BLOCK_SIZE; i++) {
                #pragma HLS UNROLL
                ap_int<32> val = tempA.data.range(i * 32 + 31, i * 32);
                A[i][k] = val;
            }
        }
    }

    // 2. Tính toán Outer-Product
    PARTIAL_SUM_K: for (int k = 0; k < N; k++) {
        #pragma HLS PIPELINE II=1
        blockvec tempB = Bcols.read();
        PARTIAL_SUM_I: for (int i = 0; i < BLOCK_SIZE; i++) {
            #pragma HLS UNROLL
            PARTIAL_SUM_J: for (int j = 0; j < BLOCK_SIZE; j++) {
                #pragma HLS UNROLL
                ap_int<32> valB = tempB.data.range(j * 32 + 31, j * 32);
                AB[i][j] += A[i][k] * valB;
            }
        }
    }

    // 3. Ghi kết quả ra ABpartial
    WRITE_OUT_ROW: for (int i = 0; i < BLOCK_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        blockvec temp_row;
        temp_row.data = 0;
        WRITE_OUT_COL: for (int j = 0; j < BLOCK_SIZE; j++) {
            #pragma HLS UNROLL
            // Đóng gói 4 phần tử int vào 128-bit data
            temp_row.data.range(j * 32 + 31, j * 32) = (ap_int<32>)AB[i][j];
        }

        if (i == BLOCK_SIZE - 1) {
            temp_row.last = 1;
        } else {
            temp_row.last = 0;
        }

        ABpartial.write(temp_row);
    }
}

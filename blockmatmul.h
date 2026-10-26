#ifndef BLOCKMATMUL_H_
#define BLOCKMATMUL_H_

#include <hls_stream.h>
#include <ap_int.h>

#ifndef N
#define N 8
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4
#endif

#if (N % BLOCK_SIZE != 0)
#error "Kich thuoc ma tran N phai chia het cho BLOCK_SIZE!"
#endif

typedef int DTYPE;

struct blockvec {
    ap_int<128> data;
    ap_uint<1>  last;
};

typedef struct {
    DTYPE out[BLOCK_SIZE][BLOCK_SIZE];
} blockmat;

void matmul_baseline(DTYPE A[N][N], DTYPE B[N][N], DTYPE AB[N][N]);

void blockmatmul(hls::stream<blockvec> &Arows,
                 hls::stream<blockvec> &Bcols,
                 hls::stream<blockvec> &ABpartial,
                 int it);

#endif // BLOCKMATMUL_H_

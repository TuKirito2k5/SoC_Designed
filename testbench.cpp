#include <iostream>
#include <cstdlib>
#include <iomanip>
#include "blockmatmul.h"

// ============================================================================
// HÀM NHÂN MA TRẬN THAM CHIẾU PHẦN MỀM (GOLDEN SOFTWARE REFERENCE)
// ============================================================================
void matmul_sw_golden(DTYPE A[N][N], DTYPE B[N][N], DTYPE AB_sw[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            DTYPE sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            AB_sw[i][j] = sum;
        }
    }
}

// ============================================================================
// TESTBENCH HỢP NHẤT (UNIFIED TESTBENCH CHO C SIMULATION & C/RTL CO-SIMULATION)
// ============================================================================
int main() {
    DTYPE A[N][N];
    DTYPE B[N][N];
    DTYPE AB_golden[N][N];
    DTYPE AB_base[N][N];
    DTYPE AB_block[N][N];

    // Cố định seed = 42 để dữ liệu ngẫu nhiên lặp lại hoàn toàn giống nhau
    std::srand(42);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (std::rand() % 20) - 10; // Giá trị trong khoảng [-10, 9]
            B[i][j] = (std::rand() % 20) - 10;
            AB_golden[i][j] = 0;
            AB_base[i][j]   = 0;
            AB_block[i][j]  = 0;
        }
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << " BAT DAU KIEM THU BO TANG TOC NHAN MA TRAN (VITIS HLS)" << std::endl;
    std::cout << " Cau hinh: N = " << N << "x" << N
              << " | BLOCK_SIZE = " << BLOCK_SIZE << "x" << BLOCK_SIZE
#ifdef USE_DIRECTIVES
              << " | Directives = ON" << std::endl;
#else
              << " | Directives = OFF" << std::endl;
#endif
    std::cout << "==========================================================" << std::endl;

    // 1. Tính kết quả chuẩn trên phần mềm (Golden Reference)
    matmul_sw_golden(A, B, AB_golden);

    // 2. Chạy kiểm thử hàm nhân ma trận thông thường (Baseline)
    matmul_baseline(A, B, AB_base);

    // 3. Chạy kiểm thử hàm nhân ma trận theo khối tương thích Stream (blockmatmul)
    hls::stream<blockvec> Arows("Arows_stream");
    hls::stream<blockvec> Bcols("Bcols_stream");
    hls::stream<blockvec, 0> ABpartial;

    int num_blocks = N / BLOCK_SIZE;
    int it = 0;

    // Duyệt qua từng khối hàng (blk_row) của A và từng khối cột (blk_col) của B
    for (int blk_row = 0; blk_row < num_blocks; blk_row++) {
        for (int blk_col = 0; blk_col < num_blocks; blk_col++) {

            // Đẩy dải ngang của A vào stream Arows ở khối cột đầu tiên (blk_col == 0)
            if (blk_col == 0) {
                for (int k = 0; k < N; k++) {
                	blockvec tempA;
                	tempA.last = 0;
                	tempA.data = 0;
                	for (int i = 0; i < BLOCK_SIZE; i++) {
                	    tempA.data.range(i * 32 + 31, i * 32) = (ap_int<32>)A[blk_row * BLOCK_SIZE + i][k];
                	}
                	Arows.write(tempA);
                }
            }

            // Đẩy dải dọc của B vào stream Bcols cho mỗi khối (blk_row, blk_col)
            for (int k = 0; k < N; k++) {
                blockvec tempB;
                tempB.last = 0;
                tempA.keep = -1;
                tempB.data = 0;
                for (int j = 0; j < BLOCK_SIZE; j++) {
                    tempB.data.range(j * 32 + 31, j * 32) = (ap_int<32>)B[k][blk_col * BLOCK_SIZE + j];
                }
                Bcols.write(tempB);
            }

            // Gọi bộ tăng tốc phần cứng blockmatmul (đã tích hợp TLAST)
            blockmatmul(Arows, Bcols, ABpartial, it);

            // Đọc kết quả từ stream ABpartial và kiểm tra tín hiệu TLAST
            for (int i = 0; i < BLOCK_SIZE; i++) {
                blockvec out_row = ABpartial.read();

                // Kiểm tra tính hợp lệ của tín hiệu TLAST do phần cứng trả về
                if (i == BLOCK_SIZE - 1) {
                    if (out_row.last != 1) {
                        std::cout << "[CANH BAO CO-SIM] Thieu tin hieu TLAST tai hang cuoi block!" << std::endl;
                    }
                } else {
                    if (out_row.last != 0) {
                        std::cout << "[CANH BAO CO-SIM] Tin hieu TLAST bi bat sai vi tri!" << std::endl;
                    }
                }

                for (int j = 0; j < BLOCK_SIZE; j++) {
                	ap_int<32> val = out_row.data.range(j * 32 + 31, j * 32);
                	    AB_block[blk_row * BLOCK_SIZE + i][blk_col * BLOCK_SIZE + j] = (DTYPE)val;
                }
            }

            it++;
        }
    }

    // 4. Đối chiếu kết quả giữa Baseline, Block Matrix và Golden Reference
    int err_baseline = 0;
    int err_block    = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (AB_base[i][j] != AB_golden[i][j]) {
                err_baseline++;
                std::cout << "[LOI BASELINE] Tai (" << i << "," << j << "): "
                          << "Nhan duoc = " << AB_base[i][j]
                          << ", Ky vong = " << AB_golden[i][j] << std::endl;
            }
            if (AB_block[i][j] != AB_golden[i][j]) {
                err_block++;
                std::cout << "[LOI BLOCKMATMUL] Tai (" << i << "," << j << "): "
                          << "Nhan duoc = " << AB_block[i][j]
                          << ", Ky vong = " << AB_golden[i][j] << std::endl;
            }
        }
    }

    if (err_baseline == 0 && err_block == 0) {
        std::cout << ">>> KET QUA: PASS! Ca hai thuat toan Baseline va Block Matrix ("
                  << N << "x" << N << ", Block " << BLOCK_SIZE << "x" << BLOCK_SIZE
                  << ") deu khop 100% voi Golden Reference." << std::endl;
        return 0; // Trả về 0 để Vitis HLS xác nhận C Simulation & Co-Simulation thành công
    } else {
        std::cout << ">>> KET QUA: FAIL! Loi Baseline = " << err_baseline
                  << ", Loi Block = " << err_block << std::endl;
        return 1;
    }
}

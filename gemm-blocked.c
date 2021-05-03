const char* gemm_desc = "Blocked gemm with packing and intrinsics.";

#include "arm_neon.h"

#define A(i,j) a[(j)*n + (i)]
#define B(i,j) b[(j)*n + (i)]
#define C(i,j) c[(j)*n + (i)]

//size of macro kernel
#define KC 256
#define MC 1024
#define NC 1024

//size of micro kernel
#define MR 8
#define NR 8

//Local buffers for storing panels from A, B and C
static float _A[MC * KC] __attribute__ ((aligned (16)));;
static float _B[KC * NC] __attribute__ ((aligned (16)));;
static float _C[MR * NR] __attribute__ ((aligned (16)));;

//Packing complete panels from A (i.e. without padding)
static void pack_MRxk(int n, int k, const float *a, float *a_to){
    int i, j;
    for(j = 0; j < k; j++){
        for (i = 0; i < MR; i++)
            a_to[i] = A(i, 0);
        a_to += MR;
        a += n;
    }
}

//Packing panels from A with padding if required
static void pack_A(int n, int mc, int kc, const float *a, float *a_to){
    int mp = mc / MR;
    int _mr = mc % MR;
    int i, j;
    for (i = 0; i < mp; i++){
        pack_MRxk(n, kc, a, a_to);
        a_to += kc * MR;
        a += MR;
    }
    if (_mr > 0){
        for (j = 0; j < kc; j++){
            for (i = 0; i < _mr; i++)
                a_to[i] = A(i, 0);
            for (i = _mr; i < MR; i++)
                a_to[i] = 0.0;
            a_to += MR;
            a += n;
        }
    }
}

//Packing complete panels from B (i.e. without padding)
static void pack_kxNR(int n, int k, const float *b, float *b_to){
    int i, j;
    for (i = 0; i < k; i++){
        for (j = 0; j < NR; j++)
            b_to[j] = B(0, j);
        b_to += NR;
        b++;
    }
}

//Packing panels from B with padding if required
static void pack_B(int n, int kc, int nc, const float *b, float *b_to){
    int np = nc / NR;
    int _nr = nc % NR;
    int i, j;
    for (j = 0; j < np; j++){
        pack_kxNR(n, kc, b, b_to);
        b_to += kc * NR;
        b += NR * n;
    }
    if (_nr > 0){
        for (i = 0; i < kc; i++){
            for (j = 0; j < _nr; j++)
                b_to[j] = B(0, j);
            for (j = _nr; j < NR; j++)
                b_to[j] = 0.0;
            b_to += NR;
            b++;
        }
    }
}

void micro_kernel(int RowC, int ColC, int kc, const float *a, const float *b, float *c, int flag){
    float AB[MR * NR] __attribute__ ((aligned (16)));;
    float32x4_t c_sum_00 = {0};
    float32x4_t c_sum_01 = {0};
    float32x4_t c_sum_02 = {0};
    float32x4_t c_sum_03 = {0};
    float32x4_t c_sum_04 = {0};
    float32x4_t c_sum_05 = {0};
    float32x4_t c_sum_06 = {0};
    float32x4_t c_sum_07 = {0};
    float32x4_t c_sum_40 = {0};
    float32x4_t c_sum_41 = {0};
    float32x4_t c_sum_42 = {0};
    float32x4_t c_sum_43 = {0};
    float32x4_t c_sum_44 = {0};
    float32x4_t c_sum_45 = {0};
    float32x4_t c_sum_46 = {0};
    float32x4_t c_sum_47 = {0};
    float32x4_t b_reg_0, b_reg_4;
    float32x4_t a_reg_0, a_reg_4;
    int i, j, l;
    for (l = 0; l < kc; l++){
        a_reg_0 = vld1q_f32(a + 8 * l );
        a_reg_4 = vld1q_f32(a + 8 * l + 4);

        b_reg_0 = vld1q_f32(b + 8 * l );
        b_reg_4 = vld1q_f32(b + 8 * l + 4);

        c_sum_00 = vfmaq_laneq_f32(c_sum_00, a_reg_0, b_reg_0, 0);
        c_sum_01 = vfmaq_laneq_f32(c_sum_01, a_reg_0, b_reg_0, 1);
        c_sum_02 = vfmaq_laneq_f32(c_sum_02, a_reg_0, b_reg_0, 2);
        c_sum_03 = vfmaq_laneq_f32(c_sum_03, a_reg_0, b_reg_0, 3);
        c_sum_04 = vfmaq_laneq_f32(c_sum_04, a_reg_0, b_reg_4, 0);
        c_sum_05 = vfmaq_laneq_f32(c_sum_05, a_reg_0, b_reg_4, 1);
        c_sum_06 = vfmaq_laneq_f32(c_sum_06, a_reg_0, b_reg_4, 2);
        c_sum_07 = vfmaq_laneq_f32(c_sum_07, a_reg_0, b_reg_4, 3);

        c_sum_40 = vfmaq_laneq_f32(c_sum_40, a_reg_4, b_reg_0, 0);
        c_sum_41 = vfmaq_laneq_f32(c_sum_41, a_reg_4, b_reg_0, 1);
        c_sum_42 = vfmaq_laneq_f32(c_sum_42, a_reg_4, b_reg_0, 2);
        c_sum_43 = vfmaq_laneq_f32(c_sum_43, a_reg_4, b_reg_0, 3);
        c_sum_44 = vfmaq_laneq_f32(c_sum_44, a_reg_4, b_reg_4, 0);
        c_sum_45 = vfmaq_laneq_f32(c_sum_45, a_reg_4, b_reg_4, 1);
        c_sum_46 = vfmaq_laneq_f32(c_sum_46, a_reg_4, b_reg_4, 2);
        c_sum_47 = vfmaq_laneq_f32(c_sum_47, a_reg_4, b_reg_4, 3);

    }
    vst1q_f32(AB, c_sum_00);
    vst1q_f32(AB + 4, c_sum_40);
    vst1q_f32(AB + 8, c_sum_01);
    vst1q_f32(AB + 12, c_sum_41);
    vst1q_f32(AB + 16, c_sum_02);
    vst1q_f32(AB + 20, c_sum_42);
    vst1q_f32(AB + 24, c_sum_03);
    vst1q_f32(AB + 28, c_sum_43);
    vst1q_f32(AB + 32, c_sum_04);
    vst1q_f32(AB + 36, c_sum_44);
    vst1q_f32(AB + 40, c_sum_05);
    vst1q_f32(AB + 44, c_sum_45);
    vst1q_f32(AB + 48, c_sum_06);
    vst1q_f32(AB + 52, c_sum_46);
    vst1q_f32(AB + 56, c_sum_07);
    vst1q_f32(AB + 60, c_sum_47);
    
    if (flag == 0){
        for (j = 0; j < NR; j++){
            for (i = 0; i < MR; i++)
                c[i * RowC + j * ColC] = 0.0;
        }
    }
    for (j = 0; j < NR; j++){
        for (i = 0; i < MR; i++)
            c[i * RowC + j * ColC] += AB[i + j * MR];
    }
}

//This is an auxiliary function for moving elements from local _C to matrix C.
static void add_vector(int n, int mr, int nr, const float * c_from, float *c){
    int i, j;
    for (j = 0; j < nr; j++){
        for (i = 0; i < mr; i++)
            C(i, j) += c_from[i + j * MR];
    }
}

//Macro Kernel for the multiplication of blocks of A and B.  
//We assume that these blocks were previously packed to buffers _A and _B.
static void macro_kernel(int n, int mc, int nc, int kc, float *c){
    int mp = (mc + MR - 1) / MR;
    int np = (nc + NR - 1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr, i, j;

    for (j = 0; j < np; j++){
        nr = (j != np - 1 || _nr == 0) ? NR : _nr;
        for (i = 0; i < mp; i++){
            mr = (i != mp - 1 || _mr == 0) ? MR : _mr;
            if (mr == MR && nr == NR)
                //If there is no padding, we just store the results in C.
                micro_kernel(1, n, kc, &_A[i * kc * MR], &_B[j * kc * NR], &C(i * MR, j * NR), 1);
            else{
                //Else, we have to store the results in local _C temporarily,
                micro_kernel(1, MR, kc, &_A[i * kc * MR], &_B[j * kc * NR], _C, 0);
                //then move them into C.
                add_vector(n, mr, nr, _C, &C(i * MR, j * NR));
            }
        }
    }
}

void square_gemm(int n, float* a, float* b, float* c){
    int m, k;
    m = k = n;

    int mb = (m + MC - 1) / MC;
    int nb = (n + NC - 1) / NC;
    int kb = (k + KC - 1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    for (j = 0; j < nb; j++){
        nc = (j != nb - 1 || _nc == 0) ? NC : _nc;
        for (l = 0; l < kb; l++){
            kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
            pack_B(n, kc, nc, &B(l * KC, j * NC), _B);
            for (i = 0; i < mb; i++){
                mc = (i != mb - 1 || _mc==0) ? MC : _mc;
                pack_A(n, mc, kc, &A(i * MC, l * KC), _A);
                macro_kernel(n, mc, nc, kc, &C(i * MC, j * NC));
            }
        }
    }
}

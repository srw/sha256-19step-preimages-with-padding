#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;

static inline u32 rotr(u32 x, int n){ return (x >> n) | (x << (32 - n)); }
static inline u32 s0(u32 x){ return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
static inline u32 Qf(u32 u){ return s0(u) - u; }

static u32 *H_pack4 = NULL;
static u32 S0_LO[65536];
static u32 REC_HI[65536];
static u16 REC_HI16[65536];

static double now_sec(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline int lookup_pack4_orig(u32 y, u32 *out){
    u32 pack = H_pack4[y >> 2];
    if(!pack) return 0;
    int k = 0;
    u32 y17 = y & 0x1ffffu;
    u32 ulo0 = pack & 0xffffu;
    if(ulo0){
        u32 idx0 = (((ulo0 + y17) ^ S0_LO[ulo0]) >> 1) & 0xffffu;
        u32 u0 = ulo0 | REC_HI[idx0];
        if(Qf(u0) == y) out[k++] = u0;
    }
    u32 ulo1 = pack >> 16;
    if(ulo1){
        u32 idx1 = (((ulo1 + y17) ^ S0_LO[ulo1]) >> 1) & 0xffffu;
        u32 u1 = ulo1 | REC_HI[idx1];
        if(Qf(u1) == y) out[k++] = u1;
    }
    return k;
}

static inline int lookup_pack4_fast(u32 y, u32 *out){
    u32 pack = H_pack4[y >> 2];
    if(!pack) return 0;
    int k = 0;
    u32 ulo0 = pack & 0xffffu;
    if(ulo0){
        u32 v0 = (ulo0 + y) ^ s0(ulo0);
        u32 u0 = ulo0 | ((u32)REC_HI16[(v0 >> 1) & 0xffffu] << 16);
        if(Qf(u0) == y) out[k++] = u0;
    }
    u32 ulo1 = pack >> 16;
    if(ulo1){
        u32 v1 = (ulo1 + y) ^ s0(ulo1);
        u32 u1 = ulo1 | ((u32)REC_HI16[(v1 >> 1) & 0xffffu] << 16);
        if(Qf(u1) == y) out[k++] = u1;
    }
    return k;
}

int main(void){
    int fd = open("/tmp/q0_u16_pack2.bin", O_RDONLY);
    if(fd < 0){ perror("open"); return 1; }
    H_pack4 = mmap(NULL, (1ULL << 30) * sizeof(u32), PROT_READ, MAP_SHARED, fd, 0);
    if(H_pack4 == MAP_FAILED){ perror("mmap"); return 1; }
    madvise(H_pack4, (1ULL << 30) * sizeof(u32), MADV_WILLNEED);

    for(u32 ulo = 0; ulo < 65536; ulo++) S0_LO[ulo] = s0(ulo) & 0x1ffffu;
    for(u32 uhi = 0; uhi < 65536; uhi++){
        u32 v17 = (s0(uhi << 16) & 0x1ffffu) ^ ((uhi & 1u) << 16);
        REC_HI[(v17 >> 1) & 0xffffu] = uhi << 16;
        REC_HI16[(v17 >> 1) & 0xffffu] = (u16)uhi;
    }

    // Touch pages sequentially to warm RAM
    u64 warm = 0;
    for(u64 idx = 0; idx < (1ULL << 30); idx += 1024) warm ^= H_pack4[idx];

    const u32 N = 1U << 24; // 16.77M lookups
    u32 cand[4];

    u64 sum1 = 0, cnt1 = 0;
    double t0 = now_sec();
    for(u32 i = 0; i < N; i += 64){
        u32 ys[64];
        for(int j = 0; j < 64; j++){
            u32 y = (i + j) * 0x9E3779B1u;
            ys[j] = y;
            __builtin_prefetch(&H_pack4[y >> 2], 0, 0);
        }
        for(int j = 0; j < 64; j++){
            int n = lookup_pack4_orig(ys[j], cand);
            cnt1 += n;
            if(n) sum1 ^= cand[0];
        }
    }
    double dt1 = now_sec() - t0;

    u64 sum2 = 0, cnt2 = 0;
    double t1 = now_sec();
    for(u32 i = 0; i < N; i += 64){
        u32 ys[64];
        for(int j = 0; j < 64; j++){
            u32 y = (i + j) * 0x9E3779B1u;
            ys[j] = y;
            __builtin_prefetch(&H_pack4[y >> 2], 0, 0);
        }
        for(int j = 0; j < 64; j++){
            int n = lookup_pack4_fast(ys[j], cand);
            cnt2 += n;
            if(n) sum2 ^= cand[0];
        }
    }
    double dt2 = now_sec() - t1;

    printf("Orig lookup_pack4: %.3f s (cnt=%llu sum=%llx, warm=%llx)\n", dt1, (unsigned long long)cnt1, (unsigned long long)sum1, (unsigned long long)warm);
    printf("Fast lookup_pack4: %.3f s (cnt=%llu sum=%llx) -> %.2fx speedup\n", dt2, (unsigned long long)cnt2, (unsigned long long)sum2, dt1 / dt2);
    return 0;
}

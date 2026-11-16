// rtm_gpu.cu : propagation d'onde 3D d'ordre 8 sur GPU, trois versions.
//
//   naive    un thread par point, les 25 valeurs lues en memoire globale
//            (on compte sur les caches L1/L2 pour la reutilisation)
//   shared   blocs 2D (32 x 8) qui avancent le long de z : le plan courant
//            est en memoire partagee (avec un halo de 4), les 8 voisins en z
//            sont gardes dans des registres et decales d'un cran a chaque plan.
//            Chaque valeur de p n'est lue qu'environ une fois en memoire globale.
//   shared-AxB  meme noyau avec des tuiles A x B (balayage de la taille de tuile)
//   shared-zs, shared-64x4-zs  tuiles + decoupage en z (assez de blocs sur
//            les petits domaines)
//   sharedp  idem, avec des lectures par le cache en lecture seule (__ldg)
//            et une boucle sur z deroulee
//
// Usage : rtm_gpu <naive|shared|sharedp|check> <nx> <ny> <nz> <steps> <reps>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rtm.h"

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

__constant__ float c_coef[R + 1];

// ------------------------------------------------------------- noyaux

__global__ void k_naive(int nx, int ny, int nz, int NX, int NY, const float* __restrict__ p,
                        const float* __restrict__ prev, const float* __restrict__ vel2, float* __restrict__ next) {
    int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y,
        k = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    size_t sx = 1, sy = NX, sz = (size_t)NX * NY;
    size_t id = (k + R) * sz + (j + R) * sy + (i + R);
    float lap = 3.0f * c_coef[0] * p[id];
#pragma unroll
    for (int r = 1; r <= R; r++)
        lap += c_coef[r] * (p[id - r * sx] + p[id + r * sx] + p[id - r * sy] + p[id + r * sy] + p[id - r * sz] +
                            p[id + r * sz]);
    next[id] = 2.0f * p[id] - prev[id] + vel2[id] * lap;
}

// Version "2.5D" (P. Micikevicius, 2009) : grille de blocs en (x, y), chaque
// bloc avance le long de z sur les plans [z0, z0 + zc). Avec zc = nz, la grille
// est 2D ; en coupant z en morceaux (blockIdx.z), on cree assez de blocs pour
// occuper tous les SM sur les petits domaines, au prix de 2R plans relus par
// morceau.
template <int BX, int BY, bool LDG>
__global__ void __launch_bounds__(BX * BY) k_shared(int nx, int ny, int nz, int zc, int NX, int NY,
                         const float* __restrict__ p, const float* __restrict__ prev,
                         const float* __restrict__ vel2, float* __restrict__ next) {
    __shared__ float tile[BY + 2 * R][BX + 2 * R];
    int i = blockIdx.x * BX + threadIdx.x, j = blockIdx.y * BY + threadIdx.y;
    int tx = threadIdx.x + R, ty = threadIdx.y + R;
    size_t sy = NX, sz = (size_t)NX * NY;
    int z0 = blockIdx.z * zc, z1 = min(z0 + zc, nz);
    size_t id = (z0 + R) * sz + (j + R) * sy + (i + R); // premier plan du morceau
    auto ld = [&](size_t q) { return LDG ? __ldg(p + q) : p[q]; };

    float behind[R], infront[R], cur;                   // behind[r-1] = p(z-r), infront[r-1] = p(z+r)
#pragma unroll
    for (int r = 0; r < R; r++) { behind[r] = ld(id - (r + 1) * sz); infront[r] = ld(id + (r + 1) * sz); }
    cur = ld(id);

#pragma unroll 2
    for (int k = z0; k < z1; k++) {
        __syncthreads();                                // le plan precedent n'est plus lu
        tile[ty][tx] = cur;
        if (threadIdx.x < R) { tile[ty][tx - R] = ld(id - R); tile[ty][tx + BX] = ld(id + BX); }
        if (threadIdx.y < R) { tile[ty - R][tx] = ld(id - R * sy); tile[ty + BY][tx] = ld(id + BY * sy); }
        __syncthreads();

        float lap = 3.0f * c_coef[0] * cur;
#pragma unroll
        for (int r = 1; r <= R; r++)
            lap += c_coef[r] * (tile[ty][tx - r] + tile[ty][tx + r] + tile[ty - r][tx] + tile[ty + r][tx] +
                                behind[r - 1] + infront[r - 1]);
        next[id] = 2.0f * cur - prev[id] + vel2[id] * lap;

        // decalage d'un plan vers le haut : seul le plan z+R+1 est lu
#pragma unroll
        for (int r = R - 1; r > 0; r--) behind[r] = behind[r - 1];
        behind[0] = cur;
        cur = infront[0];
#pragma unroll
        for (int r = 0; r < R - 1; r++) infront[r] = infront[r + 1];
        if (k < z1 - 1) infront[R - 1] = ld(id + (R + 1) * sz);
        id += sz;
    }
}

__global__ void k_source(float* p, size_t id, float val) { p[id] += val; }

// ------------------------------------------------------------ pilote

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Sim {
    Grid g; float dt, h = 10.0f, f0 = 15.0f;
    float *p0, *p1, *vel2;                              // p(t-dt) et p(t) ; le resultat ecrase p(t-dt)
    size_t src;
};

static void step(const std::string& v, Sim& s, int it) {
    const Grid& g = s.g;
    if (v == "naive") {
        dim3 b(32, 4, 2), gr((g.nx + 31) / 32, (g.ny + 3) / 4, (g.nz + 1) / 2);
        k_naive<<<gr, b>>>(g.nx, g.ny, g.nz, g.NX, g.NY, s.p1, s.p0, s.vel2, s.p0);
    } else {
        // "-zs" : on coupe z pour viser au moins ~2000 blocs (132 SM sur le H100)
        auto zchunk = [&](int bx, int by) {
            if (v.size() < 3 || v.compare(v.size() - 3, 3, "-zs") != 0) return g.nz;
            int b2 = (g.nx / bx) * (g.ny / by);
            int nch = std::max(1, std::min(g.nz / 16, (2048 + b2 - 1) / b2));
            return (g.nz + nch - 1) / nch;
        };
#define LAUNCH(BXV, BYV, LDGV) do { int zc = zchunk(BXV, BYV);                                    \
        k_shared<BXV, BYV, LDGV><<<dim3(g.nx / BXV, g.ny / BYV, (g.nz + zc - 1) / zc), dim3(BXV, BYV)>>>( \
            g.nx, g.ny, g.nz, zc, g.NX, g.NY, s.p1, s.p0, s.vel2, s.p0); } while (0)
        if (v == "shared" || v == "shared-zs") LAUNCH(32, 8, false);
        else if (v == "sharedp") LAUNCH(32, 8, true);
        else if (v == "shared-32x16") LAUNCH(32, 16, false);
        else if (v == "shared-32x4") LAUNCH(32, 4, false);
        else if (v == "shared-64x4" || v == "shared-64x4-zs") LAUNCH(64, 4, false);
        else if (v == "shared-64x8") LAUNCH(64, 8, false);
        else { fprintf(stderr, "version inconnue : %s\n", v.c_str()); exit(1); }
    }
    // ecriture en place : next[id] ne depend que de prev[id] au meme indice,
    // donc p0 peut recevoir p(t+dt) sans tampon supplementaire
    k_source<<<1, 1>>>(s.p0, s.src, ricker(it * s.dt, s.f0) * s.dt * s.dt);
    std::swap(s.p0, s.p1);
}

static void init(Sim& s, int nx, int ny, int nz) {
    s.g = make_grid(nx, ny, nz);
    size_t n = s.g.size();
    std::vector<float> v(n);
    make_model(s.g, s.h, v.data(), s.dt);
    CK(cudaMalloc(&s.p0, n * 4)); CK(cudaMalloc(&s.p1, n * 4)); CK(cudaMalloc(&s.vel2, n * 4));
    CK(cudaMemset(s.p0, 0, n * 4)); CK(cudaMemset(s.p1, 0, n * 4));
    CK(cudaMemcpy(s.vel2, v.data(), n * 4, cudaMemcpyHostToDevice));
    s.src = s.g.at(nx / 2, ny / 2, nz / 4);
}

// verification : chaque version contre une reference sequentielle sur l'hote
static int check(int nx, int ny, int nz, int steps) {
    Grid g = make_grid(nx, ny, nz); size_t n = g.size();
    std::vector<float> a(n, 0.f), b(n, 0.f), v(n), out(n);
    float dt; make_model(g, 10.0f, v.data(), dt);
    size_t src = g.at(nx / 2, ny / 2, nz / 4);
    for (int it = 0; it < steps; it++) {                // p0 = a (t-dt), p1 = b (t)
        for (int k = 0; k < nz; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    size_t id = g.at(i, j, k);
                    a[id] = update_point(b.data(), a.data(), v.data(), id, g.NX, (size_t)g.NX * g.NY);
                }
        a[src] += ricker(it * dt, 15.0f) * dt * dt;
        std::swap(a, b);
    }
    float ref = 0; for (float x : b) ref = std::max(ref, std::fabs(x));
    for (std::string ver : {"naive", "shared", "sharedp", "shared-zs", "shared-64x4-zs"}) {
        Sim s; init(s, nx, ny, nz);
        for (int it = 0; it < steps; it++) step(ver, s, it);
        CK(cudaMemcpy(out.data(), s.p1, n * 4, cudaMemcpyDeviceToHost));
        float e = 0; for (size_t q = 0; q < n; q++) e = std::max(e, std::fabs(out[q] - b[q]));
        printf("check,%s,%d,%d,%d,%d,max|p|=%.3e,max_err=%.3e,rel=%.2e\n", ver.c_str(), nx, ny, nz, steps, ref, e, e / ref);
        CK(cudaFree(s.p0)); CK(cudaFree(s.p1)); CK(cudaFree(s.vel2));
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <naive|shared|sharedp|check> nx ny nz steps reps\n", argv[0]); return 1; }
    std::string v = argv[1];
    int nx = atoi(argv[2]), ny = atoi(argv[3]), nz = atoi(argv[4]), steps = atoi(argv[5]), reps = atoi(argv[6]);
    float hc[R + 1] = {C0, C1, C2, C3, C4};
    CK(cudaMemcpyToSymbol(c_coef, hc, sizeof(hc)));
    if (v == "check") return check(nx, ny, nz, steps);
    if (v != "naive" && (nx % 64 || ny % 16)) { fprintf(stderr, "nx doit etre multiple de 64 et ny de 16\n"); return 1; }

    Sim s; init(s, nx, ny, nz);
    for (int it = 0; it < 10; it++) step(v, s, it);    // echauffement
    CK(cudaDeviceSynchronize());
    double pts = (double)nx * ny * nz;
    for (int rep = 0; rep < reps; rep++) {
        double t0 = now();
        for (int it = 0; it < steps; it++) step(v, s, it);
        CK(cudaDeviceSynchronize());
        double t = now() - t0;
        // octets ideaux par point : lire p, p(t-dt), v^2 et ecrire p(t+dt) = 16 octets
        printf("%s,%d,%d,%d,%d,%d,%.6e,%.1f,%.1f\n", v.c_str(), nx, ny, nz, steps, rep, t, pts * steps / t * 1e-6,
               16.0 * pts * steps / t * 1e-9);
        fflush(stdout);
    }
    return 0;
}

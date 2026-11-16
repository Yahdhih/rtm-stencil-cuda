// rtm_cpu.cpp : meme noyau sur le CPU Grace (72 coeurs Neoverse V2), OpenMP.
// Boucles k, j paralleles, boucle i vectorisee (SVE/NEON) : c'est la reference
// CPU face a laquelle on situe le gain du GPU.
//
// Deux versions :
//   flat     boucles (k, j) fusionnees et reparties en blocs statiques. Chaque
//            thread parcourt de longues bandes de lignes : pour reutiliser les
//            voisins en z, il lui faudrait ~9 plans en cache (~9 Mo a 512^2).
//   blocked  taches (morceau de z, bloc de JB lignes) ; dans une tache on
//            avance en z, le jeu de travail est 9 plans x JB lignes (~300 Ko
//            pour JB = 16), qui tient dans le L2 (1 Mo par coeur sur Neoverse V2).
// Usage : OMP_NUM_THREADS=72 OMP_PROC_BIND=close rtm_cpu <flat|blocked> nx ny nz steps reps
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <vector>

#include "rtm.h"

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <flat|blocked> nx ny nz steps reps\n", argv[0]); return 1; }
    const bool blocked = !strcmp(argv[1], "blocked");
    int nx = atoi(argv[2]), ny = atoi(argv[3]), nz = atoi(argv[4]), steps = atoi(argv[5]), reps = atoi(argv[6]);
    const int JB = 16, KB = 64;
    Grid g = make_grid(nx, ny, nz); size_t n = g.size();
    float *p0 = (float*)aligned_alloc(64, n * 4), *p1 = (float*)aligned_alloc(64, n * 4),
          *v = (float*)aligned_alloc(64, n * 4);
    float dt; make_model(g, 10.0f, v, dt);
    // premier contact parallele : les pages sont placees pres des threads qui les utilisent
#pragma omp parallel for
    for (size_t q = 0; q < n; q++) { p0[q] = 0.f; p1[q] = 0.f; }
    const size_t sx = g.NX, sy = (size_t)g.NX * g.NY;
    size_t src = g.at(nx / 2, ny / 2, nz / 4);
    const float c1 = C1, c2 = C2, c3 = C3, c4 = C4, c0 = 3.0f * C0;

    auto row = [&](int j, int k) {                       // une ligne de x, vectorisee
        const size_t base = g.at(0, j, k);
        const float* p = p1 + base; float* q = p0 + base; const float* vv = v + base;
#pragma omp simd
        for (int i = 0; i < nx; i++) {
            float lap = c0 * p[i]
                + c1 * (p[i - 1] + p[i + 1] + p[i - sx] + p[i + sx] + p[i - sy] + p[i + sy])
                + c2 * (p[i - 2] + p[i + 2] + p[i - 2 * sx] + p[i + 2 * sx] + p[i - 2 * sy] + p[i + 2 * sy])
                + c3 * (p[i - 3] + p[i + 3] + p[i - 3 * sx] + p[i + 3 * sx] + p[i - 3 * sy] + p[i + 3 * sy])
                + c4 * (p[i - 4] + p[i + 4] + p[i - 4 * sx] + p[i + 4 * sx] + p[i - 4 * sy] + p[i + 4 * sy]);
            q[i] = 2.0f * p[i] - q[i] + vv[i] * lap;
        }
    };
    auto step = [&](int it) {
        if (blocked) {
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
            for (int kb = 0; kb < nz; kb += KB)
                for (int jb = 0; jb < ny; jb += JB)
                    for (int k = kb; k < std::min(kb + KB, nz); k++)
                        for (int j = jb; j < std::min(jb + JB, ny); j++) row(j, k);
        } else {
#pragma omp parallel for collapse(2) schedule(static)
            for (int k = 0; k < nz; k++)
                for (int j = 0; j < ny; j++) row(j, k);
        }
        p0[src] += ricker(it * dt, 15.0f) * dt * dt;
        std::swap(p0, p1);
    };
    for (int it = 0; it < 5; it++) step(it);
    double pts = (double)nx * ny * nz;
    for (int rep = 0; rep < reps; rep++) {
        double t0 = omp_get_wtime();
        for (int it = 0; it < steps; it++) step(it);
        double t = omp_get_wtime() - t0;
        printf("cpu-%s%d,%d,%d,%d,%d,%d,%.6e,%.1f,%.1f\n", blocked ? "blocked" : "flat", omp_get_max_threads(), nx, ny, nz, steps, rep, t,
               pts * steps / t * 1e-6, 16.0 * pts * steps / t * 1e-9);
        fflush(stdout);
    }
    return 0;
}

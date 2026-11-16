// rtm.h : definitions communes au noyau de propagation d'onde acoustique
// (le coeur d'une migration par retournement temporel, RTM).
//
//   p(t+dt) = 2 p(t) - p(t-dt) + v^2 dt^2 Lap p(t)
//
// Laplacien d'ordre 8 en espace (rayon R = 4) : 25 points par mise a jour.
// Simple precision, comme dans les codes sismiques de production.
// Tableaux avec un halo de R cellules nulles de chaque cote (Dirichlet) :
// aucune condition au bord dans les noyaux. x est la dimension contigue.
#pragma once
#include <cmath>
#include <cstddef>

#define R 4
// coefficients centres d'ordre 8 de la derivee seconde
static const float C0 = -205.0f / 72.0f, C1 = 8.0f / 5.0f, C2 = -1.0f / 5.0f, C3 = 8.0f / 315.0f,
                   C4 = -1.0f / 560.0f;

struct Grid {
    int nx, ny, nz;                  // points interieurs
    int NX, NY, NZ;                  // avec halo
    size_t size() const { return (size_t)NX * NY * NZ; }
    size_t at(int i, int j, int k) const { return ((size_t)(k + R) * NY + (j + R)) * NX + (i + R); }
};
static inline Grid make_grid(int nx, int ny, int nz) { return {nx, ny, nz, nx + 2 * R, ny + 2 * R, nz + 2 * R}; }

// ondelette de Ricker (frequence dominante f0) : la source sismique classique
static inline float ricker(float t, float f0) {
    float a = (float)(M_PI * f0 * (t - 1.2f / f0));
    a *= a;
    return (1.0f - 2.0f * a) * std::exp(-a);
}

// Modele de vitesse : deux couches (1500 m/s puis 2500 m/s) ; on stocke
// directement v^2 dt^2 / h^2. Pas de temps choisi a 40 % de la limite CFL.
static inline void make_model(const Grid& g, float h, float* vel2, float& dt) {
    const float vmax = 2500.0f;
    dt = 0.4f * h / (vmax * std::sqrt(3.0f));
    for (int k = -R; k < g.nz + R; k++)
        for (int j = -R; j < g.ny + R; j++)
            for (int i = -R; i < g.nx + R; i++) {
                float v = (k < g.nz / 2) ? 1500.0f : vmax;
                vel2[((size_t)(k + R) * g.NY + (j + R)) * g.NX + (i + R)] = v * v * dt * dt / (h * h);
            }
}

// une mise a jour d'un point, version de reference (hote)
static inline float update_point(const float* p, const float* prev, const float* vel2, size_t id, size_t sx,
                                 size_t sy) {
    const float c[5] = {C0, C1, C2, C3, C4};
    float lap = 3.0f * c[0] * p[id];
    for (int r = 1; r <= R; r++)
        lap += c[r] * (p[id - r] + p[id + r] + p[id - r * sx] + p[id + r * sx] + p[id - r * sy] + p[id + r * sy]);
    return 2.0f * p[id] - prev[id] + vel2[id] * lap;
}

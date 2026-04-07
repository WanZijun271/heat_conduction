#ifndef CONFIG_H
#define CONFIG_H

#include "types.h"

constexpr int dim = 2;

constexpr int nx = 256;
constexpr int ny = 256;
constexpr int nz = 1;

constexpr scalar xmin = 0.0;
constexpr scalar xmax = 0.8333;
constexpr scalar ymin = 0.0;
constexpr scalar ymax = 0.833;
constexpr scalar zmin = 0.0;
constexpr scalar zmax = 1.0;

constexpr scalar k = 81.0;                // thermal conductivity 热导率
constexpr scalar cp = 1.0;                // specific heat capacity 比热容
constexpr scalar rho = 1.0;               // density 密度
constexpr scalar kappa = k / cp / rho;    // thermal diffusivity 热扩散率

// boundary condition
constexpr scalar tBC[6] = {
    373.0,    // east
    373.0,    // west
    293.0,    // north
    373.0,    // south
    0.0,      // top
    0.0       // bottom
};

constexpr int niter = 100000;      // iteration times 迭代次数
constexpr scalar relax = 0.75;     // 松弛因子
constexpr scalar tol = 1e-6;       // tolerance of relative residual

#endif
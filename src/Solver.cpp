#include "Solver.h"
#include <cmath>
#include <fstream>

using namespace std;

constexpr int east   = 0;
constexpr int west   = 1;
constexpr int north  = 2;
constexpr int south  = 3;
constexpr int top    = 4;
constexpr int bottom = 5;

constexpr int b  = 0;
constexpr int aP = 1;
constexpr int aE = 2;
constexpr int aW = 3;
constexpr int aN = 4;
constexpr int aS = 5;
constexpr int aT = 6;
constexpr int aB = 7;

Solver::Solver() {
    _t.assign(nx * ny * nz, 0);     // initialize temperature field

    // initialize coefficient
    if (dim == 2) {
        _coef.assign(nx * ny * 6, 0);    // allocate memory
        calcCoef2D();                    // calculate coefficient
    } else if (dim == 3) {
        _coef.assign(nx * ny * nz * 8, 0);
    }
}

void Solver::initTempField(scalar t) {
    _t.assign(nx * ny * nz, t);
}

void Solver::JacobiSolver() {
    vector<scalar> t0(_t.size(), 0);
    scalar maxNorm = -1e20;
    for (int it = 0; it < niter; ++it) {
        t0.swap(_t);
        scalar norm;
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                for (int k = 0; k < nz; ++k) {
                    // tP
                    int idP = i + j * nx + k * nx * ny;
                    scalar tP = t0[idP];
                    // tE
                    scalar tE = 0.0;
                    if ( i != nx - 1 ) {
                        int idE = (i+1) + j * nx + k * nx * ny;
                        tE = t0[idE];
                    }
                    // tW
                    scalar tW = 0.0;
                    if ( i != 0 ) {
                        int idW = (i-1) + j * nx + k * nx * ny;
                        tW = t0[idW];
                    }
                    // tN
                    scalar tN = 0.0;
                    if ( j != ny - 1 ) {
                        int idN = i + (j+1) * nx + k * nx * ny;
                        tN = t0[idN];
                    }
                    // tN
                    scalar tS = 0.0;
                    if ( j != 0 ) {
                        int idS = i + (j-1) * nx + k * nx * ny;
                        tS = t0[idS];
                    }
                    // tT
                    scalar tT = 0.0;
                    if ( k != nz - 1 ) {
                        int idT = i + j * nx + (k+1) * nx * ny;
                        tT = t0[idT];
                    }
                    // tB
                    scalar tB = 0.0;
                    if ( k != 0 ) {
                        int idB = i + j * nx + (k-1) * nx * ny;
                        tB = t0[idB];
                    }
                    
                    scalar tNew = tP;
                    if (dim == 2) {
                        int id = i * 6 + j * nx * 6;
                        tNew = _coef[id+b]
                            - _coef[id+aE] * tE
                            - _coef[id+aW] * tW
                            - _coef[id+aN] * tN
                            - _coef[id+aS] * tS;
                        tNew /= _coef[id+aP];
                    } else if (dim == 3) {
                        int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
                        tNew = _coef[id+b]
                            - _coef[id+aE] * tE
                            - _coef[id+aW] * tW
                            - _coef[id+aN] * tN
                            - _coef[id+aS] * tS
                            - _coef[id+aT] * tT
                            - _coef[id+aB] * tB;
                        tNew /= _coef[id+aP];
                    }

                    scalar dT = relax * (tNew - tP);
                    _t[idP] = tP + dT;
                    norm += dT*dT;
                }
            }
        }
        norm = sqrt(norm / (nx * ny * nz));
        maxNorm = max(norm, maxNorm);
        scalar relNorm = norm / (maxNorm + 1e-20);    // relative residual
        if (relNorm < tol) {
            break;
        }
    }
}

void Solver::writeVTK(string filename) const {
    ofstream file(filename);

    // write header
	file << "# vtk DataFile Version 3.0" << endl;
    file << "flash 3d grid and solution" << endl;
    file << "ASCII" << endl;
    file << "DATASET RECTILINEAR_GRID" << endl;

    // write mesh information
    file << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << endl;
    file << "X_COORDINATES " << nx + 1 << " float" << endl;
    for (int i = 0; i <= nx; ++i) {
        file << xmin + (scalar)i * dx << " ";
    }
    file << endl;
    file << "Y_COORDINATES " << ny + 1 << " float" << endl;
    for (int i = 0; i <= ny; ++i) {
        file << ymin + (scalar)i * dy << " ";
    }
    file << endl;
    file << "Z_COORDINATES " << nz + 1 << " float" << endl;
    for (int i = 0; i <= nz; ++i) {
        file << zmin + (scalar)i * dz << " ";
    }
    file << endl;

    // write cell data
    int ncell = nx * ny * nz;
    file << "CELL_DATA " << ncell << endl;

    file << "FIELD FieldData 1" << endl;

    // write temperature data
    file << "t 1 " << ncell << " float" << endl;
    for (scalar t : _t) {
        file << t << " ";
    }
    file << endl;

    file.close();
}

void Solver::calcCoef2D() {
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            int id = i * 6 + j * nx * 6;
            // aE
            if (i == nx - 1) {
                _coef[id+aP] += 2 * kappa * dy / dx;
                _coef[id+b] += 2 * kappa * t_BC[east] * dy / dx;
            }
            else {
                _coef[id+aE] += - kappa * dy / dx;
                _coef[id+aP] += kappa * dy / dx;
            }
            // aW
            if (i == 0) {
                _coef[id+aP] += 2 * kappa * dy / dx;
                _coef[id+b] += 2 * kappa * t_BC[west] * dy / dx;
            }
            else {
                _coef[id+aW] += - kappa * dy / dx;
                _coef[id+aP] += kappa * dy / dx;
            }
            // aN
            if (j == ny - 1) {
                _coef[id+aP] += 2 * kappa * dx / dy;
                _coef[id+b] += 2 * kappa * t_BC[north] * dx / dy;
            }
            else {
                _coef[id+aN] += - kappa * dx / dy;
                _coef[id+aP] += kappa * dx / dy;
            }
            // aS
            if (j == 0) {
                _coef[id+aP] += 2 * kappa * dx / dy;
                _coef[id+b] += 2 * kappa * t_BC[south] * dx / dy;
            }
            else {
                _coef[id+aS] += - kappa * dx / dy;
                _coef[id+aP] += kappa * dx / dy;
            }
        }
    }
}
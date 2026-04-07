#include "Solver.h"
#include "config.h"
#include <cmath>
#include <fstream>

using namespace std;

namespace {
    constexpr int east   = 0;
    constexpr int west   = 1;
    constexpr int north  = 2;
    constexpr int south  = 3;
    constexpr int top    = 4;
    constexpr int bottom = 5;

    constexpr int id_b  = 0;
    constexpr int id_aP = 1;
    constexpr int id_aE = 2;
    constexpr int id_aW = 3;
    constexpr int id_aN = 4;
    constexpr int id_aS = 5;
    constexpr int id_aT = 6;
    constexpr int id_aB = 7;
}

constexpr scalar dx = (xmax - xmin) / (scalar)nx;
constexpr scalar dy = (ymax - ymin) / (scalar)ny;
constexpr scalar dz = (zmax - zmin) / (scalar)nz;

Solver::Solver() {
    _t.assign(nx * ny * nz, 0);     // initialize temperature field

    // initialize coefficient
    if (dim == 2) {
        _coef.assign(nx * ny * 6, 0);    // allocate memory
    } else if (dim == 3) {
        _coef.assign(nx * ny * nz * 8, 0);
    }
    calcCoef();    // calculate coefficient
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
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
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
                        tNew = _coef[id+id_b]
                            - _coef[id+id_aE] * tE
                            - _coef[id+id_aW] * tW
                            - _coef[id+id_aN] * tN
                            - _coef[id+id_aS] * tS;
                        tNew /= _coef[id+id_aP];
                    } else if (dim == 3) {
                        int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
                        tNew = _coef[id+id_b]
                            - _coef[id+id_aE] * tE
                            - _coef[id+id_aW] * tW
                            - _coef[id+id_aN] * tN
                            - _coef[id+id_aS] * tS
                            - _coef[id+id_aT] * tT
                            - _coef[id+id_aB] * tB;
                        tNew /= _coef[id+id_aP];
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

void Solver::GaussSeidelSolver() {
    scalar maxNorm = -1e20;
    for (int it = 0; it < niter; ++it) {
        scalar norm;
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    // tP
                    int idP = i + j * nx + k * nx * ny;
                    scalar tP = _t[idP];
                    // tE
                    scalar tE = 0.0;
                    if ( i != nx - 1 ) {
                        int idE = (i+1) + j * nx + k * nx * ny;
                        tE = _t[idE];
                    }
                    // tW
                    scalar tW = 0.0;
                    if ( i != 0 ) {
                        int idW = (i-1) + j * nx + k * nx * ny;
                        tW = _t[idW];
                    }
                    // tN
                    scalar tN = 0.0;
                    if ( j != ny - 1 ) {
                        int idN = i + (j+1) * nx + k * nx * ny;
                        tN = _t[idN];
                    }
                    // tN
                    scalar tS = 0.0;
                    if ( j != 0 ) {
                        int idS = i + (j-1) * nx + k * nx * ny;
                        tS = _t[idS];
                    }
                    // tT
                    scalar tT = 0.0;
                    if ( k != nz - 1 ) {
                        int idT = i + j * nx + (k+1) * nx * ny;
                        tT = _t[idT];
                    }
                    // tB
                    scalar tB = 0.0;
                    if ( k != 0 ) {
                        int idB = i + j * nx + (k-1) * nx * ny;
                        tB = _t[idB];
                    }
                    
                    scalar tNew = tP;
                    if (dim == 2) {
                        int id = i * 6 + j * nx * 6;
                        tNew = _coef[id+id_b]
                            - _coef[id+id_aE] * tE
                            - _coef[id+id_aW] * tW
                            - _coef[id+id_aN] * tN
                            - _coef[id+id_aS] * tS;
                        tNew /= _coef[id+id_aP];
                    } else if (dim == 3) {
                        int id = i * 8 + j * nx * 8 + k * nx * ny * 8;
                        tNew = _coef[id+id_b]
                            - _coef[id+id_aE] * tE
                            - _coef[id+id_aW] * tW
                            - _coef[id+id_aN] * tN
                            - _coef[id+id_aS] * tS
                            - _coef[id+id_aT] * tT
                            - _coef[id+id_aB] * tB;
                        tNew /= _coef[id+id_aP];
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

void Solver::writeVTK(const string& filename) const {
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
    for (const scalar& t : _t) {
        file << t << " ";
    }
    file << endl;

    file.close();
}

void Solver::calcCoef() {
    scalar aE = - kappa * (dy * dz) / dx;
    scalar aW = - kappa * (dy * dz) / dx;
    scalar aN = - kappa * (dz * dx) / dy;
    scalar aS = - kappa * (dz * dx) / dy;
    scalar aT = - kappa * (dx * dy) / dz;
    scalar aB = - kappa * (dx * dy) / dz;
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                int id;
                if (dim == 2) {
                    id = i * 6 + j * nx * 6;
                } else if (dim == 3) {
                    id = i * 8 + j * nx * 8 + k * nx * ny * 8;
                }
                // aE
                if (i == nx - 1) {
                    _coef[id+id_aP] += -2 * aE;
                    _coef[id+id_b] += -2 * aE * tBC[east];
                }
                else {
                    _coef[id+id_aE] += aE;
                    _coef[id+id_aP] += -aE;
                }
                // aW
                if (i == 0) {
                    _coef[id+id_aP] += -2 * aW;
                    _coef[id+id_b] += -2 * aW * tBC[west];
                }
                else {
                    _coef[id+id_aW] += aW;
                    _coef[id+id_aP] += -aW;
                }
                // aN
                if (j == ny - 1) {
                    _coef[id+id_aP] += -2 * aN;
                    _coef[id+id_b] += -2 * aN * tBC[north];
                }
                else {
                    _coef[id+id_aN] += aN;
                    _coef[id+id_aP] += -aN;
                }
                // aS
                if (j == 0) {
                    _coef[id+id_aP] += -2 * aS;
                    _coef[id+id_b] += -2 * aS * tBC[south];
                }
                else {
                    _coef[id+id_aS] += aS;
                    _coef[id+id_aP] += -aS;
                }

                if (dim == 3) {
                    // aT
                    if (k == nz - 1) {
                        _coef[id+id_aP] += -2 * aT;
                        _coef[id+id_b] += -2 * aT * tBC[top];
                    }
                    else {
                        _coef[id+id_aT] += aT;
                        _coef[id+id_aP] += -aT;
                    }
                    // aB
                    if (k == 0) {
                        _coef[id+id_aP] += -2 * aB;
                        _coef[id+id_b] += -2 * aB * tBC[bottom];
                    }
                    else {
                        _coef[id+id_aB] += aB;
                        _coef[id+id_aP] += -aB;
                    }
                }
            }
        }
    }
}
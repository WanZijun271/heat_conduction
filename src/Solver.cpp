#include "Solver.h"
#include "config.h"
#include "constants.h"
#include "kernels.cuh"
#include <cmath>
#include <fstream>

using namespace std;

Solver::Solver() {
    _tempField.assign(nx * ny * nz, 0);     // initialize temperature field

    // initialize coefficient
    _coef.assign(nx * ny * nz * (2 + 2 * dim), 0);
    calcCoef();    // calculate coefficient
}

void Solver::initTempField(scalar temp) {
    _tempField.assign(nx * ny * nz, temp);
}

void Solver::pointJacobiSolver() {
    pointJacobiIterate(_tempField, _coef);
}

void Solver::GaussSeidelSolver() {
    GaussSeidelIterate(_tempField, _coef);
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
    file << "temperature 1 " << ncell << " float" << endl;
    for (const scalar& temp : _tempField) {
        file << temp << " ";
    }
    file << endl;

    file.close();
}

void Solver::calcCoef() {

    scalar areaE = dy * dz;
    scalar areaW = dy * dz;
    scalar areaN = dz * dx;
    scalar areaS = dz * dx;
    scalar areaT = dx * dy;
    scalar areaB = dx * dy;

    scalar aE = -thermalConductivity * areaE / dx;
    scalar aW = -thermalConductivity * areaW / dx;
    scalar aN = -thermalConductivity * areaN / dy;
    scalar aS = -thermalConductivity * areaS / dy;
    scalar aT = -thermalConductivity * areaT / dz;
    scalar aB = -thermalConductivity * areaB / dz;

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {

                int id;
                if (dim == 2) {
                    id = i * 6 + j * nx * 6;
                } else if (dim == 3) {
                    id = i * 8 + j * nx * 8 + k * nx * ny * 8;
                }

                _coef[id+id_aE] += aE;
                _coef[id+id_aW] += aW;
                _coef[id+id_aN] += aN;
                _coef[id+id_aS] += aS;
                _coef[id+id_aP] += -(aE + aW + aN + aS);
                if (dim == 3) {
                    _coef[id+id_aT] += aT;
                    _coef[id+id_aB] += aB;
                    _coef[id+id_aP] += -(aT + aB);
                }

                // set boundary conditios
                if (i == nx - 1) { // east
                    if (typeBC[east] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aE;
                        _coef[id+id_b] += -2 * aE * valueOfTempBCs[east];
                        _coef[id+id_aE] = 0.0;
                    } else if (typeBC[east] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aE;
                        _coef[id+id_b] += -valueOfTempBCs[east] * areaE;
                        _coef[id+id_aE] = 0.0;
                    }
                }
                if (i == 0) { // west
                    if (typeBC[west] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aW;
                        _coef[id+id_b] += -2 * aW * valueOfTempBCs[west];
                        _coef[id+id_aW] = 0.0;
                    } else if (typeBC[west] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aW;
                        _coef[id+id_b] += -valueOfTempBCs[west] * areaW;
                        _coef[id+id_aW] = 0.0;
                    }
                }
                if (j == ny - 1) { // north
                    if (typeBC[north] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aN;
                        _coef[id+id_b] += -2 * aN * valueOfTempBCs[north];
                        _coef[id+id_aN] = 0.0;
                    } else if (typeBC[north] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aN;
                        _coef[id+id_b] += -valueOfTempBCs[north] * areaN;
                        _coef[id+id_aN] = 0.0;
                    }
                }
                if (j == 0) { // south
                    if (typeBC[south] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aS;
                        _coef[id+id_b] += -2 * aS * valueOfTempBCs[south];
                        _coef[id+id_aS] = 0.0;
                    } else if (typeBC[south] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aS;
                        _coef[id+id_b] += -valueOfTempBCs[south] * areaS;
                        _coef[id+id_aS] = 0.0;
                    }
                }
                if (dim == 3 && k == nz - 1) { // top
                    if (typeBC[top] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aT;
                        _coef[id+id_b] += -2 * aT * valueOfTempBCs[top];
                        _coef[id+id_aT] = 0.0;
                    } else if (typeBC[top] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aT;
                        _coef[id+id_b] += -valueOfTempBCs[top] * areaT;
                        _coef[id+id_aT] = 0.0;
                    }
                }
                if (dim == 3 && k == 0) { // bottom
                    if (typeBC[bottom] == 0) { // "Dirichlet" type
                        _coef[id+id_aP] += -aB;
                        _coef[id+id_b] += -2 * aB * valueOfTempBCs[bottom];
                        _coef[id+id_aB] = 0.0;
                    } else if (typeBC[bottom] == 1) { // "Neumann" type
                        _coef[id+id_aP] += aB;
                        _coef[id+id_b] += -valueOfTempBCs[bottom] * areaB;
                        _coef[id+id_aB] = 0.0;
                    }
                }
            }
        }
    }
}
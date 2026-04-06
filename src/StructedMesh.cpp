#include "StructedMesh.h"
#include <fstream>

using namespace std;
using namespace cfd;

#define b 0
#define aP 1
#define aE 2
#define aW 3
#define aN 4
#define aS 5
#define aT 6
#define aB 7

StructedMesh::StructedMesh(int dim, int nc[], scalar domain[])
    : _dim(dim)
    , _ncx(nc[0]), _ncy(nc[1]), _ncz(nc[2])
    , _nx(nc[0]+1), _ny(nc[1]+1), _nz(nc[2]+1)
    , _xmin(domain[0]), _xmax(domain[1]), _ymin(domain[2]), _ymax(domain[3]), _zmin(domain[4]), _zmax(domain[5])
    , _dx((_xmax - _xmin) / (scalar)_ncx), _dy((_ymax - _ymin) / (scalar)_ncy) , _dz((_zmax - _zmin) / (scalar)_ncz) {

    // create coordinates
    _x.assign(_nx, 0);
    _y.assign(_ny, 0);
    _z.assign(_nz, 0);

    _xc.assign(_ncx, 0);
    _yc.assign(_ncy, 0);
    _zc.assign(_ncz, 0);

    // Mesh generation
    for (int i = 0; i < _nx; ++i) {
        _x[i] = _xmin + (scalar)i * _dx;
    }
    for (int i = 0; i < _ny; ++i) {
        _y[i] = _ymin + (scalar)i * _dy;
    }
    for (int i = 0; i < _nz; ++i) {
        _z[i] = _zmin + (scalar)i * _dz;
    }

    for (int i = 0; i < _ncx; ++i) {
        _xc[i] = 0.5 * (_x[i] + _x[i+1]);
    }
    for (int i = 0; i < _ncy; ++i) {
        _yc[i] = 0.5 * (_y[i] + _y[i+1]);
    }
    for (int i = 0; i < _ncz; ++i) {
        _zc[i] = 0.5 * (_z[i] + _z[i+1]);
    }

    // initial field data
    _t.assign(_ncx * _ncy * _ncz, 0);
    _t0.assign(_ncx * _ncy * _ncz, 0);
}

void StructedMesh::setInitialT(scalar t) {
    _t.assign(_ncx * _ncy * _ncz, t);
}

void StructedMesh::createCoefMeshData() {
    if (_dim == 2) {
        _ncoef = 6;      // aP, aW, aE, aS, aN, bsr
    } else if (_dim == 3) {
        _ncoef = 8;      // aP, aW, aE, aS, aN, aB, aT, bsr
    }

    _ct.assign(_ncx * _ncy * _ncz * _ncoef, 0);
}

void StructedMesh::createSimulationData() {
    
}

void StructedMesh::writeVTKCollocatedTemp(string filename) const {
    ofstream file(filename);

    // write header
	file << "# vtk DataFile Version 3.0" << endl;
    file << "flash 3d grid and solution" << endl;
    file << "ASCII" << endl;
    file << "DATASET RECTILINEAR_GRID" << endl;

    // write mesh information
    file << "DIMENSIONS " << _nx << " " << _ny << " " << _nz << endl;
    file << "X_COORDINATES " << _nx << " float" << endl;
    for (scalar x : _x) {
        file << x << " ";
    }
    file << endl;
    file << "Y_COORDINATES " << _ny << " float" << endl;
    for (scalar y : _y) {
        file << y << " ";
    }
    file << endl;
    file << "Z_COORDINATES " << _nz << " float" << endl;
    for (scalar z : _z) {
        file << z << " ";
    }
    file << endl;

    // write cell data
    int ncell = _ncx * _ncy * _ncz;
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
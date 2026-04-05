#include "pch.h"
#include "StructedMesh.h"

using namespace std;
using namespace MP;

#define b 0
#define aP 1
#define aE 2
#define aW 3
#define aN 4
#define aS 5
#define aT 6
#define aB 7

StructedMesh::StructedMesh(int dim, int ncx, int ncy, int ncz) : _dim(dim), _ncx(ncx), _ncy(ncy), _ncz(ncz) {
    if (_dim == 2) {
        _ncz = 1;
    }

    _nx = _ncx + 1, _ny = _ncy + 1, _nz = _ncz + 1;
}

void StructedMesh::createCoordinates(fp xmin, fp xmax, fp ymin, fp ymax, fp zmin, fp zmax) {
    _xmin = xmin, _xmax = xmax, _ymin = ymin, _ymax = ymax, _zmin = zmin, _zmax = zmax;

    if (_dim == 2) {
        _zmin = 0.0;
        _zmax = 1.0;
    }

    _x.assign(_nx, 0);
    _y.assign(_ny, 0);
    _z.assign(_nz, 0);

    _xc.assign(_ncx, 0);
    _yc.assign(_ncy, 0);
    _zc.assign(_ncz, 0);

    // Mesh generation
    _dx = (_xmax - _xmin) / (fp)_ncx;
    for (int i = 0; i < _nx; ++i) {
        _x[i] = _xmin + (fp)i * _dx;
    }
    _dy = (_ymax - _ymin) / (fp)_ncy;
    for (int i = 0; i < _ny; ++i) {
        _y[i] = _ymin + (fp)i * _dy;
    }
    _dz = (_zmax - _zmin) / (fp)_ncz;
    for (int i = 0; i < _nz; ++i) {
        _z[i] = _zmin + (fp)i * _dz;
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
}

void StructedMesh::createFieldMeshData() {
    _t.assign(_ncx * _ncy * _ncz, 0);
    _t0.assign(_ncx * _ncy * _ncz, 0);
}

void StructedMesh::setInitialT(fp t) {
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
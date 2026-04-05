#include "pch.h"
#include "StructedMesh.h"

using namespace MP;

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

    _x = Mat::Zero(_nx, _nx);
    _y = Mat::Zero(_ny, _ny);
    _z = Mat::Zero(_nz, _nz);
}
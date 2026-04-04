#include "StructedMesh.h"

StructedMesh::StructedMesh(int dim, int ncx, int ncy, int ncz) {
    _dim = dim;

    _ncx = ncx, _ncy = ncy, _ncz = ncz;

    if (dim == 2) {
        _ncz = 1;
    }

    _nx = _ncx + 1, _ny = _ncy + 1, _nz = _ncz + 1;
}
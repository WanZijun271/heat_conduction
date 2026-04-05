#ifndef STRUCTEDMESH_H
#define STRUCTEDMESH_H

#include "types.h"

class StructedMesh {
private:
    int _dim;
    int _ncx, _ncy, _ncz;                               // Number of cell in each direction
    int _nx, _ny, _nz;                                  // Number of nodes in each direction
    MP::fp _xmin, _xmax, _ymin, _ymax, _zmin, _zmax;   // Domain coordinates
    MP::Mat _x, _y, _z;                                // Mesh coordinates

public:
    StructedMesh(int dim, int ncx, int ncy, int ncz=1);
    void createCoordinates(MP::fp xmin, MP::fp xmax, MP::fp ymin, MP::fp ymax, MP::fp zmin=0.0, MP::fp zmax=1.0);
};

#endif
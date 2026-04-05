#ifndef STRUCTEDMESH_H
#define STRUCTEDMESH_H

#include "pch.h"
#include "types.h"

class StructedMesh {
private:
    const int _dim;
    const int _ncx, _ncy, _ncz;                                 // number of cell in each direction
    const int _nx, _ny, _nz;                                    // number of nodes in each direction
    const MP::fp _xmin, _xmax, _ymin, _ymax, _zmin, _zmax;      // domain coordinates
    const MP::fp _dx, _dy, _dz;

    std::vector<MP::fp> _x, _y, _z;                             // mesh coordinates
    std::vector<MP::fp> _xc, _yc, _zc;                          // cell-centered mesh coordinates

    std::vector<MP::fp> _t;                                     // temperature fied
    std::vector<MP::fp> _t0;                                    // old temperature fied

    int _ncoef;
    std::vector<MP::fp> _ct;

public:
    StructedMesh(int dim, int nc[], MP::fp domain[]);
    void setInitialT(MP::fp t);
    void createCoefMeshData();
    void writeVTKCollocatedTemp(std::string filename) const;
};

#endif
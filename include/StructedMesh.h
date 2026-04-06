#ifndef STRUCTEDMESH_H
#define STRUCTEDMESH_H

#include "pch.h"
#include "types.h"

class StructedMesh {
private:
    const int _dim;
    const int _ncx, _ncy, _ncz;                                 // number of cell in each direction
    const int _nx, _ny, _nz;                                    // number of nodes in each direction
    const cfd::scalar _xmin, _xmax, _ymin, _ymax, _zmin, _zmax;      // domain coordinates
    const cfd::scalar _dx, _dy, _dz;

    std::vector<cfd::scalar> _x, _y, _z;                             // mesh coordinates
    std::vector<cfd::scalar> _xc, _yc, _zc;                          // cell-centered mesh coordinates

    std::vector<cfd::scalar> _t;                                     // temperature fied
    std::vector<cfd::scalar> _t0;                                    // old temperature fied

    int _ncoef;
    std::vector<cfd::scalar> _ct;

public:
    StructedMesh(int dim, int nc[], cfd::scalar domain[]);
    void setInitialT(cfd::scalar t);
    void createCoefMeshData();
    void createSimulationData();
    void writeVTKCollocatedTemp(std::string filename) const;
};

#endif
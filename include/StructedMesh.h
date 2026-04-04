#ifndef STRUCTMESH_H
#define STRUCTMESH_H

class StructedMesh {
private:
    int _dim;
    int _ncx, _ncy, _ncz; // Number of cell in each direction
    int _nx, _ny, _nz;    // Number of nodes in each direction

public:
    StructedMesh(int dim, int ncx, int ncy, int ncz=1);
};

#endif
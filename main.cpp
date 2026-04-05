#include "pch.h"
#include <iostream>
#include <chrono>
#include "types.h"
#include "StructedMesh.h"

using namespace MP;

int dim = 2;

int ncx = 64;
int ncy = 64;
int ncz = 1;

fp xmin = 0.0;
fp xmax = 0.833;
fp ymin = 0.0;
fp ymax = 0.83;
fp zmin = 0.0;
fp zmax = 1.0;

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    StructedMesh case_(dim, ncx, ncy, ncz);
    case_.createCoordinates(xmin, xmax, ymin, ymax, zmin, zmax);
    case_.createFieldMeshData();
    case_.createCoefMeshData();

    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Elapsed time: " << duration_ms.count() << " ms\n";
    return 0;
}
#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <string>
#include "types.h"
#include "config.h"

class Solver {
private:
    std::vector<scalar> _t;        // temperature field

    std::vector<scalar> _coef;     // coefficient

public:
    Solver();
    void initTempField(scalar t);
    void JacobiSolver();
    void writeVTK(std::string filename) const;
private:
    void calcCoef2D();
};

#endif
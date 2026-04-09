#ifndef KERNELS_H
#define KERNELS_H

#include "types.h"
#include <vector>

void JacobiIterate(std::vector<scalar>& t, const std::vector<scalar>& t0, const std::vector<scalar>& coef, scalar& norm);

void GaussSeidelIterate(std::vector<scalar>& t, const std::vector<scalar>& coef, scalar& norm);

#endif
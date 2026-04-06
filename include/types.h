#ifndef TYPES_H
#define TYPES_H

#include "pch.h"

namespace cfd {
    using scalar = double;

    using Mat = Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic>;
}

#endif
#ifndef TYPES_H
#define TYPES_H

#include <Eigen/Dense>

namespace MP {
    using fp = double;

    using Mat = Eigen::Matrix<fp, Eigen::Dynamic, Eigen::Dynamic>;
}

#endif
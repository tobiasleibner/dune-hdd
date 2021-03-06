// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include "config.h"

#include "thermalblock.hh"
#include <dune/stuff/common/reenable_warnings.hh> // <- here for the python bindings!


template class ThermalblockExample< Dune::SGrid< 2, 2 > >;

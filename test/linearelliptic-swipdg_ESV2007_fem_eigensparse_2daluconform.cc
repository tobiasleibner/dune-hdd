﻿// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include "config.h"

#include <dune/hdd/playground/linearelliptic/testcases/ESV2007.hh>

#include "linearelliptic-swipdg.hh"

#if HAVE_ALUGRID && HAVE_DUNE_FEM && HAVE_EIGEN

namespace Dune {
namespace HDD {
namespace LinearElliptic {
namespace Tests {


template class EocStudySWIPDG< TestCases::ESV2007< ALUGrid< 2, 2, simplex, conforming > >,
                               1,
                               GDT::ChooseSpaceBackend::fem,
                               Stuff::LA::ChooseBackend::eigen_sparse >;


} // namespace Tests
} // namespace LinearElliptic
} // namespace HDD
} // namespace Dune

#endif // HAVE_ALUGRID && HAVE_DUNE_FEM && HAVE_EIGEN

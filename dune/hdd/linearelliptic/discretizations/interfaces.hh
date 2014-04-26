// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Albrecht
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_LINEARELLIPTIC_DISCRETIZATIONS_INTERFACES_HH
#define DUNE_HDD_LINEARELLIPTIC_DISCRETIZATIONS_INTERFACES_HH

#include <memory>
#include <type_traits>

#include <dune/grid/common/gridview.hh>

#include <dune/stuff/grid/boundaryinfo.hh>
#include <dune/stuff/common/crtp.hh>
#include <dune/stuff/common/logging.hh>

#include <dune/pymor/parameters/base.hh>
#include <dune/pymor/discretizations/interfaces.hh>

#include "../problems/interfaces.hh"

namespace Dune {
namespace HDD {
namespace LinearElliptic {


template< class Traits >
class DiscretizationInterface
  : public Pymor::StationaryDiscretizationInterface< Traits >
{
  typedef Pymor::StationaryDiscretizationInterface< Traits > BaseType;
public:
  typedef typename Traits::derived_type     derived_type;
  typedef typename Traits::GridViewType     GridViewType;
  typedef typename Traits::TestSpaceType    TestSpaceType;
  typedef typename Traits::AnsatzSpaceType  AnsatzSpaceType;

  typedef typename GridViewType::Grid::template Codim< 0 >::Entity EntityType;

  typedef typename GridViewType::ctype    DomainFieldType;
  static const unsigned int               dimDomain = GridViewType::dimension;
  typedef typename Traits::RangeFieldType RangeFieldType;
  static const unsigned int               dimRange = Traits::dimRange;

  typedef Stuff::Grid::BoundaryInfoInterface< typename GridViewType::Intersection >             BoundaryInfoType;
  typedef ProblemInterface< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange >  ProblemType;

private:
  static_assert(std::is_base_of< GridView< typename GridViewType::Traits >, GridViewType >::value,
                "GridViewType has to be derived from GridView!");

public:
  DiscretizationInterface(const Pymor::ParameterType tt = Pymor::ParameterType())
    : BaseType(tt)
  {}

  DiscretizationInterface(const Pymor::Parametric& other)
    : BaseType(other)
  {}

  static const std::string static_id()
  {
    return "hdd.linearelliptic.discretization";
  }

  const std::shared_ptr< const GridViewType >& grid_view() const
  {
    CHECK_CRTP(this->as_imp(*this).grid_view());
    return this->as_imp(*this).grid_view();
  }

  const std::shared_ptr< const TestSpaceType >& test_space() const
  {
    CHECK_CRTP(this->as_imp(*this).test_space());
    return this->as_imp(*this).test_space();
  }

  const std::shared_ptr< const TestSpaceType >& ansatz_space() const
  {
    CHECK_CRTP(this->as_imp(*this).ansatz_space());
    return this->as_imp(*this).ansatz_space();
  }

  const Stuff::Common::ConfigTree& boundary_info_cfg() const
  {
    CHECK_CRTP(this->as_imp(*this).boundary_info_cfg());
    return this->as_imp(*this).boundary_info_cfg();
  }

  const BoundaryInfoType& boundary_info() const
  {
    CHECK_CRTP(this->as_imp(*this).boundary_info());
    return this->as_imp(*this).boundary_info();
  }

  const ProblemType& problem() const
  {
    CHECK_CRTP(this->as_imp(*this).model());
    return this->as_imp(*this).model();
  }

  void init(std::ostream& out = Dune::Stuff::Common::Logger().devnull(), const std::string prefix = "")
  {
    CHECK_AND_CALL_CRTP(this->as_imp(*this).init(out, prefix));
  }
}; // class DiscretizationInterface


} // namespace LinearElliptic
} // namespace HDD
} // namespace Dune

#endif // DUNE_HDD_LINEARELLIPTIC_DISCRETIZATIONS_INTERFACES_HH

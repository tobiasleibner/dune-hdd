// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_HYPERBOLIC_PROBLEMS_INTERFACES_HH
#define DUNE_HDD_HYPERBOLIC_PROBLEMS_INTERFACES_HH

#include <ostream>

#include <dune/grid/yaspgrid.hh>

#include <dune/stuff/common/configuration.hh>
#include <dune/stuff/common/exceptions.hh>
#include <dune/stuff/functions/default.hh>
#include <dune/stuff/functions/expression.hh>

namespace Dune {
namespace HDD {
namespace Hyperbolic {


template < class EntityImp, class DomainFieldImp, int domainDim, class RangeFieldImp, int rangeDim >
class ProblemInterface
{
public:
  ProblemInterface()
  {
    DUNE_THROW(Dune::NotImplemented, "Not implemented for rangeDim != 1");
  }
};


/* Interface for problem of the form delta_t u + div f(u) = q(u) where u: R^d \to R.
 * The flux f is a function f: R \to R^d, and q: R \to R is a source. */
template< class EntityImp, class DomainFieldImp, int domainDim, class RangeFieldImp>
class ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
{
  typedef ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > ThisType;
public:
  typedef EntityImp         EntityType;
  typedef DomainFieldImp    DomainFieldType;
  static const unsigned int dimDomain = domainDim;
  typedef RangeFieldImp     RangeFieldType;
  static const unsigned int dimRange = 1;

//  we do not have a grid for the range space, but still need an EntityType with dimension dimRange for FluxType and
//  SourceType, so just choose an arbitrary one
  typedef typename Dune::YaspGrid< dimRange >::template Codim< 0 >::Entity           FluxSourceEntityType;
  typedef Dune::Stuff::GlobalFunctionInterface< FluxSourceEntityType, RangeFieldType, dimRange, RangeFieldType, dimDomain > FluxType;
  typedef Dune::Stuff::GlobalFunctionInterface< FluxSourceEntityType, RangeFieldType, dimRange, RangeFieldType, dimDomain, dimRange > FluxDerivativeType;
  typedef Dune::Stuff::GlobalFunctionInterface< FluxSourceEntityType, RangeFieldType, dimRange, RangeFieldType, 1 >         SourceType; // depends on u
//  typedef Dune::Stuff::LocalizableFunctionInterface
//      < EntityType, DomainFieldType, 1, RangeFieldType, 1 >          SourceType; // depends on x
  typedef Dune::Stuff::LocalizableFunctionInterface
      < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange >  FunctionType;
  typedef Dune::Stuff::Common::Configuration                                ConfigType;

  virtual ~ProblemInterface() {}

  static std::string static_id()
  {
    return "hdd.hyperbolic.problem";
  }

  virtual std::string type() const
  {
    return "hdd.hyperbolic.problem";
  }

  virtual const std::shared_ptr< const FluxType >& flux() const = 0;

  virtual const std::shared_ptr< const FluxDerivativeType >& flux_derivative() const = 0;

  virtual const std::shared_ptr< const SourceType >& source() const = 0;

  virtual const std::shared_ptr< const FunctionType >& initial_values() const = 0;

  virtual const ConfigType grid_config() const = 0;

  virtual const ConfigType boundary_info() const = 0;

  virtual const std::shared_ptr< const FunctionType >& boundary_values() const = 0;

  template< class G >
  void visualize(/*const GridView< G >& grid_view, std::string filename, other types*/) const
  {
    // TODO: implement
  } // ... visualize(...) const

  virtual void report(std::ostream& out, std::string prefix = "") const
  {
    out << prefix << "problem '" << type() << "':" << std::endl;
    // TODO: implement
  } // ... report(...)

private:
  template< class T >
  friend std::ostream& operator<<(std::ostream& /*out*/, const ThisType& /*problem*/);
}; // ProblemInterface


template< class E, class D, int d, class R, int r >
std::ostream& operator<<(std::ostream& out, const ProblemInterface< E, D, d, R, r >& problem)
{
  problem.report(out);
  return out;
} // ... operator<<(...)


} // namespace Hyperbolic
} // namespace HDD
} // namespace Dune

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_INTERFACES_HH

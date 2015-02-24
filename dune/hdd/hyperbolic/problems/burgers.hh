// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH
#define DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH

#include <memory>
#include <vector>
#include <string>

#include <dune/common/static_assert.hh>

#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>

#include "default.hh"

namespace Dune {
namespace HDD {
namespace Hyperbolic {
namespace Problems {


template< class E, class D, int d, class R, int r = 1 >
class Burgers
{
  Burgers() { static_assert(AlwaysFalse< E >::value, "Not available for these dimensions!"); }
};


template< class EntityImp, class DomainFieldImp, int domainDim, class RangeFieldImp >
class Burgers< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
  : public Default< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
{
  typedef Burgers< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > ThisType;
  typedef Default< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > BaseType;

public:
  typedef typename BaseType::DefaultFluxType            DefaultFluxType;
  typedef typename BaseType::DefaultFluxDerivativeType  DefaultFluxDerivativeType;
  typedef typename BaseType::DefaultFunctionType        DefaultFunctionType;
  typedef typename BaseType::DefaultSourceType          DefaultSourceType;

  typedef typename BaseType::FluxType                   FluxType;
  typedef typename BaseType::FluxDerivativeType         FluxDerivativeType;
  typedef typename BaseType::SourceType                 SourceType;
  typedef typename BaseType::FunctionType               FunctionType;
  typedef typename BaseType::ConfigType                 ConfigType;

  static std::string static_id()
  {
    return BaseType::static_id() + ".burgers";
  }

  std::string type() const
  {
    return BaseType::type() + ".burgers";
  }

protected:
  using BaseType::default_grid_config;
  using BaseType::default_boundary_info_config;

public:
  static std::unique_ptr< ThisType > create(const ConfigType cfg = default_config(),
                                            const std::string sub_name = static_id())
  {
    const ConfigType config = cfg.has_sub(sub_name) ? cfg.sub(sub_name) : cfg;
    const std::shared_ptr< const DefaultSourceType > source(DefaultSourceType::create(config.sub("source")));
    const std::shared_ptr< const DefaultFunctionType > initial_values(DefaultFunctionType::create(config.sub("initial_values")));
    const ConfigType grid_config = config.sub("grid");
    const ConfigType boundary_info = config.sub("boundary_info");
    const std::shared_ptr< const DefaultFunctionType > boundary_values(DefaultFunctionType::create(config.sub("boundary_values")));
    return Stuff::Common::make_unique< ThisType >(source, initial_values,
                                                  grid_config, boundary_info, boundary_values);
  } // ... create(...)

  static ConfigType default_config(const std::string sub_name = "")
  {
    ConfigType config = BaseType::default_config();
    ConfigType flux_config = DefaultFluxType::default_config();
    flux_config["type"] = FluxType::static_id();
    flux_config["variable"] = "u";
    flux_config["expression"] = "[1.0/2.0*u[0]*u[0] 1.0/2.0*u[0]*u[0] 1.0/2.0*u[0]*u[0]]";
    flux_config["order"] = "2";
    config.add(flux_config, "flux", true);
    ConfigType flux_derivative_config = DefaultFluxDerivativeType::default_config();
    flux_derivative_config["type"] = FluxDerivativeType::static_id();
    flux_derivative_config["variable"] = "u";
    flux_derivative_config["expression"] = "[u[0] u[0] u[0]]";
    flux_derivative_config["order"] = "1";
    config.add(flux_derivative_config, "flux_derivative", true);
    if (sub_name.empty())
      return config;
    else {
      ConfigType tmp;
      tmp.add(config, sub_name);
      return tmp;
    }
  } // ... default_config(...)

  Burgers(const std::shared_ptr< const SourceType > source = std::make_shared< DefaultSourceType >("u", "0", 0),
          const std::shared_ptr< const FunctionType > initial_values = std::make_shared< DefaultFunctionType >("x", "sin(pi*x[0])", 10),
          const ConfigType& grid_config = default_grid_config(),
          const ConfigType& boundary_info = default_boundary_info_config(),
          const std::shared_ptr< const FunctionType > boundary_values = std::make_shared< DefaultFunctionType >("x", "0", 0))
    : BaseType(std::make_shared< DefaultFluxType >("u", std::vector< std::string >(3, "1.0/2.0*u[0]*u[0]"), 2),
               std::make_shared< DefaultFluxDerivativeType >("u", std::vector< std::string >(3, "u[0]"), 1),
               source,
               initial_values,
               grid_config,
               boundary_info,
               boundary_values)
  {}
};

} // namespace Problems
} // namespace Hyperbolic
} // namespace HDD
} // namespace Dune

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH

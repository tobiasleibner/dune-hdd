// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_HYPERBOLIC_PROBLEMS_TRANSPORT_HH
#define DUNE_HDD_HYPERBOLIC_PROBLEMS_TRANSPORT_HH

#include <memory>
#include <vector>
#include <string>

#include <dune/common/static_assert.hh>

#include "default.hh"

namespace Dune {
namespace HDD {
namespace Hyperbolic {
namespace Problems {


template< class EntityImp, class DomainFieldImp, size_t domainDim, class RangeFieldImp, size_t rangeDim >
class Transport
  : public Default< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim >
{
  typedef Transport< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim > ThisType;
  typedef Default< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim > BaseType;

public:
  using typename BaseType::DefaultFluxType;
  using typename BaseType::DefaultFunctionType;
  using typename BaseType::DefaultSourceType;
  using typename BaseType::DefaultBoundaryValueType;

  using typename BaseType::FluxType;
  using typename BaseType::SourceType;
  using typename BaseType::FunctionType;
  using typename BaseType::BoundaryValueType;
  using typename BaseType::ConfigType;

  using BaseType::dimDomain;
  using BaseType::dimRange;

  static std::string static_id()
  {
    return BaseType::static_id() + ".transport";
  }

  std::string type() const
  {
    return BaseType::type() + ".transport";
  }

protected:
  static ConfigType default_grid_config()
  {
    ConfigType grid_config;
    grid_config["type"] = "provider.cube";
    grid_config["lower_left"] = "[0.0 0.0 0.0]";
    grid_config["upper_right"] = "[1.0 1.0 1.0]";
    grid_config["num_elements"] = "[500 60 60]";
    return grid_config;
  }

  static ConfigType default_boundary_info_config()
  {
    ConfigType boundary_config;
    boundary_config["type"] = "periodic";
    return boundary_config;
  }

public:
  static std::unique_ptr< ThisType > create(const ConfigType cfg = default_config(),
                                            const std::string sub_name = static_id())
  {
    const ConfigType config = cfg.has_sub(sub_name) ? cfg.sub(sub_name) : cfg;
    const std::shared_ptr< const DefaultFluxType > flux(DefaultFluxType::create(config.sub("flux")));
    const std::shared_ptr< const DefaultSourceType > source(DefaultSourceType::create(config.sub("source")));
    const std::shared_ptr< const DefaultFunctionType > initial_values(DefaultFunctionType::create(config.sub("initial_values")));
    const ConfigType grid_config = config.sub("grid");
    const ConfigType boundary_info = config.sub("boundary_info");
    const std::shared_ptr< const DefaultBoundaryValueType > boundary_values(DefaultBoundaryValueType::create(config.sub("boundary_values")));
    return Stuff::Common::make_unique< ThisType >(flux, source, initial_values,
                                                  grid_config, boundary_info, boundary_values);
  } // ... create(...)

  static ConfigType default_config(const std::string sub_name = "")
  {
    ConfigType config = BaseType::default_config();
    config.add(default_grid_config(), "grid", true);
    config.add(default_boundary_info_config(), "boundary_info", true);
    ConfigType flux_config = DefaultFluxType::default_config();
    flux_config["type"] = DefaultFluxType::static_id();
    flux_config["variable"] = "u";
    flux_config["expression"] = "[u[0] u[0] 3*u[0]; 4*u[0] 5*u[0] 6*u[0]; 7*u[0] 8*u[0] 9*u[0]]";
    flux_config["order"] = "1";
    flux_config["gradient"] = "[1 0 0; 4 0 0; 7 0 0]";
    flux_config["gradient.0"] = "[1 0 0; 4 0 0; 7 0 0]";
    flux_config["gradient.1"] = "[1 0 0; 5 0 0; 8 0 0]";
    flux_config["gradient.2"] = "[3 0 0; 6 0 0; 9 0 0]";
    config.add(flux_config, "flux", true);
    ConfigType initial_value_config = DefaultFunctionType::default_config();
//    initial_value_config["type"] = DefaultFunctionType::static_id();
//    initial_value_config["variable"] = "x";
//    if (dimDomain == 1)
//      initial_value_config["expression"] = "[sin(pi*x[0]) sin(pi*x[0]) sin(pi*x[0])]";            // simple sine wave
//    else
//      initial_value_config["expression"] = "[1.0/40.0*exp(1-(2*pi*x[0]-pi)*(2*pi*x[0]-pi)-(2*pi*x[1]-pi)*(2*pi*x[1]-pi))]"; //bump, only in 2D or higher
//    initial_value_config["order"] = "10";
    initial_value_config["lower_left"] = "[0.0 0.0 0.0]";
    initial_value_config["upper_right"] = "[1.0 1.0 1.0]";
    initial_value_config["num_elements"] = "[5 1 1]";
    initial_value_config["variable"] = "x";
    initial_value_config["values"] = "[0 sin(pi/2+5*pi*(x[0]-0.3))*exp(-(200*(x[0]-0.3)*(x[0]-0.3))) 0 1.0 0.0]";
    initial_value_config["name"] = static_id();
    config.add(initial_value_config, "initial_values", true);
    if (sub_name.empty())
      return config;
    else {
      ConfigType tmp;
      tmp.add(config, sub_name);
      return tmp;
    }
  } // ... default_config(...)

  Transport(const std::shared_ptr< const FluxType > flux,
            const std::shared_ptr< const SourceType > source,
            const std::shared_ptr< const FunctionType > initial_values,
            const ConfigType& grid_config,
            const ConfigType& boundary_info,
            const std::shared_ptr< const BoundaryValueType > boundary_values)
    : BaseType(flux,
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

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_TRANSPORT_HH

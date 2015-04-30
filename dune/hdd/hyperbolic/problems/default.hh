// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_HYPERBOLIC_PROBLEMS_DEFAULT_HH
#define DUNE_HDD_HYPERBOLIC_PROBLEMS_DEFAULT_HH

#include <memory>

#include <dune/common/static_assert.hh>

#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>

#include "interfaces.hh"

namespace Dune {
namespace HDD {
namespace Hyperbolic {
namespace Problems {


template< class EntityImp, class DomainFieldImp, size_t domainDim, class RangeFieldImp, size_t rangeDim >
class Default
  : public ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim >
{
  typedef ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim > BaseType;
  typedef Default< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim > ThisType;
protected:
  using typename BaseType::FluxSourceEntityType;

public:
  using BaseType::dimDomain;
  using BaseType::dimRange;
  typedef typename Dune::Stuff::Functions::Expression
                < FluxSourceEntityType, RangeFieldImp, dimRange, RangeFieldImp, dimRange, dimDomain > DefaultFluxType;
  typedef typename Dune::Stuff::Functions::Expression
                < EntityImp, DomainFieldImp, dimDomain, RangeFieldImp, dimRange, 1 > DefaultFunctionType;
  typedef typename Dune::Stuff::Functions::Expression
                < FluxSourceEntityType, RangeFieldImp, dimRange, DomainFieldImp, dimRange, 1 > DefaultSourceType;

  using typename BaseType::FluxType;
  using typename BaseType::SourceType;
  using typename BaseType::FunctionType;
  using typename BaseType::ConfigType;
  using typename BaseType::RangeFieldType;

  typedef typename DefaultFunctionType::DomainType    DomainType;
  typedef typename DefaultFunctionType::RangeType     RangeType;

  static std::string static_id()
  {
    return BaseType::static_id() + ".default";
  }

  std::string type() const override
  {
    return BaseType::type() + ".default";
  }

protected:
  static ConfigType default_grid_config()
  {
    ConfigType grid_config;
    grid_config["type"] = "provider.cube";
    grid_config["lower_left"] = "[0.0 0.0 0.0 0.0]";
    grid_config["upper_right"] = "[10.0 1.0 1.0 1.0]";
    grid_config["num_elements"] = "[1000 100 60 8]";
    return grid_config;
  }

  static ConfigType default_boundary_info_config()
  {
    ConfigType boundary_config;
    boundary_config["type"] = "periodic";
    return boundary_config;
  }

public:
  static ConfigType default_config(const std::string sub_name = "")
  {
    ConfigType config;
    config.add(default_grid_config(), "grid");
    config.add(default_boundary_info_config(), "boundary_info");
    ConfigType flux_config = DefaultFluxType::default_config();
    flux_config["type"] = FluxType::static_id();
    flux_config["variable"] = "u";
    flux_config["expression"] = "[u[1] u[1]*u[1]/u[0]+9.81*0.5*u[0]*u[0] 2*u[2]]";
    flux_config["order"] = "2";
    flux_config["gradient"] = "[0 1 0; -1*u[1]*u[1]/(u[0]*u[0])+9.81*u[0] 2*u[1]/u[0] 0; 0 0 2]";
    config.add(flux_config, "flux");
    ConfigType source_config = DefaultSourceType::default_config();
    source_config["type"] = SourceType::static_id();
    source_config["variable"] = "u";
    source_config["expression"] = "[0 0 0]";
    source_config["order"] = "1";
    config.add(source_config, "source");
    ConfigType initial_value_config = DefaultFunctionType::default_config();
    initial_value_config["type"] = DefaultFunctionType::static_id();
    initial_value_config["variable"] = "x";
//    initial_value_config["expression"] = "[sin(pi*x[0]) sin(pi*x[0]) sin(pi*x[0])]";                                  // simple sine wave
//    initial_value_config["expression"] = "sin(pi*(x[0]-4)*(x[0]-10))*exp(-(x[0]-8)^4)";     // waves for 1D, domain [0,16] or the like
    initial_value_config["expression"] = "[1+cos(pi/2*(x[0]-5))*exp(-(x[0]-5)^4) 0]";         // bump for shallow water equations, domain [0,10], Leveque p.257
//    initial_value_config["expression"] = "[1.0/40.0*exp(1-(2*pi*x[0]-pi)*(2*pi*x[0]-pi)-(2*pi*x[1]-pi)*(2*pi*x[1]-pi))]"; //bump, only in 2D or higher
    initial_value_config["order"] = "10";
    config.add(initial_value_config, "initial_values");
    ConfigType boundary_value_config = DefaultFunctionType::default_config();
    boundary_value_config["type"] = DefaultFunctionType::static_id();
    boundary_value_config["variable"] = "x";
    boundary_value_config["expression"] = "[0 0 0]";
    boundary_value_config["order"] = "1";
    config.add(boundary_value_config, "boundary_values");
    if (sub_name.empty())
      return config;
    else {
      ConfigType tmp;
      tmp.add(config, sub_name);
      return tmp;
    }
  } // ... default_config(...)

  static std::unique_ptr< ThisType > create(const ConfigType cfg = default_config(),
                                            const std::string sub_name = static_id())
  {
    const ConfigType config = cfg.has_sub(sub_name) ? cfg.sub(sub_name) : cfg;
    const std::shared_ptr< const DefaultFluxType > flux(DefaultFluxType::create(config.sub("flux")));
    const std::shared_ptr< const DefaultSourceType > source(DefaultSourceType::create(config.sub("source")));
    const std::shared_ptr< const DefaultFunctionType > initial_values(DefaultFunctionType::create(config.sub("initial_values")));
    const ConfigType grid_config = config.sub("grid");
    const ConfigType boundary_info = config.sub("boundary_info");
    const std::shared_ptr< const DefaultFunctionType > boundary_values(DefaultFunctionType::create(config.sub("boundary_values")));
    return Stuff::Common::make_unique< ThisType >(flux, source, initial_values,
                                                  grid_config, boundary_info, boundary_values);
  } // ... create(...)

  Default(const std::shared_ptr< const FluxType > flux,
          const std::shared_ptr< const SourceType > source = std::make_shared< DefaultSourceType >("u", "[0 0 0]", 0),
          const std::shared_ptr< const FunctionType > initial_values = std::make_shared< DefaultFunctionType >("x", "[sin(pi*x[0]) sin(pi*x[0]) sin(pi*x[0])]", 10),
          const ConfigType& grid_config = default_grid_config(),
          const ConfigType& boundary_info = default_boundary_info_config(),
          const std::shared_ptr< const FunctionType > boundary_values = std::make_shared< DefaultFunctionType >("x", "[0 0 0]", 0))
    : flux_(flux)
    , source_(source)
    , initial_values_(initial_values)
    , grid_config_(grid_config)
    , boundary_info_(boundary_info)
    , boundary_values_(boundary_values)
  {}

  virtual const std::shared_ptr< const FluxType >& flux() const override
  {
    return flux_;
  }

  virtual const std::shared_ptr< const SourceType >& source() const override
  {
    return source_;
  }

  virtual const std::shared_ptr< const FunctionType >& initial_values() const override
  {
    return initial_values_;
  }

  virtual const ConfigType grid_config() const override
  {
    return grid_config_;
  }

  virtual const ConfigType boundary_info() const override
  {
    return boundary_info_;
  }

  virtual const std::shared_ptr< const FunctionType >& boundary_values() const override
  {
    return boundary_values_;
  }

private:
  const std::shared_ptr< const FluxType >           flux_;
  const std::shared_ptr< const SourceType >         source_;
  const std::shared_ptr< const FunctionType >       initial_values_;
  const ConfigType                                  grid_config_;
  const ConfigType                                  boundary_info_;
  const std::shared_ptr< const FunctionType >       boundary_values_;
};

} // namespace Problems
} // namespace Hyperbolic
} // namespace HDD
} // namespace Dune

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_DEFAULT_HH

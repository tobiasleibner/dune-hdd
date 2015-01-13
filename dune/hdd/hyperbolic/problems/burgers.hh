// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH
#define DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH

#include <memory>

#include <boost/numeric/conversion/cast.hpp>

#include <dune/common/static_assert.hh>
#include <dune/common/timer.hh>

#include <dune/grid/multiscale/provider/cube.hh>

#include <dune/stuff/common/memory.hh>
#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/functions/expression.hh>
#include <dune/stuff/playground/functions/indicator.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/boundaryinfo.hh>

#include <dune/pymor/functions/default.hh>
#include <dune/pymor/functions/checkerboard.hh>

#include "interfaces.hh"

namespace Dune {
namespace HDD {
namespace Hyperbolic {
namespace Problems {


// Default -> default.hh

template< class E, class D, int d, class R, int r = 1 >
class Burgers
{
  Burgers() { static_assert(AlwaysFalse< E >::value, "Not available for these dimensions!"); }
};


template< class EntityImp, class DomainFieldImp, int domainDim, class RangeFieldImp >
class Burgers< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
  : public ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
{
  typedef ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > BaseType;
  typedef Burgers< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > ThisType;

public:
  typedef typename Dune::Stuff::Functions::Expression
                < EntityImp, RangeFieldImp, 1, RangeFieldImp, domainDim > DefaultFluxType;
  typedef typename Dune::Stuff::Functions::Expression
                < EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 > DefaultFunctionType;
  typedef typename Dune::Stuff::Functions::Expression
                < EntityImp, DomainFieldImp, 1, RangeFieldImp, 1, 1 > DefaultSourceType;

  typedef typename BaseType::FluxType          FluxType;
  typedef typename BaseType::SourceType        SourceType;
  typedef typename BaseType::FunctionType      FunctionType;
  typedef typename BaseType::ConfigType        ConfigType;
  typedef typename BaseType::RangeFieldType    RangeFieldType;

//  typedef typename DefaultFunctionType::DomainType    DomainType;

  static std::string static_id()
  {
    return BaseType::static_id() + ".burgers";
  }

//  virtual std::string type() const override
//  {
//    return BaseType::type() + ".burgers";
//  }


  std::string type() const
  {
    return BaseType::type() + ".burgers";
  }

private:

  static ConfigType default_grid_config()
  {
    ConfigType grid_config;
    grid_config["type"] = "provider.cube";
    grid_config["ll"] = "[0, 0]";
    grid_config["ur"] = "[1, 1]";
    grid_config["num_elements"] = "[100 8 8 8]";
    return grid_config;
  }

  static ConfigType default_boundary_info_config()
  {
    ConfigType boundary_config;
    boundary_config["type"] = "dirichlet";
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
    flux_config["expression"] = "1.0/2.0*u[0]*u[0]";
    flux_config["order"] = "2";
    config.add(flux_config, "flux");
    ConfigType source_config = DefaultSourceType::default_config();
    source_config["type"] = SourceType::static_id();
    source_config["variable"] = "u";
    source_config["expression"] = "0";
    source_config["order"] = "0";
    config.add(source_config, "source");
    ConfigType initial_value_config = DefaultFunctionType::default_config();
    initial_value_config["type"] = DefaultFunctionType::static_id();
    initial_value_config["variable"] = "x";
    initial_value_config["expression"] = "sin(pi*x[0])";
    initial_value_config["order"] = "1";
    config.add(initial_value_config, "initial_values");
    ConfigType boundary_value_config = DefaultFunctionType::default_config();
    boundary_value_config["type"] = DefaultFunctionType::static_id();
    boundary_value_config["variable"] = "x";
    boundary_value_config["expression"] = "x[0]";
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

  Burgers(const std::shared_ptr< const FluxType > flux = std::make_shared< DefaultFluxType >("u", "1.0/2.0*u[0]*u[0]", 2),
          const std::shared_ptr< const SourceType > source = std::make_shared< DefaultSourceType >("u", "0", 0),
          const std::shared_ptr< const FunctionType > initial_values = std::make_shared< DefaultFunctionType >("x", "sin(pi*x[0])", 10),
          const ConfigType& grid_config = default_grid_config(),
          const ConfigType& boundary_info = default_boundary_info_config(),
          const std::shared_ptr< const FunctionType > boundary_values = std::make_shared< DefaultFunctionType >("x", "0", 0))
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
  const std::shared_ptr< const FluxType >     flux_;
  const std::shared_ptr< const SourceType >   source_;
  const std::shared_ptr< const FunctionType > initial_values_;
  const ConfigType                            grid_config_;
  const ConfigType                            boundary_info_;
  const std::shared_ptr< const FunctionType > boundary_values_;
};

} // namespace Problems
} // namespace Hyperbolic
} // namespace HDD
} // namespace Dune

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH

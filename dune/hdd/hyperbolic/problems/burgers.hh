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


template< class E, class D, int d, class R, int r = 1 >
class Burgers
  : public ProblemInterface< E, D, d, R, r >
{
  Burgers() { static_assert(AlwaysFalse< E >::value, "Not available for these dimensions!"); }
};


template< class EntityImp, class DomainFieldImp, int domainDim, class RangeFieldImp >
class Burgers< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
  : public ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 >
{
  typedef ProblemInterface< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > BaseType;
  typedef Burgers< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > ThisType;

  typedef typename Dune::Stuff::Functions::Expression
                < EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1, 1 > ExpressionFunctionType;
  typedef typename Dune::Stuff::Functions::Constant
                < EntityImp, DomainFieldImp, dimDomain, RangeFieldImp, 1, 1 > ConstantFunctionType;
  typedef typename Dune::Stuff::Functions::Constant
                < EntityImp, DomainFieldImp, 1, RangeFieldImp, 1, 1 > Constant1dFunctionType;

public:
  typedef typename BaseType::FluxType          FluxType;
  typedef typename BaseType::SourceType        SourceType;
  typedef typename BaseType::FunctionType      FunctionType;
  typedef typename BaseType::ConfigType        ConfigType;
  typedef typename BaseType::BoundaryInfoType  BoundaryInfoType;
  typedef typename BaseType::BoundaryValueType BoundaryValueType;

  static std::string static_id()
  {
    return BaseType::static_id() + ".burgers";
  }

  virtual std::string type() const override
  {
    return BaseType::type() + ".burgers";
  }

  static Stuff::Common::Configuration default_config(const std::string sub_name = "")
  {
    typedef Stuff::Functions::Constant< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, 1 > ConstantFunctionType;
    Stuff::Common::Configuration config;
    Stuff::Common::Configuration grid_config;
    grid_config["name"] = "grid";
    grid_config["type"] = "provider.cube";
    grid_config["ll"] = "[0, 0]";
    grid_config["ur"] = "[1, 1]";
    config.add(grid_config, "grid");
    Stuff::Common::Configuration boundary_config;
    boundary_config["name"] = "boundary_info";
    boundary_config["type"] = "dirichlet";
    config.add(boundary_config, "boundary_info");
    Stuff::Common::Configuration flux_config = FluxType::default_config();
    flux_config["name"] = "flux";
    flux_config["type"] = FluxType::static_id();
    config.add(flux_config, "flux");
    Stuff::Common::Configuration source_config = SourceType::default_config();
    source_config["name"] = "source";
    source_config["type"] = SourceType::static_id();
    config.add(source_config, "source");
    Stuff::Common::Configuration constant_config = ConstantFunctionType::default_config();
    constant_config["type"] = ConstantFunctionType::static_id();
    constant_config["name"] = "initial_values";
    constant_config["value"] = "0";
    config.add(constant_config, "initial_values");
    constant_config["name"] = "boundary_values";
    config.add(constant_config, "boundary_values");
    if (sub_name.empty())
      return config;
    else {
      Stuff::Common::Configuration tmp;
      tmp.add(config, sub_name);
      return tmp;
    }
  } // ... default_config(...)

  static std::unique_ptr< ThisType > create(const Stuff::Common::Configuration config = default_config(),
                                            const std::string sub_name = static_id())
  {
    const Stuff::Common::Configuration cfg = config.has_sub(sub_name) ? config.sub(sub_name) : config;
    std::shared_ptr< CheckerboardFunctionType >
        checkerboard_function(CheckerboardFunctionType::create(cfg.sub("diffusion_factor")));
    return Stuff::Common::make_unique< ThisType >(checkerboard_function,
                                                  BaseType::create_matrix_function("diffusion_tensor", cfg),
                                                  BaseType::create_vector_function("force", cfg),
                                                  BaseType::create_vector_function("dirichlet", cfg),
                                                  BaseType::create_vector_function("neumann", cfg));
  } // ... create(...)

  Burgers(const std::shared_ptr< const ConfigType > grid_config,
          const std::shared_ptr< const FluxType > flux = std::make_shared< ExpressionFunctionType >("u", "u^2/2", 2),
          const std::shared_ptr< const SourceType > source = std::make_shared< Constant1dFunctionType >(RangeType(0)),
          const std::shared_ptr< const FunctionType > initial_values = std::make_shared< ConstantFunctionType >(RangeFieldImp(0)),
          const std::shared_ptr< const ConfigType > boundary_info = std::make_shared< ConfigType >("type", "dirichlet"),
          const std::shared_ptr< const FunctionType > boundary_values = std::make_shared< ConstantFunctionType >(RangeFieldImp(0)))
    : grid_config_(grid_config)
    , flux_(flux)
    , source_(source)
    , initial_values_(initial_values)
    , boundary_info_(boundary_info)
    , boundary_values (boundary_values)
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

  virtual const std::shared_ptr< const ConfigType >& grid_cfg() const override
  {
    return grid_config_;
  }

  virtual const std::shared_ptr< const ConfigType >& boundary_info() const override
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
  const std::shared_ptr< const ConfigType >   grid_config_;
  const std::shared_ptr< const ConfigType >   boundary_info_;
  const std::shared_ptr< const FunctionType > boundary_values_;
  const std::shared_ptr< const FunctionType > initial_values_;
};


} // namespace Problems
} // namespace Hyperbolic
} // namespace HDD
} // namespace Dune

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_BURGERS_HH

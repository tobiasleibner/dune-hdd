// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#ifndef DUNE_HDD_HYPERBOLIC_PROBLEMS_ONEBEAM_HH
#define DUNE_HDD_HYPERBOLIC_PROBLEMS_ONEBEAM_HH

#include <cmath>
#include <memory>
#include <vector>
#include <string>

#include <dune/grid/yaspgrid.hh>

#include <dune/common/static_assert.hh>
#include <dune/common/exceptions.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/stuff/common/string.hh>
#include <dune/stuff/functions/checkerboard.hh>
#include <dune/stuff/grid/provider.hh>

#include "twobeams.hh"

namespace Dune {
namespace HDD {
namespace Hyperbolic {
namespace Problems {


template< class EntityImp, class DomainFieldImp, size_t domainDim, class RangeFieldImp, size_t rangeDim >
class OneBeam
  : public TwoBeams< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim >
{
  typedef OneBeam< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim > ThisType;
  typedef TwoBeams< EntityImp, DomainFieldImp, domainDim, RangeFieldImp, rangeDim >  BaseType;

public:
  using BaseType::dimDomain;
  using BaseType::dimRange;
  using typename BaseType::FluxSourceEntityType;
  using typename BaseType::DefaultFluxType;
  using typename BaseType::DefaultFunctionType;
  using typename BaseType::DefaultSourceType;
  using typename BaseType::DefaultBoundaryValueType;

  using typename BaseType::FluxType;
  using typename BaseType::SourceType;
  using typename BaseType::FunctionType;
  using typename BaseType::BoundaryValueType;
  using typename BaseType::ConfigType;

  static std::string static_id()
  {
    return BaseType::static_id() + ".onebeam";
  }

  std::string type() const
  {
    return BaseType::type() + ".onebeam";
  }
private:
  static double factorial(size_t n)
  {
      return (n == 1 || n == 0) ? 1 : factorial(n - 1)*n;
  }


//! compute integral of function over entity with given order
template<class Entity, class Function>
static double integrateEntity (const Entity &entity, const Function &f, int p)
{
  // dimension of the entity
  const int dim = Entity::dimension;

  // type used for coordinates in the grid
  typedef typename Entity::Geometry::ctype ctype;

  // get geometry
  const typename Entity::Geometry geometry = entity.geometry();

  // get geometry type
  const Dune::GeometryType gt = geometry.type();

  // get quadrature rule of order p
  const Dune::QuadratureRule<ctype,dim>&
  rule = Dune::QuadratureRules<ctype,dim>::rule(gt,p);     /*@\label{ieh:qr}@*/

  // ensure that rule has at least the requested order
  if (rule.order()<p)
    DUNE_THROW(Dune::Exception,"order not available");

  // compute approximate integral
  double result=0;
  for (typename Dune::QuadratureRule<ctype,dim>::const_iterator i=rule.begin();
       i!=rule.end(); ++i)
  {
    // we do not integrate from -1 to 1, but from 0 to 1, so evaluate at 2*x - 1 and multiply by 2
    double fval = 2*f.evaluate(2*geometry.global(i->position()) - 1);       /*@\label{ieh:fval}@*/
    double weight = i->weight();                        /*@\label{ieh:weight}@*/
    double detjac = geometry.integrationElement(i->position());       /*@\label{ieh:detjac}@*/
    result += fval * weight * detjac;                   /*@\label{ieh:result}@*/
  }

  // return result
  return result;
}

static std::string create_legendre_polynomial(const size_t n)
{
  std::string str;
  for (size_t kk = 0; kk <= n/2.0; ++kk) {
    if (kk == 0)
      str += DSC::toString(factorial(2*n)/(factorial(n)*factorial(n)*std::pow(2,n))) + "*((m[0])^" + DSC::toString(n) + ")";
    else
    str += "+(" + DSC::toString(std::pow(-1.0, kk)*factorial(2*n - 2*kk)) + ")/(" + DSC::toString(factorial(n-kk)*factorial(n-2*kk)*factorial(kk)*std::pow(2,n)) + ")" + "*((m[0])^" + DSC::toString(n - 2*kk) + ")";
  }
  std::cout << "legendre " << n << ": " << str << std::endl;
  return str;
}

static RangeFieldImp get_left_boundary_value(const size_t n) {
      typedef Dune::YaspGrid< dimDomain >                     GridType;
      typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
      ConfigType grid_config;
      grid_config["type"] = "provider.cube";
      grid_config["lower_left"] = "[0.0]";
      grid_config["upper_right"] = "[1.0]";
      grid_config["num_elements"] = "[1]";
      GridProviderType grid_provider = *(GridProviderType::create(grid_config));
      const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();
      const auto it = grid->template leafbegin< 0 >();
      const auto& entity = *it;
      typedef typename GridType::template Codim< 0 >::Entity VelocityEntityType;
      const std::string integrand_string = "(" + create_legendre_polynomial(n) + ")*3*exp(3*m[0]+3)/(exp(6)-1)";
      typedef typename Dune::Stuff::Functions::Expression
                    < VelocityEntityType, DomainFieldImp, 1, RangeFieldImp, 1, 1 > IntegrandType;
      const IntegrandType integrand("m", integrand_string);
      // highest possible quadrature order is 60
      return integrateEntity(entity, integrand, 60);
}


  // sigma_a = 10 if 0.4 <= x <= 0.7, 0 else
  // T = Q = 0
  // l-th component of Source is -(sigma_a + T/2*l(l+1))*u[l] + \int_{-1}^1 Q*P_l d\mu), so here
  // Source[l] = -10*u[l]       if 0.4 <= x <= 0.7
  //           = 0              else
  template< size_t N >
  struct CreateSource {
    static std::string value_str()
    {
      std::string str = "[";
      for (size_t rr = 0; rr < 10; ++rr) {
        if (rr > 0)
          str += "; ";
        for (size_t cc = 0; cc < N; ++cc) {
          if (cc > 0)
            str += " ";
          if (rr >= 4 && rr <= 7)      // 0.4 <= x <= 0.7
            str += "-10.0*u[" + DSC::toString(cc) + "]";
          else
            str += "0.0";
        }
      }
      str += "]";
      return str;
    }
  };



  // boundary value has to be (l-th component) int_{-1}^1 3*exp(3*m + 3)/(exp(6) - 1) * P_l(m) dm at x = 0 and [0.0002 0 0 ... ] at x = 1
  template< size_t N >
  struct CreateBoundaryValues {
    static std::string value_str()
    {
      std::string str = "[";
      for (size_t cc = 0; cc < N; ++cc) {
          if (cc > 0)
            str += " ";
          if (cc == 0)
            str += DSC::toString(0.0002 - get_left_boundary_value(cc)) + "*x[0]+" + DSC::toString(get_left_boundary_value(cc));
          else
            str += DSC::toString(get_left_boundary_value(cc)) + "*x[0]+" + DSC::toString(get_left_boundary_value(cc));
      }
      str += "]";
      return str;
    }
  };

protected:
  static ConfigType default_grid_config()
  {
    ConfigType grid_config;
    grid_config["type"] = "provider.cube";
    grid_config["lower_left"] = "[0.0]";
    grid_config["upper_right"] = "[1.0]";
    grid_config["num_elements"] = "[1000]";
    return grid_config;
  }

  static ConfigType default_boundary_info_config()
  {
    ConfigType boundary_config;
    boundary_config["type"] = "dirichlet";
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
    ConfigType source_config = DefaultSourceType::default_config();
    source_config["lower_left"] = "[0.0]";
    source_config["upper_right"] = "[1.0]";
    source_config["num_elements"] = "[10]";
    source_config["variable"] = "u";
    source_config["values"] = CreateSource< dimRange >::value_str();;
    source_config["values_are_vectors"] = "true";
    source_config["name"] = static_id();
    config.add(source_config, "source", true);
    ConfigType boundary_value_config = DefaultBoundaryValueType::default_config();
    boundary_value_config["type"] = DefaultBoundaryValueType::static_id();
    boundary_value_config["variable"] = "x";
    boundary_value_config["expression"] = CreateBoundaryValues< dimRange >::value_str();
    boundary_value_config["order"] = "10";
    config.add(boundary_value_config, "boundary_values", true);
    if (sub_name.empty())
      return config;
    else {
      ConfigType tmp;
      tmp.add(config, sub_name);
      return tmp;
    }
  } // ... default_config(...)

  OneBeam(const std::shared_ptr< const FluxType > flux,
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

#endif // DUNE_HDD_HYPERBOLIC_PROBLEMS_ONEBEAM_HH

// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include "config.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <dune/common/parallel/mpihelper.hh>

#include <dune/stuff/common/string.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/grid/periodicview.hh>
#include <dune/stuff/la/container/common.hh>
#include <dune/gdt/timestepper/rungekutta.hh>

#include <dune/stuff/playground/functions/indicator.hh>

#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/spaces/fv/default.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/projections.hh>
#include <dune/gdt/operators/advection.hh>

#include <dune/grid/alugrid/common/declaration.hh>

#include <dune/hdd/hyperbolic/problems/burgers.hh>
#include <dune/hdd/hyperbolic/problems/default.hh>

using namespace Dune::GDT;
using namespace Dune::HDD;

int main()
{
  try {
    static const int dimDomain = 1;
    static const int dimRange = 1;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >                                     GridType;
//    typedef Dune::ALUGrid< dimDomain, dimDomain, Dune::simplex, Dune::conforming >      GridType;
    typedef typename GridType::Codim< 0 >::Entity           EntityType;

    //configure Problem
//    typedef Dune::HDD::Hyperbolic::Problems::Burgers< EntityType, double, dimDomain, double, dimRange > ProblemType;
    typedef Dune::HDD::Hyperbolic::Problems::Default< EntityType, double, dimDomain, double, dimRange > ProblemType;
    typedef typename ProblemType::FunctionType BoundaryValueFunctionType;
    typedef typename ProblemType::ConfigType ConfigType;
    ConfigType problem_config = ProblemType::default_config();
    //set boundary type ("periodic" or "dirichlet")
    ConfigType boundary_config;
//    boundary_config["type"] = "dirichlet";
    boundary_config["type"] = "periodic";
    problem_config.add(boundary_config, "boundary_info", true);
    //set boundary values (ignored if boundary is periodic)
    ConfigType boundary_value_config = ProblemType::DefaultFunctionType::default_config();
    boundary_value_config["type"] = ProblemType::DefaultFunctionType::static_id();
    boundary_value_config["variable"] = "x";
    boundary_value_config["expression"] = "x[0]";
    boundary_value_config["order"] = "1";
    problem_config.add(boundary_value_config, "boundary_values", true);

    //create Problem
    const auto problem_ptr = ProblemType::create();
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    typedef typename ProblemType::ConfigType ConfigType;
    const auto grid_config = problem.grid_config();

    //get analytical flux and initial values
    typedef typename ProblemType::FluxType            AnalyticalFluxType;
    typedef typename ProblemType::SourceType          SourceType;
    typedef typename ProblemType::FunctionType        FunctionType;
    typedef typename FunctionType::DomainFieldType    DomainFieldType;
    typedef typename ProblemType::RangeFieldType      RangeFieldType;
    typedef typename Dune::Stuff::Functions::Indicator < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 > IndicatorFunctionType;
    typedef typename IndicatorFunctionType::DomainType DomainType;
    const std::shared_ptr< const AnalyticalFluxType > analytical_flux = problem.flux();
    const std::shared_ptr< const FunctionType > initial_values = problem.initial_values();
//    const std::shared_ptr< const FunctionType > initial_values = IndicatorFunctionType::create(); //std::make_shared< const IndicatorFunctionType >
//            (std::vector< std::tuple < DomainType, DomainType, RangeFieldType > > (1, std::make_tuple< DomainType, DomainType, RangeFieldType >(DomainType(0.5), DomainType(1), RangeFieldType(1))));
    const std::shared_ptr< const SourceType > source = problem.source();

    //create grid
    std::cout << "Creating Grid..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config, "grid"));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    // make a finite volume space on the leaf grid
    std::cout << "Creating GridView..." << std::endl;
    typedef typename GridType::LeafGridView                                     GridViewType;
    const GridViewType grid_view = grid->leafGridView();
    const GridViewType& grid_view_ref = grid_view;
    typedef typename Dune::Stuff::Grid::PeriodicGridView< GridViewType >        PeriodicGridViewType;
    typedef Spaces::FV::Default< PeriodicGridViewType, RangeFieldType, 1 >      FVSpaceType;
    std::bitset< dimDomain > periodic_directions;
    if (problem.boundary_info()["type"] == "periodic")
      periodic_directions.set();
    std::cout << "Creating PeriodicGridView..." << std::endl;
    const PeriodicGridViewType periodic_grid_view(grid_view_ref, periodic_directions);
    std::cout << "Creating FiniteVolumeSpace..." << std::endl;
    const FVSpaceType fv_space(periodic_grid_view);

    // allocate a discrete function for the concentration and another one to temporary store the update in each step
    std::cout << "Allocating discrete functions..." << std::endl;
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< RangeFieldType > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");

    //visualize initial values
    std::cout << "Projecting initial values..." << std::endl;
    project(*initial_values, u);

    const double dt=0.0002;
    const double saveInterval = 0.004;
    const double t_end = 3;

    //calculate dx and create lambda = dt/dx for the Lax-Friedrichs flux
    std::cout << "Calculating dx..." << std::endl;
    Dune::Stuff::Grid::Dimensions< PeriodicGridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 > ConstantFunctionType;
    ConstantFunctionType ratio_dt_dx(dt/dx);
    std::cout <<" dt/dx: "<< dt/dx << std::endl;

    //create operator
    typedef typename Dune::GDT::Operators::AdvectionLaxFriedrichs< AnalyticalFluxType, ConstantFunctionType, FVSpaceType > OperatorType;
    OperatorType lax_friedrichs_operator(*analytical_flux, ratio_dt_dx, fv_space);

    //create butcher_array
    // forward euler
    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType >  >("[0]"));
    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType >  >("[1]"));
    // generic second order, x = 1 (see https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
//    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType >  >("[0 0; 1 0]"));
//    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType >  >("[0.5 0.5]"));
    // classic fourth order RK
//    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType >  >("[0 0 0 0; 0.5 0 0 0; 0 0.5 0 0; 0 0 1 0]"));
//    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType >  >("[" + DSC::toString(1.0/6.0) + " " + DSC::toString(1.0/3.0) + " " + DSC::toString(1.0/3.0) + " " + DSC::toString(1.0/6.0) + "]"));

    //create timestepper
    std::cout << "Creating TimeStepper..." << std::endl;
    Dune::GDT::TimeStepper::RungeKutta< OperatorType, FVFunctionType, SourceType > timestepper(lax_friedrichs_operator, u, *source, 0.0, A, b);

    // now do the time steps
    timestepper.solve(t_end, dt, saveInterval, true);

    std::cout << "Finished!!\n";

    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

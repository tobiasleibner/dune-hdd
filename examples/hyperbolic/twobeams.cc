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
#include <dune/stuff/la/container/common.hh>

#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/spaces/fv/defaultproduct.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/projections.hh>
#include <dune/gdt/operators/advection.hh>
#include <dune/gdt/timestepper/rungekutta.hh>

#include <dune/hdd/hyperbolic/problems/twobeams.hh>
#include <dune/hdd/hyperbolic/problems/twopulses.hh>
#include <dune/hdd/hyperbolic/problems/rectangularic.hh>
#include <dune/hdd/hyperbolic/problems/sourcebeam.hh>
#include <dune/hdd/hyperbolic/problems/onebeam.hh>

using namespace Dune::GDT;
using namespace Dune::HDD;

int main()
{
  try {
    static const size_t dimDomain = 1;
    // for dimRange > 250, an "exceeded maximum recursive template instantiation limit" error occurs (tested with
    // clang 3.5). You need to pass -ftemplate-depth=N with N >= dimRange + 5 to clang for higher dimRange.
    static const size_t dimRange = 11;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >                                     GridType;
    typedef typename GridType::Codim< 0 >::Entity                           EntityType;

    //configure Problem
    typedef Dune::HDD::Hyperbolic::Problems::TwoBeams< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::TwoPulses< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::RectangularIC< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::SourceBeam< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::OneBeam< EntityType, double, dimDomain, double, dimRange > ProblemType;
    //create Problem
    const auto problem_ptr = ProblemType::create();
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    const auto grid_config = problem.grid_config();

    //get analytical flux, initial and boundary values
    typedef typename ProblemType::FluxType              AnalyticalFluxType;
    typedef typename ProblemType::SourceType            SourceType;
    typedef typename ProblemType::FunctionType          FunctionType;
    typedef typename ProblemType::BoundaryValueType     BoundaryValueType;
    typedef typename FunctionType::DomainFieldType      DomainFieldType;
    typedef typename ProblemType::RangeFieldType        RangeFieldType;
    const std::shared_ptr< const AnalyticalFluxType > analytical_flux = problem.flux();
    const std::shared_ptr< const FunctionType > initial_values = problem.initial_values();
    const std::shared_ptr< const BoundaryValueType > boundary_values = problem.boundary_values();
    const std::shared_ptr< const SourceType > source = problem.source();

    //create grid
    std::cout << "Creating Grid..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    // make a product finite volume space on the leaf grid
    std::cout << "Creating GridView..." << std::endl;
    typedef typename GridType::LeafGridView                                        GridViewType;
    const GridViewType grid_view = grid->leafGridView();
    typedef Spaces::FV::DefaultProduct< GridViewType, RangeFieldType, dimRange >   FVSpaceType;
    std::cout << "Creating FiniteVolumeSpace..." << std::endl;
    const FVSpaceType fv_space(grid_view);

    // allocate a discrete function for the concentration
    std::cout << "Allocating discrete functions..." << std::endl;
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< RangeFieldType > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");

    //project initial values
    std::cout << "Projecting initial values..." << std::endl;
    project(*initial_values, u);

    //choose initial time step length and end time
    double dt = 0.4;
    const double t_end = 7;

    //calculate dx and create lambda = dt/dx for the Lax-Friedrichs flux
    std::cout << "Calculating dx..." << std::endl;
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > ConstantFunctionType;
    ConstantFunctionType ratio_dt_dx(dt/dx);

    //create operator
    typedef typename Dune::GDT::Operators::AdvectionGodunov< AnalyticalFluxType, ConstantFunctionType, BoundaryValueType, FVSpaceType > OperatorType;
    OperatorType advection_operator(*analytical_flux, ratio_dt_dx, *boundary_values, fv_space, true);

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
    Dune::GDT::TimeStepper::RungeKutta< OperatorType, FVFunctionType, SourceType > timestepper(advection_operator, u, *source, A, b);

    //search suitable time step length
    std::pair< bool, double > dtpair = std::make_pair(bool(false), dt);
    while (!(dtpair.first)) {
      dtpair = timestepper.find_suitable_dt(dt, 2, 500, 200);
      dt = dtpair.second;
    }
    std::cout <<" dt/dx: "<< dt/dx << std::endl;
    const double saveInterval = t_end/1000 > dt ? t_end/1000 : dt;
    // now do the time steps
    timestepper.solve(t_end, dt, saveInterval, true);

    std::cout << "Finished!!\n";

    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include "config.h"

#include <sys/resource.h>

#include <cstdio>
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/fvector.hh>

#include <dune/stuff/common/string.hh>
#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container/common.hh>
#include <dune/stuff/playground/functions/composition.hh>

#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/advection.hh>
#include <dune/gdt/operators/projections.hh>
#include <dune/gdt/spaces/fv/defaultproduct.hh>
#include <dune/gdt/timestepper/rungekutta.hh>

#include <dune/hdd/hyperbolic/problems/twobeams.hh>
#include <dune/hdd/hyperbolic/problems/twopulses.hh>
#include <dune/hdd/hyperbolic/problems/rectangularic.hh>
#include <dune/hdd/hyperbolic/problems/sourcebeam.hh>
#include <dune/hdd/hyperbolic/problems/onebeam.hh>
#include <dune/hdd/hyperbolic/problems/transport.hh>
#include <dune/hdd/hyperbolic/problems/2dboltzmann.hh>
#include <dune/hdd/hyperbolic/problems/2dboltzmanncheckerboard.hh>
#include <dune/hdd/hyperbolic/problems/shallowwater.hh>

using namespace Dune::GDT;
using namespace Dune::HDD;

int main(int argc, char* argv[])
{
  try {
    // setup MPI
    typedef Dune::MPIHelper MPIHelper;
    MPIHelper::instance(argc, argv);

    // parse options
    if (argc < 5) {
      std::cerr << "Usage: " << argv[0] << "-threading.max_count THREADS -global.datadir DIR [-gridsize GRIDSIZE]" << std::endl;
      return 1;
    }
    size_t num_threads;
    std::string output_dir;
    std::string grid_size = "1000";
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "-threading.max_count") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          num_threads = DSC::fromString< size_t >(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
          DS::threadManager().set_max_threads(num_threads);
        } else {
          std::cerr << "-threading.max_count option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-global.datadir") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          output_dir = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
          DSC_CONFIG.set("global.datadir", output_dir, true);
        } else {
          std::cerr << "-global.datadir option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-gridsize") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          grid_size = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
        } else {
          std::cerr << "-gridsize option requires one argument." << std::endl;
          return 1;
        }
      }
    }

    // setup threadmanager
    DSC_CONFIG.set("threading.partition_factor", 1, true);
    // set dimensions
    static const size_t dimDomain = 1;
    // for dimRange > 250, an "exceeded maximum recursive template instantiation limit" error occurs (tested with
    // clang 3.5). You need to pass -ftemplate-depth=N with N > dimRange + 10 to clang for higher dimRange.
    static const size_t momentOrder = 5;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >                                     GridType;
    typedef typename GridType::Codim< 0 >::Entity                           EntityType;

    //configure Problem
//    typedef Dune::HDD::Hyperbolic::Problems::Transport< EntityType, double, dimDomain, double, momentOrder + 1 > ProblemType;
    typedef Dune::HDD::Hyperbolic::Problems::TwoBeams< EntityType, double, dimDomain, double, momentOrder + 1 > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::TwoPulses< EntityType, double, dimDomain, double, momentOrder + 1 > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::RectangularIC< EntityType, double, dimDomain, double, momentOrder + 1 > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::SourceBeam< EntityType, double, dimDomain, double, momentOrder + 1 > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::OneBeam< EntityType, double, dimDomain, double, momentOrder + 1 > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::Boltzmann2DLineSource< EntityType, double, dimDomain, double, momentOrder > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::Boltzmann2DCheckerboard< EntityType, double, dimDomain, double, momentOrder > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::ShallowWater< EntityType, double, dimDomain, double, momentOrder > ProblemType;

    static const size_t dimRange = ProblemType::dimRange;

    //create Problem
    const auto problem_ptr = ProblemType::create(/*"legendre_pol.csv"*/);
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    auto grid_config = problem.grid_config();
    grid_config["num_elements"] = "[" + grid_size;
    for (size_t ii = 1; ii < dimDomain; ++ii)
        grid_config["num_elements"] += " " + grid_size;
    grid_config["num_elements"] += "]";


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
    typedef Spaces::FV::DefaultProduct< GridViewType, RangeFieldType, dimRange, 1 >   FVSpaceType;
    std::cout << "Creating FiniteVolumeSpace..." << std::endl;
    const FVSpaceType fv_space(grid_view);

    // allocate a discrete function for the concentration and another one to temporary store the update in each step
    std::cout << "Allocating discrete functions..." << std::endl;
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< RangeFieldType > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");

    //project initial values
    std::cout << "Projecting initial values..." << std::endl;
    project(*initial_values, u);

    //calculate dx and choose t_end and initial dt
    std::cout << "Calculating dx..." << std::endl;
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    const double CFL = 0.5;
    double dt = CFL*dx;
    const double t_end = 0.5;

    //define operator types
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > ConstantFunctionType;
    typedef typename Dune::GDT::Operators::AdvectionGodunov
            < AnalyticalFluxType, ConstantFunctionType, BoundaryValueType, FVSpaceType/*, Dune::GDT::Operators::SlopeLimiters::superbee*/ > OperatorType;
    typedef typename Dune::GDT::Operators::AdvectionSource< SourceType, FVSpaceType > SourceOperatorType;
    typedef typename Dune::GDT::TimeStepper::RungeKutta< OperatorType, SourceOperatorType, FVFunctionType, double > TimeStepperType;

    //create butcher_array
    // forward euler
    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0]"));
    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[1]"));
    Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0]"));
    // generic second order, x = 1 (see https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
//    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0 0; 1 0]"));
//    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0.5 0.5]"));
//    Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0 1]"));
    // classic fourth order RK
//    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0 0 0 0; 0.5 0 0 0; 0 0.5 0 0; 0 0 1 0]"));
//    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[" + DSC::toString(1.0/6.0) + " " + DSC::toString(1.0/3.0) + " " + DSC::toString(1.0/3.0) + " " + DSC::toString(1.0/6.0) + "]"));
//    Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0 0.5 0.5 1]"));


    // save 100 time steps
    const double saveInterval = t_end/100.0 > dt ? t_end/100.0 : dt;

    //create Operators
    ConstantFunctionType dx_function(dx);
    OperatorType advection_operator(*analytical_flux, dx_function, dt, *boundary_values, fv_space, true /*, false, true*/);
    SourceOperatorType source_operator(*source, fv_space);

    //create timestepper
    std::cout << "Creating TimeStepper..." << std::endl;
    TimeStepperType timestepper(advection_operator, source_operator, u, dx, A, b, c);

    // solve
    std::cout << "Solving...";
    DSC_PROFILER.startTiming("fv.solve");
    timestepper.solve(t_end, dt, saveInterval, false, true, ProblemType::static_id());
    DSC_PROFILER.stopTiming("fv.solve");
    std::cout << "done.\n took: " << DSC_PROFILER.getTiming("fv.solve", true)/1000.0 << "seconds" << std::endl;

    DSC_PROFILER.setOutputdir(output_dir);
    DSC_PROFILER.outputTimings("timings_twobeams");
    std::cout << " done" << std::endl;
    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

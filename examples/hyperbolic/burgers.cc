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
#include <dune/stuff/playground/functions/indicator.hh>

#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/spaces/fv/default.hh>
#include <dune/gdt/spaces/fv/defaultproduct.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/projections.hh>
#include <dune/gdt/operators/advection.hh>
#include <dune/gdt/timestepper/rungekutta.hh>

#if HAVE_ALUGRID
# include <dune/grid/alugrid.hh>
#endif

#include <dune/hdd/hyperbolic/problems/burgers.hh>
#include <dune/hdd/hyperbolic/problems/shallowwater.hh>
#include <dune/hdd/hyperbolic/problems/transport.hh>

using namespace Dune::GDT;
using namespace Dune::HDD;

int main(int argc, char** argv)
{
  try {
    // setup MPI
    typedef Dune::MPIHelper MPIHelper;
    MPIHelper::instance(argc, argv);
    //  typename MPIHelper::MPICommunicator world = MPIHelper::getCommunicator();

    // parse options
    if (argc < 3) {
      std::cerr << "Usage: " << argv[0] << "-threading.max_count [-gridsize GRIDSIZE]" << std::endl;
      return 1;
    }
    size_t num_threads;
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


    static const size_t dimDomain = 1;
    static const size_t dimRange = 1;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >                                     GridType;
//    typedef Dune::ALUGrid< dimDomain, dimDomain, Dune::simplex, Dune::conforming >      GridType;
//    typedef Dune::ALUGrid< dimDomain, dimDomain, Dune::cube, Dune::nonconforming >      GridType;
    typedef typename GridType::Codim< 0 >::Entity           EntityType;

    //configure Problem
//    typedef Dune::HDD::Hyperbolic::Problems::Burgers< EntityType, double, dimDomain, double, dimRange > ProblemType;
    typedef Dune::HDD::Hyperbolic::Problems::Transport< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::ShallowWater< EntityType, double, dimDomain, double, dimRange > ProblemType;

    //create Problem
    const auto problem_ptr = ProblemType::create();
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    auto grid_config = problem.grid_config();
    grid_config["num_elements"] = grid_size;

    //get analytical flux and initial values
    typedef typename ProblemType::FluxType            AnalyticalFluxType;
    typedef typename ProblemType::SourceType          SourceType;
    typedef typename ProblemType::FunctionType        FunctionType;
    typedef typename ProblemType::BoundaryValueType   BoundaryValueType;
    typedef typename FunctionType::DomainFieldType    DomainFieldType;
    typedef typename ProblemType::RangeFieldType      RangeFieldType;
//    typedef typename Dune::Stuff::Functions::Indicator < EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > IndicatorFunctionType;
//    typedef typename IndicatorFunctionType::DomainType DomainType;
//    const std::shared_ptr< const FunctionType > initial_values = IndicatorFunctionType::create();   // Indicator with value 1 on [0.25,0.75]
//    std::make_shared< const IndicatorFunctionType > (std::vector< std::tuple < DomainType, DomainType, RangeFieldType > > (1, std::make_tuple< DomainType, DomainType, RangeFieldType >(DomainType(0.5), DomainType(1), RangeFieldType(1))));
    const std::shared_ptr< const FunctionType > initial_values = problem.initial_values();
    const std::shared_ptr< const AnalyticalFluxType > analytical_flux = problem.flux();
    const std::shared_ptr< const BoundaryValueType > boundary_values = problem.boundary_values();
    const std::shared_ptr< const SourceType > source = problem.source();

    //create grid
    std::cout << "Creating Grid..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    // make a finite volume space on the leaf grid
    std::cout << "Creating GridView..." << std::endl;
    typedef typename GridType::LeafGridView                                     GridViewType;
    const GridViewType grid_view = grid->leafGridView();
    const GridViewType& grid_view_ref = grid_view;
    typedef typename Dune::Stuff::Grid::PeriodicGridView< GridViewType >        PeriodicGridViewType;
//    typedef Spaces::FV::DefaultProduct< GridViewType, RangeFieldType, dimRange >      FVSpaceType;
    typedef Spaces::FV::DefaultProduct< PeriodicGridViewType, RangeFieldType, dimRange > FVSpaceType;
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

    //project initial values
    std::cout << "Projecting initial values..." << std::endl;
    project(*initial_values, u);

    //calculate dx and choose t_end and initial dt
    std::cout << "Calculating dx..." << std::endl;
    Dune::Stuff::Grid::Dimensions< PeriodicGridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    double dt = dx*0.5;
    const double t_end = 1;

    //define operator types
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > ConstantFunctionType;
    typedef typename Dune::GDT::Operators::AdvectionGodunov
            < AnalyticalFluxType, ConstantFunctionType, BoundaryValueType, FVSpaceType/*, Dune::GDT::Operators::SlopeLimiters::no_slope*/ > OperatorType;
    typedef typename Dune::GDT::Operators::AdvectionSource< SourceType, FVSpaceType > SourceOperatorType;
    typedef typename Dune::GDT::TimeStepper::RungeKutta< OperatorType, SourceOperatorType, FVFunctionType, double > TimeStepperType;

    //create butcher_array
    // forward euler
        Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0]"));
        Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[1]"));
        Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0]"));
    // Heun's method
//        Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0 0; 1 0]"));
//        Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0.5 0.5]"));
//        Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0 1]"));
    // optimal third order SSP
//    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0 0 0; 1 0 0; 0.25 0.25 0]"));
//    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0.166666666666666666666666 0.1666666666666666666666666 0.6666666666666666666666666]"));
//    Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0 1 0.5]"));
    // classical fourth order RK4
//    Dune::DynamicMatrix< RangeFieldType > A(DSC::fromString< Dune::DynamicMatrix< RangeFieldType > >("[0 0 0 0; 0.5 0 0 0; 0 0.5 0 0; 0 0 1 0]"));
//    Dune::DynamicVector< RangeFieldType > b(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[" + DSC::toString(1.0/6.0, 15) + " " + DSC::toString(1.0/3.0, 15) + " " + DSC::toString(1.0/3.0, 15) + " " + DSC::toString(1.0/6.0, 15) + "]"));
//    Dune::DynamicVector< RangeFieldType > c(DSC::fromString< Dune::DynamicVector< RangeFieldType > >("[0 0.5 0.5 1]"));


    // search suitable time step length
//    std::pair< bool, double > dtpair = std::make_pair(bool(false), dt);
//    while (!(dtpair.first)) {
//      ConstantFunctionType dx_function(dx);
//      OperatorType advection_operator(*analytical_flux, dx_function, dt, *boundary_values, fv_space, false);
//      SourceOperatorType source_operator(*source, fv_space);
//          TimeStepperType timestepper(advection_operator, source_operator, u, dx, A, b);
//      dtpair = timestepper.find_suitable_dt(dt, 2, 500, 200);
//      dt = dtpair.second;
//    }
    std::cout <<" dt/dx: "<< dt/dx << std::endl;

    //create timestepper
    std::cout << "Creating TimeStepper..." << std::endl;
    ConstantFunctionType dx_function(dx);
    OperatorType advection_operator(*analytical_flux, dx_function, dt, *boundary_values, fv_space, true);
    SourceOperatorType source_operator(*source, fv_space);
    TimeStepperType timestepper(advection_operator, source_operator, u, dx, A, b, c);

    const double saveInterval = t_end/100 > dt ? t_end/100 : dt;
    // now do the time steps
    DSC_PROFILER.startTiming("fv.solve");
    timestepper.solve(t_end, dt, saveInterval, true, false);
    DSC_PROFILER.stopTiming("fv.solve");
    std::cout << "Solving took: " << DSC_PROFILER.getTiming("fv.solve", true)/1000.0 << " or " << DSC_PROFILER.getTiming("fv.solve", false)/1000.0 << std::endl;

    timestepper.visualize_solution("transport_heun_godunov");

    std::cout << "Finished!!\n";

    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

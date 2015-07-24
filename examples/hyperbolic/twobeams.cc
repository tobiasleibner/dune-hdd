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
//#include <dune/gdt/playground/functions/entropymomentfunction.hh>

#include <dune/hdd/hyperbolic/problems/twobeams.hh>
#include <dune/hdd/hyperbolic/problems/twopulses.hh>
#include <dune/hdd/hyperbolic/problems/rectangularic.hh>
#include <dune/hdd/hyperbolic/problems/sourcebeam.hh>
#include <dune/hdd/hyperbolic/problems/onebeam.hh>
#include <dune/hdd/hyperbolic/problems/transport.hh>

using namespace Dune::GDT;
using namespace Dune::HDD;

template< class GridViewType, class SolutionType >
void write_solution_to_csv(const GridViewType& grid_view, SolutionType solution, const std::string filename)
{
  std::cout << "Writing solution to .csv file..." << std::endl;
  //remove file if already existing
  std::string timefilename = filename + "_timesteps";
  remove(filename.c_str());
  remove(timefilename.c_str());
  // open empty file
  std::ofstream output_file(filename);
  std::ofstream time_output_file(timefilename);
  // write first line
  const auto it_end_2 = grid_view.template end< 0 >();
//  if (write_time_to_first_col)
//    output_file << ", ";
//  for (auto it = grid_view.template begin< 0 >(); it != it_end_2; ++it) {
//    const auto& entity = *it;
//    output_file << DSC::toString(entity.geometry().center()[0]);
//  }
  //  output_file << std::endl;
  for (auto& pair : solution) {
    time_output_file << DSC::toString(pair.first) << std::endl;
    const auto discrete_func = pair.second;
    const auto const_it_begin = grid_view.template begin< 0 >();
    for (auto it = grid_view.template begin< 0 >(); it != it_end_2; ++it) {
      const auto& entity = *it;
      if (it != const_it_begin)
        output_file << ", ";
      output_file << DSC::toString(discrete_func.local_discrete_function(entity)->evaluate(entity.geometry().local(entity.geometry().center()))[0]);
    }
    output_file << std::endl;
  }
  output_file.close();
}

template< class GridViewType, class SolutionType >
double compute_L1_norm(const GridViewType& grid_view, const SolutionType solution)
{
    double norm = 0;
    for (size_t ii = 0; ii < solution.size(); ++ii) {
      double spatial_integral = 0;
      const auto it_end = grid_view.template end< 0 >();
      for (auto it = grid_view.template begin< 0 >(); it != it_end; ++it) {
        const auto& entity = *it;
        double value = std::abs(solution[ii].second.vector()[grid_view.indexSet().index(entity)]);
        spatial_integral += value*entity.geometry().volume();
//          std::cout << "value: " << value <<  "entity.geometry.volume: " << entity.geometry().volume() << std::endl;
      }
//        const double dt = ii == 0 ? function[ii].first : function[ii].first - function[ii-1].first;
      const double dt = (ii == solution.size() - 1) ? solution[ii].first - solution[ii-1].first : solution[ii+1].first - solution[ii].first;
      norm += dt*spatial_integral;
//        std::cout << "dt = " << dt << ", spatial: " << spatial_integral << std::endl;
    }
//      std::cout << "norm: " << norm << std::endl;
    return norm;
}

void mem_usage() {
  auto comm = Dune::MPIHelper::getCollectiveCommunication();
  // Compute the peak memory consumption of each processes
  int who = RUSAGE_SELF;
  struct rusage usage;
  getrusage(who, &usage);
  long peakMemConsumption = usage.ru_maxrss;
  // compute the maximum and mean peak memory consumption over all processes
  long maxPeakMemConsumption = comm.max(peakMemConsumption);
  long totalPeakMemConsumption = comm.sum(peakMemConsumption);
  long meanPeakMemConsumption = totalPeakMemConsumption / comm.size();
  // write output on rank zero
  if (comm.rank() == 0) {
    std::unique_ptr<boost::filesystem::ofstream> memoryConsFile(
        DSC::make_ofstream(std::string(DSC_CONFIG_GET("global.datadir", "data/")) + std::string("/memory.csv")));
    *memoryConsFile << "global.maxPeakMemoryConsumption,global.meanPeakMemoryConsumption\n" << maxPeakMemConsumption
                    << "," << meanPeakMemConsumption << std::endl;
  }
}


int main(int argc, char* argv[])
{
  try {
    // setup MPI
    typedef Dune::MPIHelper MPIHelper;
    MPIHelper::instance(argc, argv);
    //  typename MPIHelper::MPICommunicator world = MPIHelper::getCommunicator();

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
    static const size_t dimRange = 1;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >                                     GridType;
    typedef typename GridType::Codim< 0 >::Entity                           EntityType;

    //configure Problem
    typedef Dune::HDD::Hyperbolic::Problems::Transport< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::TwoBeams< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::TwoPulses< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::RectangularIC< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::SourceBeam< EntityType, double, dimDomain, double, dimRange > ProblemType;
//    typedef Dune::HDD::Hyperbolic::Problems::OneBeam< EntityType, double, dimDomain, double, dimRange > ProblemType;
    //create Problem
    const auto problem_ptr = ProblemType::create(/*"legendre_pol.csv"*/);
    const auto& problem = *problem_ptr;

    //get grid configuration from problem
    auto grid_config = problem.grid_config();
    grid_config["num_elements"] = grid_size;

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
    double dt = CFL*dx; //dx/4.0;
    const double t_end = 2;

    //define operator types
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, dimRange, 1 > ConstantFunctionType;
    typedef typename Dune::GDT::Operators::AdvectionGodunov
            < AnalyticalFluxType, ConstantFunctionType, BoundaryValueType, FVSpaceType/*, Dune::GDT::Operators::SlopeLimiters::superbee*/ > OperatorType;
    typedef typename Dune::GDT::Operators::AdvectionSource< SourceType, FVSpaceType > SourceOperatorType;

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

    //search suitable time step length
//    std::pair< bool, double > dtpair = std::make_pair(bool(false), dt);
//    while (!(dtpair.first)) {
//      ConstantFunctionType dx_function(dx);
//      OperatorType advection_operator(*analytical_flux, dx_function, dt, *boundary_values, fv_space, true);
//      Dune::GDT::TimeStepper::RungeKutta< OperatorType, FVFunctionType, SourceType > timestepper(advection_operator, u, *source, dx, A, b);
//      dtpair = timestepper.find_suitable_dt(dt, 2, 500, 1000);
//      dt = dtpair.second;
//    }
    std::cout <<" dt/dx: "<< dt/dx << std::endl;

    const double saveInterval = t_end/1000.0 > dt ? t_end/1000.0 : dt;

    //create Operators
    ConstantFunctionType dx_function(dx);
    OperatorType advection_operator(*analytical_flux, dx_function, dt, *boundary_values, fv_space, true);
    SourceOperatorType source_operator(*source, fv_space);

    //create timestepper
    std::cout << "Creating TimeStepper..." << std::endl;
    typedef typename Dune::GDT::TimeStepper::RungeKutta< OperatorType, SourceOperatorType, FVFunctionType, double > TimeStepperType;
    TimeStepperType timestepper(advection_operator, source_operator, u, dx, A, b);

//    typedef typename TimeStepperType::SolutionType SolutionType;
//    std::unique_ptr< SolutionType > solution1, solution2;

    // solve five times to average timings
    for (size_t run = 0; run < 5; ++run) {
    // now do the time steps
    timestepper.reset();

//    boost::timer::cpu_timer timer;
    DSC_PROFILER.startTiming("fv.solve");
    std::vector< std::pair< double, FVFunctionType > > solution_as_discrete_function;
    timestepper.solve(t_end, dt, saveInterval, solution_as_discrete_function);
    DSC_PROFILER.stopTiming("fv.solve");
//    const auto duration = timer.elapsed();
//    std::cout << "took: " << duration.wall*1e-9 << " seconds(" << duration.user*1e-9 << ", " << duration.system*1e-9 << ")" << std::endl;
//    std::cout << "took: " << DSC_PROFILER.getTiming("solve")/1000.0 << "(walltime " << DSC_PROFILER.getTiming("solve", true)/1000.0 << ")" << std::endl;
    DSC_PROFILER.nextRun();

      // write timings to file
//      const bool file_already_exists = boost::filesystem::exists("time.csv");
//      std::ofstream output_file("time.csv", std::ios_base::app);
//      if (!file_already_exists) { // write header
//      output_file << "Problem: " + problem.static_id()
//                  << ", dimRange = " << dimRange
// //                  << ", number of grid cells: " << grid_config["num_elements"]
//                  << ", dt = " << DSC::toString(dt)
//                  << std::endl;
//      output_file << "num_processes, num_grid_cells, wall, user, system" << std::endl;
//      }
//      output_file << argv[1] << ", " << grid_config["num_elements"] << ", " << duration.wall*1e-9 << ", " << duration.user*1e-9 << ", " << duration.system*1e-9 << std::endl;
//      output_file.close();

      // visualize solution
//      timestepper.visualize_solution();
//    }
    for (size_t ii = 0; ii < solution_as_discrete_function.size(); ++ii) {
      auto& pair = solution_as_discrete_function[ii];
      pair.second.template visualize_factor< 0 >("prefix_factor_" + DSC::toString(run)
                                                                       + "_" + DSC::toString(ii), true);
    }
}
      // write solution to *.csv file
      write_solution_to_csv(grid_view, timestepper.solution(), problem.short_id() + "_P" + DSC::toString(dimRange - 1) + "_n" + DSC::toString(1.0/dx) + "_CFL" + DSC::toString(CFL) + "_CGLegendre_fractional_exact.csv");

//    // compute L1 error norm
//    SolutionType difference(*solution1);
//    for (size_t ii = 0; ii < difference.size(); ++ii) {
//      assert(DSC::FloatCmp::eq(difference[ii].first, solution2->operator[](ii).first) && "Time steps must be the same");
//      difference[ii].second.vector() = difference[ii].second.vector() - solution2->operator[](ii).second.vector();
//    }
//    std::cout << "error: " << DSC::toString(compute_L1_norm(grid_view, difference)) << std::endl;

    mem_usage();
    DSC_PROFILER.setOutputdir(output_dir);
    DSC_PROFILER.outputTimings("profiler");
    std::cout << " done" << std::endl;
    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

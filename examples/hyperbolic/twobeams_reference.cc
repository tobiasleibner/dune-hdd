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
#include <dune/hdd/hyperbolic/problems/2dboltzmann.hh>
#include <dune/hdd/hyperbolic/problems/2dboltzmanncheckerboard.hh>
#include <dune/hdd/hyperbolic/problems/shallowwater.hh>

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

double sigma_a(const double /*t*/,const double /*x*/) {
  return 4.0;
}

double T(const double /*t*/, const double /*x*/) {
  return 0;
}

double Q(const double /*t*/, const double /*x*/, const double /*mu*/) {
  return 0;
}

double boundary_conditions_left(const double psi, const double mu, const bool on_top_boundary) {
  if (mu > 0)
    return on_top_boundary ? 100.0 : 0.0;
  else
    return psi;
}

double boundary_conditions_right(const double psi, const double mu, const bool on_bottom_boundary) {
  if (mu < 0) {
    return on_bottom_boundary ? 100.0 : 0.0;
  } else {
    return psi;
  }
}


template< class DiscreteFunctionType >
void step(double& t, const double dt, const double dx, const double dmu, DiscreteFunctionType& u_n)
{
    const auto u_n_tmp = u_n;
    auto& u_n_vector = u_n.vector();
    const auto& u_n_tmp_vector = u_n_tmp.vector();
    const auto& mapper = u_n.space().mapper();
    const auto& grid_view = u_n.space().grid_view();
    const auto it_end = grid_view.template end< 0 >();
    auto it = grid_view.template begin< 0 >();
    auto left = *it;
    auto right = left;
    auto top = left;
    auto bottom = left;
    double mu_left;
    double mu_right;
    for (; it != it_end; ++it) {
      bool on_left_boundary(false);
      bool on_right_boundary(false);
      bool on_top_boundary(false);
      bool on_bottom_boundary(false);
      const auto entity = *it;
      const auto entity_coords = entity.geometry().center();
      const auto x = entity_coords[0];
      const auto mu = entity_coords[1];
      const auto i_it_end = grid_view.iend(entity);
      for (auto i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
        const auto& intersection = *i_it;
        const auto intersection_coords = intersection.geometry().center();
        if (DSC::FloatCmp::eq(intersection_coords[0],entity_coords[0])) {
          if (intersection_coords[1] > entity_coords[1]) {
            if (intersection.neighbor())
              top = intersection.outside();
            else
              on_top_boundary = true;
          } else {
            if (intersection.neighbor())
              bottom = intersection.outside();
            else
              on_bottom_boundary = true;
          }
        } else if (DSC::FloatCmp::eq(intersection_coords[1],entity_coords[1])) {
          if (intersection_coords[0] > entity_coords[0]) {
            if (intersection.neighbor())
              right = intersection.outside();
            else
              on_right_boundary = true;
          } else {
            if (intersection.neighbor())
              left = intersection.outside();
            else
              on_left_boundary = true;
          }
        } else {
          DUNE_THROW(Dune::InvalidStateException, "This should not happen!");
        }
      }
      const auto entity_index = mapper.mapToGlobal(entity, 0);
      const auto left_index = mapper.mapToGlobal(left, 0);
      const auto right_index = mapper.mapToGlobal(right, 0);
      const auto top_index = mapper.mapToGlobal(top, 0);
      const auto bottom_index = mapper.mapToGlobal(bottom, 0);
      const auto psi_i_k = u_n_tmp_vector[entity_index];
      const auto psi_iplus1_k = on_right_boundary ? boundary_conditions_right(psi_i_k, mu, on_bottom_boundary) : u_n_tmp_vector[right_index];
      const auto psi_iminus1_k = on_left_boundary ? boundary_conditions_left(psi_i_k, mu, on_top_boundary) : u_n_tmp_vector[left_index];
      const auto psi_i_kplus1 = on_top_boundary ? psi_i_k : u_n_tmp_vector[top_index];
      const auto psi_i_kminus1 = on_bottom_boundary ? psi_i_k : u_n_tmp_vector[bottom_index];
      u_n_vector[entity_index] += dt*(-1.0*mu*(psi_iplus1_k-psi_iminus1_k)/(2.0*dx) - sigma_a(t,x)*psi_i_k
                                      + Q(t,x,mu) + 0.5*T(t,x)*((1-mu*mu)*(psi_i_kplus1 - 2*psi_i_k + psi_i_kminus1)/(dmu*dmu)
                                                                - mu*(psi_i_kplus1-psi_i_kminus1)/dmu));


    }
    t += dt;
    std::cout << t << " and " << dt << std::endl;
}

template <class DiscreteFunctionType>
void solve(DiscreteFunctionType& u_n,
           const double t_end,
           const double first_dt,
           const double dx,
           const double dmu,
           const double save_step_length,
           const bool save_solution,
           const bool write_solution,
           const std::string filename_prefix,
           std::vector< std::pair< double, DiscreteFunctionType > >& solution)
{
  double t_ = 0;
  double dt = first_dt;
  assert(t_end - t_ >= dt);
  size_t time_step_counter = 0;

  const double save_interval = DSC::FloatCmp::eq(save_step_length, 0.0) ? dt : save_step_length;
  double next_save_time = t_ + save_interval > t_end ? t_end : t_ + save_interval;
  size_t save_step_counter = 1;

  // clear solution
  if (save_solution) {
    solution.clear();
    solution.emplace_back(std::make_pair(t_, u_n));
  }
  if (write_solution)
    u_n.visualize(filename_prefix + "_0");

  while (t_ + dt < t_end)
  {
    // do a timestep
    step(t_, dt, dx, dmu, u_n);

    // check if data should be written in this timestep (and write)
    if (DSC::FloatCmp::ge(t_, next_save_time - 1e-10)) {
      if (save_solution)
        solution.emplace_back(std::make_pair(t_, u_n));
      if (write_solution)
        u_n.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
      next_save_time += save_interval;
      ++save_step_counter;
    }

    // augment time step counter
    ++time_step_counter;
  } // while (t_ < t_end)

  // do last step s.t. it matches t_end exactly
  if (!DSC::FloatCmp::ge(t_, t_end - 1e-10)) {
    step(t_, t_end - t_, dx, dmu, u_n);
    solution.emplace_back(std::make_pair(t_, u_n));
  }
} // ... solve(...)

template <class FDDiscreteFunction, class IntegratedDiscretFunctionType>
void integrate_over_mu(const FDDiscreteFunction& u, IntegratedDiscretFunctionType& u_integrated, const double dmu)
{
  const auto& grid_view = u.space().grid_view();
  const auto& x_grid_view = u_integrated.space().grid_view();
  const auto& mapper = u.space().mapper();
  const auto& x_mapper = u_integrated.space().mapper();
  const auto it_end = grid_view.template end< 0 >();
  for (auto it = grid_view.template begin< 0 >(); it != it_end; ++it) {
    const auto entity = *it;
    const auto entity_coords = entity.geometry().center();
    const auto x = entity_coords[0];
    const auto x_it_end = x_grid_view.template end< 0 >();
    for(auto x_it = x_grid_view.template begin< 0 >(); x_it != x_it_end; ++x_it) {
      const auto x_entity = *x_it;
      if (DSC::FloatCmp::eq(x, x_entity.geometry().center()[0]))
        u_integrated.vector()[x_mapper.mapToGlobal(x_entity,0)] += u.vector()[mapper.mapToGlobal(entity,0)];
    }
  }
  u_integrated.vector() *= dmu;
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
    std::string grid_size_x = "1000";
    std::string grid_size_mu = "800";
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
      } else if (std::string(argv[i]) == "-gridsize_x") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          grid_size_x = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
        } else {
          std::cerr << "-gridsize option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-gridsize_mu") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          grid_size_mu = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
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

    //choose GridType
    typedef Dune::YaspGrid< 2*dimDomain, Dune::EquidistantOffsetCoordinates< double, 2*dimDomain > >  GridType;
    typedef typename GridType::Codim< 0 >::Entity                                         EntityType;

    //get grid configuration from problem
    size_t x_grid_size = DSC::fromString< size_t >(grid_size_x);
    size_t mu_grid_size = DSC::fromString< size_t >(grid_size_mu);
    Dune::Stuff::Common::Configuration grid_config;
    grid_config["type"] = "provider.cube";
    grid_config["lower_left"] = "[-0.5 -1.0]";
    grid_config["upper_right"] = "[0.5 1.0]";
    grid_config["num_elements"] = "[" + grid_size_x;
    for (size_t ii = 1; ii < 2*dimDomain; ++ii)
        grid_config["num_elements"] += " " + grid_size_mu;
    grid_config["num_elements"] += "]";

    //create grid
    std::cout << "Creating Grid..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    // make a product finite volume space on the leaf grid
    std::cout << "Creating GridView..." << std::endl;
    typedef typename GridType::LeafGridView                                        GridViewType;
    const GridViewType grid_view = grid->leafGridView();
    typedef Spaces::FV::Default< GridViewType, double, 1, 1 >              FVSpaceType;
    std::cout << "Creating FiniteVolumeSpace..." << std::endl;
    const FVSpaceType fv_space(grid_view);

    // allocate a discrete function for the concentration and another one to temporary store the update in each step
    std::cout << "Allocating discrete functions..." << std::endl;
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< double > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");

    //project initial values
    DS::Functions::Constant< EntityType, double, 2, double, 1, 1 > initial_values(0.0001);
    std::cout << "Projecting initial values..." << std::endl;
    project(initial_values, u);

    //calculate dx and choose t_end and initial dt
    std::cout << "Calculating dx..." << std::endl;
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(fv_space.grid_view());
    const double dx = 1.0/x_grid_size;
    const double dmu = 2.0/mu_grid_size;
    std::cout << "dx: " << dx << " dmu: " << dmu << std::endl;
    const double CFL = 0.1;
    double dt = CFL*dx*dx;
    const double t_end = 0.01;
    const double saveInterval = t_end/100.0 > dt ? t_end/100.0 : dt;

    std::vector< std::pair< double, FVFunctionType > > solution;

    solve(u,
          t_end,
          dt,
          dx,
          dmu,
          saveInterval,
          true,
          true,
          "finite_difference",
          solution);

    typedef Dune::YaspGrid< dimDomain, Dune::EquidistantOffsetCoordinates< double, dimDomain > >  XGridType;

    Dune::Stuff::Common::Configuration x_grid_config;
    x_grid_config["type"] = "provider.cube";
    x_grid_config["lower_left"] = "[-0.5]";
    x_grid_config["upper_right"] = "[0.5]";
    x_grid_config["num_elements"] = "[" + grid_size_x;
    x_grid_config["num_elements"] += "]";

    //create grid
    std::cout << "Creating XGrid..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< XGridType >  XGridProviderType;
    XGridProviderType x_grid_provider = *(XGridProviderType::create(grid_config));
    const std::shared_ptr< const XGridType > x_grid = x_grid_provider.grid_ptr();

    // make a product finite volume space on the leaf grid
    std::cout << "Creating GridView..." << std::endl;
    typedef typename XGridType::LeafGridView                                        XGridViewType;
    const XGridViewType x_grid_view = x_grid->leafGridView();
    typedef Spaces::FV::Default< XGridViewType, double, 1, 1 >              XFVSpaceType;
    std::cout << "Creating FiniteVolumeSpace..." << std::endl;
    const XFVSpaceType x_fv_space(x_grid_view);

    // allocate a discrete function for the concentration and another one to temporary store the update in each step
    std::cout << "Allocating discrete functions..." << std::endl;
    typedef DiscreteFunction< XFVSpaceType, Dune::Stuff::LA::CommonDenseVector< double > > XFVFunctionType;

    size_t counter = 0;
    for (const auto& pair : solution) {
      const auto& u_n = pair.second;
      XFVFunctionType u_integrated(x_fv_space, "x_solution");
      integrate_over_mu(u_n, u_integrated, dmu);
      u_integrated.visualize("fd_integrated_" + DSC::toString(counter));
      ++counter;
    }

    std::cout << " done" << std::endl;
    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

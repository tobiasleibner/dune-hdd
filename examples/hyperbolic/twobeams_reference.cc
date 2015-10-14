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

#include <dune/common/version.hh>

#if DUNE_VERSION_NEWER(DUNE_COMMON,3,9) //EXADUNE
# include <dune/grid/utility/partitioning/ranged.hh>
# include <dune/stuff/common/parallel/threadmanager.hh>
#endif

#if HAVE_TBB
# include <tbb/blocked_range.h>
# include <tbb/parallel_reduce.h>
# include <tbb/tbb_stddef.h>
#endif

#include <dune/stuff/aliases.hh>
#include <dune/stuff/common/memory.hh>
#include <dune/stuff/common/ranges.hh>

#include <boost/timer/timer.hpp>
#include <boost/filesystem.hpp>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/fvector.hh>

#include <dune/stuff/common/string.hh>
#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container/common.hh>
#include <dune/stuff/la/container/pattern.hh>
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

// TwoBeams

//double sigma_a(const double /*t*/,const double /*x*/) {
//  return 4.0;
//}

//double T(const double /*t*/, const double /*x*/) {
//  return 0;
//}

//double Q(const double /*t*/, const double /*x*/, const double /*mu*/) {
//  return 0;
//}

//double boundary_conditions_left(const double psi, const double mu, const bool on_top_boundary, const double dmu) {
//  if (mu > 0)
//    return on_top_boundary ? 50.0/dmu : 0.0;
//  else
//    return psi;
//}

//double boundary_conditions_right(const double psi, const double mu, const bool on_bottom_boundary, const double dmu) {
//  if (mu < 0) {
//    return on_bottom_boundary ? 50.0/dmu : 0.0;
//  } else {
//    return psi;
//  }
//}

// SourceBeam

double sigma_a(const double /*t*/, const double x) {
  return x > 2.0 ? 0.0 : 1.0;
}

double T(const double /*t*/, const double x) {
  if (x > 2.0)
    return 10.0;
  else if (x > 1.0)
    return 2.0;
  else
    return 0.0;
}

double Q(const double /*t*/, const double x, const double /*mu*/) {
  return x < 1 || x > 1.5 ? 0.0 : 1.0;
}

double boundary_conditions_left(const double psi, const double mu, const bool on_top_boundary, const double dmu) {
  if (mu > 0)
    return on_top_boundary ? 1/dmu : 0.0;
  else
    return psi;
}

double boundary_conditions_right(const double psi, const double mu, const bool /*on_bottom_boundary*/, const double /*dmu*/) {
  if (mu < 0)
    return 0.0001;
  else
    return psi;
}

template< class EntityRange, class DiscreteFunctionType >
void walk_grid_parallel(const EntityRange& entity_range,
                        const DiscreteFunctionType& u_n,
                        DiscreteFunctionType& u_update,
                        const double t,
                        const double dx,
                        const double dmu)
{
  const auto& mapper = u_n.space().mapper();
  const auto& grid_view = u_n.space().grid_view();
  const auto& u_n_vector = u_n.vector();
  typedef typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType IndexType;
  IndexType left_index, right_index, top_index, bottom_index, entity_index;
  for (const auto& entity : entity_range) {
    bool on_left_boundary(false);
    bool on_right_boundary(false);
    bool on_top_boundary(false);
    bool on_bottom_boundary(false);
    const auto entity_coords = entity.geometry().center();
    entity_index = mapper.mapToGlobal(entity, 0);
    const auto x = entity_coords[0];
    const auto mu = entity_coords[1];
    // find neighbors
    const auto i_it_end = grid_view.iend(entity);
    for (auto i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
      const auto& intersection = *i_it;
      const auto intersection_coords = intersection.geometry().center();
      if (DSC::FloatCmp::eq(intersection_coords[0], entity_coords[0])) {
        if (intersection_coords[1] > entity_coords[1]) {
          if (intersection.neighbor())
            top_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_top_boundary = true;
        } else {
          if (intersection.neighbor())
            bottom_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_bottom_boundary = true;
        }
      } else if (DSC::FloatCmp::eq(intersection_coords[1], entity_coords[1])) {
        if (intersection_coords[0] > entity_coords[0]) {
          if (intersection.neighbor())
            right_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_right_boundary = true;
        } else {
          if (intersection.neighbor())
            left_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_left_boundary = true;
        }
      } else {
        DUNE_THROW(Dune::InvalidStateException, "This should not happen!");
      }
    }

    const auto psi_i_k = u_n_vector[entity_index];
    const auto psi_iplus1_k = on_right_boundary ? boundary_conditions_right(psi_i_k, mu, on_bottom_boundary, dmu) : u_n_vector[right_index];
    const auto psi_iminus1_k = on_left_boundary ? boundary_conditions_left(psi_i_k, mu, on_top_boundary, dmu) : u_n_vector[left_index];
    const auto psi_i_kplus1 = on_top_boundary ? psi_i_k : u_n_vector[top_index];
    const auto psi_i_kminus1 = on_bottom_boundary ? psi_i_k : u_n_vector[bottom_index];
    u_update.vector()[entity_index] = -1.0*mu*(psi_iplus1_k-psi_iminus1_k)/(2.0*dx) - sigma_a(t,x)*psi_i_k
                                                       + Q(t,x,mu) + 0.5*T(t,x)*((1-mu*mu)*(psi_i_kplus1 - 2.0*psi_i_k + psi_i_kminus1)/(dmu*dmu)
                                                                                             - mu*(psi_i_kplus1-psi_i_kminus1)/dmu);
  }
}


template< class EntityRange, class DiscreteFunctionType >
void walk_grid_parallel_rosenbrock(const EntityRange& entity_range,
                                   const DiscreteFunctionType& u_n,
                                   DiscreteFunctionType& u_update,
                                   const double t,
                                   const double dx,
                                   const double dmu,
                                   DS::LA::EigenRowMajorSparseMatrix< double >& jacobian,
                                   const size_t step)
{
  const auto& mapper = u_n.space().mapper();
  const auto& grid_view = u_n.space().grid_view();
  const auto& u_n_vector = u_n.vector();
  typedef typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType IndexType;
  IndexType left_index, right_index, top_index, bottom_index, entity_index;
  for (const auto& entity : entity_range) {
    bool on_left_boundary(false);
    bool on_right_boundary(false);
    bool on_top_boundary(false);
    bool on_bottom_boundary(false);
    const auto entity_coords = entity.geometry().center();
    entity_index = mapper.mapToGlobal(entity, 0);
    const auto x = entity_coords[0];
    const auto mu = entity_coords[1];
    // find neighbors
    const auto i_it_end = grid_view.iend(entity);
    for (auto i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
      const auto& intersection = *i_it;
      const auto intersection_coords = intersection.geometry().center();
      if (DSC::FloatCmp::eq(intersection_coords[0], entity_coords[0])) {
        if (intersection_coords[1] > entity_coords[1]) {
          if (intersection.neighbor())
            top_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_top_boundary = true;
        } else {
          if (intersection.neighbor())
            bottom_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_bottom_boundary = true;
        }
      } else if (DSC::FloatCmp::eq(intersection_coords[1], entity_coords[1])) {
        if (intersection_coords[0] > entity_coords[0]) {
          if (intersection.neighbor())
            right_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_right_boundary = true;
        } else {
          if (intersection.neighbor())
            left_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_left_boundary = true;
        }
      } else {
        DUNE_THROW(Dune::InvalidStateException, "This should not happen!");
      }
    }

    if (step == 0) {
      jacobian.set_entry(entity_index, entity_index, -1.0 * sigma_a(t,x) - T(t,x) * (1.0 - mu * mu)/(dmu * dmu));
      if (!on_right_boundary)
        jacobian.set_entry(entity_index, right_index, mu/(-2.0 *dx));
      if (!on_left_boundary)
        jacobian.set_entry(entity_index, left_index, mu/(2.0 *dx));
      if (!on_bottom_boundary)
        jacobian.set_entry(entity_index, bottom_index, 0.5 * T(t,x) * (1.0 - mu * mu)/(dmu * dmu) + mu/dmu);
      if (!on_top_boundary)
        jacobian.set_entry(entity_index, top_index, 0.5 * T(t,x) * (1.0 - mu * mu)/(dmu * dmu) - mu/dmu);
    } else {
      const auto psi_i_k = u_n_vector[entity_index];
      const auto psi_iplus1_k = on_right_boundary ? boundary_conditions_right(psi_i_k, mu, on_bottom_boundary, dmu) : u_n_vector[right_index];
      const auto psi_iminus1_k = on_left_boundary ? boundary_conditions_left(psi_i_k, mu, on_top_boundary, dmu) : u_n_vector[left_index];
      const auto psi_i_kplus1 = on_top_boundary ? psi_i_k : u_n_vector[top_index];
      const auto psi_i_kminus1 = on_bottom_boundary ? psi_i_k : u_n_vector[bottom_index];
      u_update.vector()[entity_index] = -1.0*mu*(psi_iplus1_k-psi_iminus1_k)/(2.0*dx) - sigma_a(t,x)*psi_i_k
                                                         + Q(t,x,mu) + 0.5*T(t,x)*((1.0-mu*mu)*(psi_i_kplus1 - 2.0*psi_i_k + psi_i_kminus1)/(dmu*dmu)
                                                                                               - mu*(psi_i_kplus1-psi_i_kminus1)/dmu);
    }
  }
} // walk_grid_parallel_rosenbrock


#if HAVE_TBB
  template< class PartitioningType, class DiscreteFunctionType >
  struct Body
  {
    Body(PartitioningType& partitioning,
         const DiscreteFunctionType& u_n,
         DiscreteFunctionType& u_update,
         const double t,
         const double dx,
         const double dmu)
      : partitioning_(partitioning)
      , u_n_(u_n)
      , u_update_(u_update)
      , t_(t)
      , dx_(dx)
      , dmu_(dmu)
    {}

    Body(Body& other, tbb::split /*split*/)
      : partitioning_(other.partitioning_)
      , u_n_(other.u_n_)
      , u_update_(other.u_update_)
      , t_(other.t_)
      , dx_(other.dx_)
      , dmu_(other.dmu_)
    {}

    void operator()(const tbb::blocked_range< std::size_t > &range) const
    {
      // for all partitions in tbb-range
      for(std::size_t p = range.begin(); p != range.end(); ++p) {
        auto partition = partitioning_.partition(p);
        walk_grid_parallel(partition, u_n_, u_update_, t_, dx_, dmu_);
      }
    }

    void join(Body& /*other*/)
    {}

  PartitioningType& partitioning_;
  const DiscreteFunctionType& u_n_;
  DiscreteFunctionType& u_update_;
  const double t_;
  const double dx_;
  const double dmu_;
  }; // struct Body
#endif //HAVE_TBB

#if HAVE_TBB
  template< class PartitioningType, class DiscreteFunctionType >
  struct Body_rosenbrock
  {
    Body_rosenbrock(PartitioningType& partitioning,
                    const DiscreteFunctionType& u_n,
                    DiscreteFunctionType& u_update,
                    const double t,
                    const double dx,
                    const double dmu,
                    DS::LA::EigenRowMajorSparseMatrix< double >& jacobian,
                    const size_t stage)
      : partitioning_(partitioning)
      , u_n_(u_n)
      , u_update_(u_update)
      , t_(t)
      , dx_(dx)
      , dmu_(dmu)
      , jacobian_(jacobian)
      , stage_(stage)
    {}

    Body_rosenbrock(Body_rosenbrock& other, tbb::split /*split*/)
      : partitioning_(other.partitioning_)
      , u_n_(other.u_n_)
      , u_update_(other.u_update_)
      , t_(other.t_)
      , dx_(other.dx_)
      , dmu_(other.dmu_)
      , jacobian_(other.jacobian_)
      , stage_(other.stage_)
    {}

    void operator()(const tbb::blocked_range< std::size_t > &range) const
    {
      // for all partitions in tbb-range
      for(std::size_t p = range.begin(); p != range.end(); ++p) {
        auto partition = partitioning_.partition(p);
        walk_grid_parallel_rosenbrock(partition, u_n_, u_update_, t_, dx_, dmu_, jacobian_, stage_);
      }
    }

    void join(Body_rosenbrock& /*other*/)
    {}

  PartitioningType& partitioning_;
  const DiscreteFunctionType& u_n_;
  DiscreteFunctionType& u_update_;
  const double t_;
  const double dx_;
  const double dmu_;
  DS::LA::EigenRowMajorSparseMatrix< double >& jacobian_;
  const size_t stage_;
  }; // struct Body_rosenbrock
#endif //HAVE_TBB


template <class DiscreteFunctionType>
void apply_finite_difference(const DiscreteFunctionType& u_n,
                             DiscreteFunctionType& u_update,
                             const double t,
                             const double dx,
                             const double dmu)
{
  typedef typename DiscreteFunctionType::SpaceType::GridViewType GridViewType;
#if DUNE_VERSION_NEWER(DUNE_COMMON,3,9) && HAVE_TBB //EXADUNE
    const auto num_partitions = DSC_CONFIG_GET("threading.partition_factor", 1u)
                                * DS::threadManager().current_threads();
    const auto partitioning = DSC::make_unique< Dune::RangedPartitioning< GridViewType, 0 > >(u_n.space().grid_view(), num_partitions);
    tbb::blocked_range< std::size_t > blocked_range(0, partitioning->partitions());
    Body< Dune::RangedPartitioning< GridViewType, 0 >, DiscreteFunctionType > body(*partitioning,
                                                                                   u_n,
                                                                                   u_update,
                                                                                   t,
                                                                                   dx,
                                                                                   dmu);
    tbb::parallel_reduce(blocked_range, body);
#endif
}

template <class DiscreteFunctionType>
void apply_finite_difference_rosenbrock(const DiscreteFunctionType& u_n,
                                        DiscreteFunctionType& u_update,
                                        const double t,
                                        const double dx,
                                        const double dmu,
                                        DS::LA::EigenRowMajorSparseMatrix< double >& jacobian,
                                        const size_t stage)
{
  typedef typename DiscreteFunctionType::SpaceType::GridViewType GridViewType;
#if DUNE_VERSION_NEWER(DUNE_COMMON,3,9) && HAVE_TBB //EXADUNE
    const auto num_partitions = DSC_CONFIG_GET("threading.partition_factor", 1u)
                                * DS::threadManager().current_threads();
    const auto partitioning = DSC::make_unique< Dune::RangedPartitioning< GridViewType, 0 > >(u_n.space().grid_view(), num_partitions);
    tbb::blocked_range< std::size_t > blocked_range(0, partitioning->partitions());
    Body_rosenbrock< Dune::RangedPartitioning< GridViewType, 0 >, DiscreteFunctionType > body(*partitioning,
                                                                                              u_n,
                                                                                              u_update,
                                                                                              t,
                                                                                              dx,
                                                                                              dmu,
                                                                                              jacobian,
                                                                                              stage);
    tbb::parallel_reduce(blocked_range, body);
#endif
}


template< class DiscreteFunctionType >
double step(double& t,
          const double initial_dt,
          const double dx, const double dmu,
          DiscreteFunctionType& u_n,
          Dune::DynamicMatrix< double >& A,
          Dune::DynamicVector< double >& b_1,
          Dune::DynamicVector< double >& b_2,
          Dune::DynamicVector< double >& c)
{
    static DiscreteFunctionType last_stage_of_last_step = u_n;
    static bool first_step = true;
    const auto num_stages = A.rows();
    std::vector< DiscreteFunctionType > u_intermediate_stages(num_stages, last_stage_of_last_step);
    const auto b_diff = b_2 - b_1;
    auto u_n_tmp = u_n;
    double abs_error = 10.0;
    double rel_error = 10.0;
    double TOL = 0.0001;
    double dt = initial_dt;
    double scale_max = 10;
    double scale_factor = 1.0;

    if (first_step) {
      apply_finite_difference(u_n_tmp, last_stage_of_last_step, t, dx, dmu);
      first_step = false;
      std::cout << "first step" << std::endl;
    }

    double u_n_norm = 0;
    for (auto& value : u_n.vector()) {
      u_n_norm += std::abs(value);
    }

    // fuer zeitunabhaengige funktionen hier hin
    u_intermediate_stages[0].vector() = last_stage_of_last_step.vector();
    for (size_t ii = 1; ii < num_stages; ++ii) {
      u_intermediate_stages[ii].vector() *= 0.0;
      u_n_tmp.vector() = u_n.vector();
      for (size_t jj = 0; jj < ii; ++jj)
        u_n_tmp.vector() += u_intermediate_stages[jj].vector()*(dt*(A[ii][jj]));
      apply_finite_difference(u_n_tmp, u_intermediate_stages[ii], t+c[ii]*dt, dx, dmu);
    }

    while (rel_error > TOL) {
      dt *= scale_factor;
      // fur zeitabhaengige funktionen muss das hier hin
//      u_intermediate_stages[0].vector() = last_stage_of_last_step.vector();
//      for (size_t ii = 1; ii < num_stages; ++ii) {
//        u_intermediate_stages[ii].vector() *= 0.0;
//        u_n_tmp.vector() = u_n.vector();
//        for (size_t jj = 0; jj < ii; ++jj)
//          u_n_tmp.vector() += u_intermediate_stages[jj].vector()*(dt*(A[ii][jj]));
//        apply_finite_difference(u_n_tmp, u_intermediate_stages[ii], t+c[ii]*dt, dx, dmu);
//      }

      auto error = u_intermediate_stages[0].vector()*b_diff[0];
      for (size_t ii = 1; ii < num_stages; ++ii) {
        error += u_intermediate_stages[ii].vector()*b_diff[ii];
      }
      abs_error = 0.0;
      for (auto& value : error) {
        abs_error += std::abs(value);
      }
      abs_error *= dt;
      rel_error = abs_error/u_n_norm;
      std::cout << rel_error << std::endl;
      scale_factor = std::min(std::pow(0.9*TOL/rel_error, 1.0/3.0), scale_max);
    }

    for (size_t ii = 0; ii < num_stages; ++ii) {
      u_n.vector() += u_intermediate_stages[ii].vector()*(dt*b_1[ii]);
    }

    last_stage_of_last_step.vector() = u_intermediate_stages[num_stages - 1].vector();

    t += dt;

    std::cout << t << " and dt " << dt << std::endl;

    return dt*scale_factor;
}

template< class DiscreteFunctionType >
double step_rosenbrock(double& t,
                       const double initial_dt,
                       const double dx, const double dmu,
                       DiscreteFunctionType& u_n,
                       Dune::DynamicMatrix< double >& A_new,
                       Dune::DynamicVector< double >& m_1,
                       Dune::DynamicVector< double >& m_2,
                       Dune::DynamicVector< double >& c,
                       Dune::DynamicVector< double >& d,
                       Dune::DynamicMatrix< double >& C,
                       const double gamma,
                       DS::LA::EigenRowMajorSparseMatrix< double >& jacobian,
                       DS::LA::EigenRowMajorSparseMatrix< double >& system_matrix)
{
  typedef typename DS::LA::Solver< typename DS::LA::EigenRowMajorSparseMatrix< double > > SolverType;
  std::unique_ptr< SolverType > solver = DSC::make_unique< SolverType >(system_matrix);

  static DiscreteFunctionType last_stage_of_last_step = u_n;
  static bool first_step = true;
  const auto num_stages = A_new.rows();
  std::vector< DiscreteFunctionType > u_intermediate_stages(num_stages, last_stage_of_last_step);
  const auto m_diff = m_2 - m_1;
  auto u_n_tmp = u_n;
  auto k_i_tmp = last_stage_of_last_step;
  auto k_sum = u_n;
  double abs_error = 10.0;
  double rel_error = 10.0;
  double TOL = 0.0001;
  double dt = initial_dt;
  double scale_max = 10;
  double scale_factor = 1.0;

  if (first_step) {
    std::fill(k_i_tmp.vector().begin(), k_i_tmp.vector().end(), 0.0);
    apply_finite_difference_rosenbrock(u_n_tmp, k_i_tmp, t, dx, dmu, jacobian, 0);
    apply_finite_difference_rosenbrock(u_n_tmp, k_i_tmp, t, dx, dmu, jacobian, 1);
    system_matrix = jacobian;
    system_matrix.scal(-1.0);
    for (size_t row = 0; row < system_matrix.rows(); ++row)
      system_matrix.add_to_entry(row, row, 1.0/(gamma*dt));
    solver = DSC::make_unique< SolverType >(system_matrix);
    solver->apply(k_i_tmp.vector(), last_stage_of_last_step.vector());
    first_step = false;
    std::cout << "first step" << std::endl;
  }

  double u_n_norm = 0;
  for (auto& value : u_n.vector()) {
    u_n_norm += std::abs(value);
  }

  while (rel_error > TOL) {
    dt *= scale_factor;

    for (size_t ii = 0; ii < num_stages; ++ii) {
      std::fill(k_i_tmp.vector().begin(), k_i_tmp.vector().end(), 0.0);
      std::fill(k_sum.vector().begin(), k_sum.vector().end(), 0.0);
      u_n_tmp.vector() = u_n.vector();
      for (size_t jj = 0; jj < ii; ++jj) {
        u_n_tmp.vector().axpy(A_new[ii][jj], u_intermediate_stages[jj].vector());
        k_sum.vector().axpy(C[ii][jj]/dt, u_intermediate_stages[jj].vector());
      }
      apply_finite_difference_rosenbrock(u_n_tmp, k_i_tmp, t+c[ii]*dt, dx, dmu, jacobian, ii);
      // as C_ii is the same for all i, we only need to calculate the matrix in the first step
      if (ii == 0) {
        u_intermediate_stages[ii].vector() = last_stage_of_last_step.vector();
        // create solver
        system_matrix = jacobian;
        system_matrix.scal(-1.0);
        for (size_t row = 0; row < system_matrix.rows(); ++row)
          system_matrix.add_to_entry(row, row, 1.0/(gamma*dt));
        solver = DSC::make_unique< SolverType >(system_matrix);
      } else {
      // fuer explizit zeitabhaengige Funktionen fehlt hier ein Term (siehe Wikipedia)
      // ...
      //
      k_i_tmp.vector() += k_sum.vector();
      // create solver
      solver->apply(k_i_tmp.vector(), u_intermediate_stages[ii].vector());
      }
    }

    auto error = u_intermediate_stages[0].vector()*m_diff[0];
    for (size_t ii = 1; ii < num_stages; ++ii) {
      error += u_intermediate_stages[ii].vector()*m_diff[ii];
    }
    abs_error = 0.0;
    for (auto& value : error) {
      abs_error += std::abs(value);
    }
    abs_error *= dt;
    rel_error = abs_error/u_n_norm;
    std::cout << rel_error << std::endl;
    scale_factor = std::min(std::pow(0.9*TOL/rel_error, 1.0/3.0), scale_max);
  }

  for (size_t ii = 0; ii < num_stages; ++ii) {
    u_n.vector().axpy(dt*m_1[ii], u_intermediate_stages[ii].vector());
  }

  t += dt;

  last_stage_of_last_step.vector() = std::move(u_intermediate_stages[num_stages - 1].vector());

  std::cout << t << " and dt " << dt << std::endl;

  return dt*scale_factor;
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
           std::vector< std::pair< double, DiscreteFunctionType > >& solution,
           Dune::DynamicMatrix< double > A,
           Dune::DynamicVector< double > b_1,
           Dune::DynamicVector< double > b_2,
           Dune::DynamicVector< double > c)
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
    dt = step(t_, dt, dx, dmu, u_n, A, b_1, b_2, c);

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
    step(t_, t_end - t_, dx, dmu, u_n, A, b_1, b_2, c);
    solution.emplace_back(std::make_pair(t_, u_n));
  }
} // ... solve(...)

template< class DiscreteFunctionType >
Dune::Stuff::LA::SparsityPatternDefault assemble_pattern(DiscreteFunctionType& u_n)
{
  const auto& mapper = u_n.space().mapper();
  const auto& grid_view = u_n.space().grid_view();
  typedef typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType IndexType;
  IndexType left_index, right_index, top_index, bottom_index, entity_index;
  const auto num_grid_elements = grid_view.size(0);
  Dune::Stuff::LA::SparsityPatternDefault pattern(num_grid_elements);

  for (const auto& entity : DSC::entityRange(grid_view)) {
    bool on_left_boundary(false);
    bool on_right_boundary(false);
    bool on_top_boundary(false);
    bool on_bottom_boundary(false);
    const auto entity_coords = entity.geometry().center();
    entity_index = mapper.mapToGlobal(entity, 0);
    // find neighbors
    const auto i_it_end = grid_view.iend(entity);
    for (auto i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
      const auto& intersection = *i_it;
      const auto intersection_coords = intersection.geometry().center();
      if (DSC::FloatCmp::eq(intersection_coords[0], entity_coords[0])) {
        if (intersection_coords[1] > entity_coords[1]) {
          if (intersection.neighbor())
            top_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_top_boundary = true;
        } else {
          if (intersection.neighbor())
            bottom_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_bottom_boundary = true;
        }
      } else if (DSC::FloatCmp::eq(intersection_coords[1], entity_coords[1])) {
        if (intersection_coords[0] > entity_coords[0]) {
          if (intersection.neighbor())
            right_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_right_boundary = true;
        } else {
          if (intersection.neighbor())
            left_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            on_left_boundary = true;
        }
      } else {
        DUNE_THROW(Dune::InvalidStateException, "This should not happen!");
      }
    }

    pattern.insert(entity_index, entity_index);
    if (!on_right_boundary)
      pattern.insert(entity_index, right_index);
    if (!on_left_boundary)
      pattern.insert(entity_index, left_index);
    if (!on_bottom_boundary)
      pattern.insert(entity_index, bottom_index);
    if (!on_top_boundary)
      pattern.insert(entity_index, top_index);
  } // grid_walk
  pattern.sort();
  return pattern;
}



template <class DiscreteFunctionType>
void solve_rosenbrock(DiscreteFunctionType& u_n,
                      const double t_end,
                      const double first_dt,
                      const double dx,
                      const double dmu,
                      const double save_step_length,
                      const bool save_solution,
                      const bool write_solution,
                      const std::string filename_prefix,
                      std::vector< std::pair< double, DiscreteFunctionType > >& solution,
                      Dune::DynamicMatrix< double > A,
                      Dune::DynamicVector< double > b_1,
                      Dune::DynamicVector< double > b_2,
                      Dune::DynamicVector< double > c,
                      Dune::DynamicVector< double > d,
                      Dune::DynamicMatrix< double > Gamma)
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

  const auto pattern = assemble_pattern(u_n);

  const auto num_grid_elements = u_n.space().grid_view().size(0);

  DS::LA::EigenRowMajorSparseMatrix< double > jacobian(num_grid_elements, num_grid_elements, pattern);
  DS::LA::EigenRowMajorSparseMatrix< double > system_matrix(num_grid_elements, num_grid_elements, pattern);

  // transform variables for faster calculations, C = diag(1/gamma, ... , 1/gamma) - Gamma^(-1),
  // A_new = A*Gamma^(-1), m = (b_1,...,b_n)*Gamma^(-1)
  auto Gamma_inv = Gamma;
  Gamma_inv.invert();
  auto C = Gamma_inv;
  C *= -1.0;
  for (size_t ii = 0; ii < C.rows(); ++ii)
    C[ii][ii] += 1.0/Gamma[ii][ii];
  A.rightmultiply(Gamma_inv);
  auto m_1 = b_1;
  Gamma_inv.mtv(b_1, m_1);
  auto m_2 = b_2;
  Gamma_inv.mtv(b_2, m_2);
  const auto gamma = Gamma[0][0];

  while (t_ + dt < t_end)
  {
    // do a timestep
    dt = step_rosenbrock(t_, dt, dx, dmu, u_n, A, m_1, m_2, c, d, C, gamma, jacobian, system_matrix);

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
    step_rosenbrock(t_, t_end - t_, dx, dmu, u_n, A, m_1, m_2, c, d, C, gamma, jacobian, system_matrix);
    solution.emplace_back(std::make_pair(t_, u_n));
  }
} // ... solve_rosenbrock(...)

template <class FDDiscreteFunction, class IntegratedDiscretFunctionType>
void integrate_over_mu(const FDDiscreteFunction& u, IntegratedDiscretFunctionType& u_integrated, const double dmu)
{
  const auto& grid_view = u.space().grid_view();
  const auto& x_grid_view = u_integrated.space().grid_view();
  const auto& mapper = u.space().mapper();
  const auto& x_mapper = u_integrated.space().mapper();
  const auto it_end = grid_view.template end< 0 >();
  for (auto it = grid_view.template begin< 0 >(); it != it_end; ++it) {
    const auto& entity = *it;
    const auto entity_index = mapper.mapToGlobal(entity,0);
    const auto entity_coords = entity.geometry().center();
    const auto x = entity_coords[0];
    const auto x_it_end = x_grid_view.template end< 0 >();
    for(auto x_it = x_grid_view.template begin< 0 >(); x_it != x_it_end; ++x_it) {
      const auto& x_entity = *x_it;
      if (DSC::FloatCmp::eq(x, x_entity.geometry().center()[0])) {
        u_integrated.vector()[x_mapper.mapToGlobal(x_entity,0)] += u.vector()[entity_index];
        break;
      }
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
//    grid_config["lower_left"] = "[-0.5 -1.0]";
//    grid_config["upper_right"] = "[0.5 1.0]";
    grid_config["lower_left"] = "[0 -1.0]";
    grid_config["upper_right"] = "[3 1.0]";
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
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::EigenDenseVector< double > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");

    //project initial values
    DS::Functions::Constant< EntityType, double, 2, double, 1, 1 > initial_values(0.0001);
    std::cout << "Projecting initial values..." << std::endl;
    project(initial_values, u);

    //calculate dx and choose t_end and initial dt
    std::cout << "Calculating dx..." << std::endl;
//    const double dx = 1.0/x_grid_size;
    const double dx = 3.0/x_grid_size;
    const double dmu = 2.0/mu_grid_size;
    std::cout << "dx: " << dx << " dmu: " << dmu << std::endl;
    const double CFL = 0.1;
    double dt = CFL*dx;
    const double t_end = 0.1;
    const double saveInterval = t_end/100.0 > dt ? t_end/100.0 : dt;

    std::vector< std::pair< double, FVFunctionType > > solution;


//    // Bogacki-Shampine
//    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
//                                    ("[0 0 0 0; 0.5 0 0 0; 0 0.75 0 0; "
//                                     + DSC::toString(2.0/9.0) + " " + DSC::toString(1.0/3.0) + " " + DSC::toString(4.0/9.0) + " 0]"));
//    Dune::DynamicVector< double > b_1(DSC::fromString< Dune::DynamicVector< double > >("["
//                                                                                       + DSC::toString(2.0/9.0, 15) + " "
//                                                                                       + DSC::toString(1.0/3.0, 15) + " "
//                                                                                       + DSC::toString(4.0/9.0, 15)
//                                                                                       + " 0]"));
//    Dune::DynamicVector< double > b_2(DSC::fromString< Dune::DynamicVector< double > >("["
//                                                                                       + DSC::toString(7.0/24.0, 15) + " "
//                                                                                       + DSC::toString(1.0/4.0, 15) + " "
//                                                                                       + DSC::toString(1.0/3.0, 15) + " "
//                                                                                       + DSC::toString(1.0/8.0, 15) + " 0]"));
//    Dune::DynamicVector< double > c(DSC::fromString< Dune::DynamicVector< double > >("[0.5 0.75 1 0]"));

//    // Dormand–Prince
//    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
//                                            (std::string("[0 0 0 0 0 0 0;") +
//                                             " 0.2 0 0 0 0 0 0;" +
//                                             " 0.075 0.225 0 0 0 0 0;" +
//                                             " " + DSC::toString(44.0/45.0, 15) + " " + DSC::toString(-56.0/15.0, 15) + " " + DSC::toString(32.0/9.0, 15) + " 0 0 0 0;" +
//                                             " " + DSC::toString(19372.0/6561.0, 15) + " " + DSC::toString(-25360.0/2187.0, 15) + " " + DSC::toString(64448.0/6561.0, 15) + " " + DSC::toString(-212.0/729.0, 15) + " 0 0 0;" +
//                                             " " + DSC::toString(9017.0/3168.0, 15) + " " + DSC::toString(-355.0/33.0, 15) + " " + DSC::toString(46732.0/5247.0, 15) + " " + DSC::toString(49.0/176.0, 15) + " " + DSC::toString(-5103.0/18656.0, 15) + " 0 0;" +
//                                             " " + DSC::toString(35.0/384.0, 15) + " 0 " + DSC::toString(500.0/1113.0, 15) + " " + DSC::toString(125.0/192.0, 15) + " " + DSC::toString(-2187.0/6784.0, 15) + " " + DSC::toString(11.0/84.0, 15) + " 0]"));

//    Dune::DynamicVector< double > b_1(DSC::fromString< Dune::DynamicVector< double > >("["
//                                                                                       + DSC::toString(35.0/384.0, 15)
//                                                                                       + " 0 "
//                                                                                       + DSC::toString(500.0/1113.0, 15) + " "
//                                                                                       + DSC::toString(125.0/192.0, 15) + " "
//                                                                                       + DSC::toString(-2187.0/6784.0, 15) + " "
//                                                                                       + DSC::toString(11.0/84.0, 15)
//                                                                                       + " 0]"));
//    Dune::DynamicVector< double > b_2(DSC::fromString< Dune::DynamicVector< double > >("["
//                                                                                       + DSC::toString(5179.0/57600.0, 15)
//                                                                                       + " 0 "
//                                                                                       + DSC::toString(7571.0/16695.0, 15) + " "
//                                                                                       + DSC::toString(393.0/640.0, 15) + " "
//                                                                                       + DSC::toString(-92097.0/339200.0, 15) + " "
//                                                                                       + DSC::toString(187.0/2100.0, 15) + " "
//                                                                                       + DSC::toString(1.0/40.0, 15)
//                                                                                       + "]"));
//    Dune::DynamicVector< double > c(DSC::fromString< Dune::DynamicVector< double > >("[0 0.2 0.3 0.8 "+ DSC::toString(8.0/9.0, 15) +" 1 1]"));

    // Rosenbrock-Wanner
//    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
//                                    ("[0 0; " + DSC::toString(2.0/3.0, 15) + " 0]"));
//    Dune::DynamicVector< double > b(DSC::fromString< Dune::DynamicVector< double > >("[0.25 0.75]"));
//    Dune::DynamicVector< double > c(DSC::fromString< Dune::DynamicVector< double > >("[0 " + DSC::toString(2.0/3.0, 15) + "]"));
//    const double gamma = (1.0 + 1.0/std::sqrt(3))/std::sqrt(2);
//    Dune::DynamicMatrix< double > Gamma(DSC::fromString< Dune::DynamicMatrix< double > >
//                                    ("[" + DSC::toString(gamma, 15) + " 0; " + DSC::toString(-4.0/3.0*gamma, 15) + " " + DSC::toString(gamma, 15) + "]"));

    // GRK4T
    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
                                    (std::string("[0 0 0 0;") +
                                     " 0.462 0 0 0;" +
                                     " -0.0815668168327 0.961775150166 0 0;" +
                                     " 0.217487371653 0.486229037990 0 0.296283590357]"));
    Dune::DynamicVector< double > b_1(DSC::fromString< Dune::DynamicVector< double > >("[0.217487371653 0.486229037990 0 0.296283590357]"));
    Dune::DynamicVector< double > b_2(DSC::fromString< Dune::DynamicVector< double > >("[-0.717088504499 1.77617912176 -0.0590906172617 0]"));
    Dune::DynamicVector< double > c(DSC::fromString< Dune::DynamicVector< double > >("[0 " + DSC::toString(2.0/3.0, 15) + "]"));
    Dune::DynamicMatrix< double > Gamma(DSC::fromString< Dune::DynamicMatrix< double > >
                                        (std::string("[0.231 0 0 0;") +
                                         " -0.270629667752 0.231 0 0;" +
                                         " 0.311254483294 0.00852445628482 0.231 0;" +
                                         " 0.282816832044 -0.457959483281 -0.111208333333 0.231]"));


    Dune::DynamicVector< double > d(Gamma.rows());
    for (size_t ii = 0; ii < Gamma.rows(); ++ii) {
      d[ii] = 0.0;
      for (size_t jj = 0; jj <= ii; ++jj)
        d[ii] += Gamma[ii][jj];
    }


    solve_rosenbrock(u,
                     t_end,
                     dt,
                     dx,
                     dmu,
                     saveInterval,
                     true,
                     true,
                     "finite_difference_grk4t",
                     solution,
                     A,
                     b_1,
                     b_2,
                     c,
                     d,
                     Gamma);

    typedef Dune::YaspGrid< dimDomain, Dune::EquidistantOffsetCoordinates< double, dimDomain > >  XGridType;

    Dune::Stuff::Common::Configuration x_grid_config;
    x_grid_config["type"] = "provider.cube";
//    x_grid_config["lower_left"] = "[-0.5]";
//    x_grid_config["upper_right"] = "[0.5]";
    x_grid_config["lower_left"] = "[0]";
    x_grid_config["upper_right"] = "[3]";
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
      u_integrated.visualize("fd_integrated_grk4t" + DSC::toString(counter));
      ++counter;
    }

    std::cout << " done" << std::endl;
    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

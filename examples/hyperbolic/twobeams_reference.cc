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

#include <dune/stuff/common/string.hh>
#include <dune/stuff/common/profiler.hh>
#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/grid/search.hh>
#include <dune/stuff/functions/constant.hh>
#include <dune/stuff/la/container/common.hh>
#include <dune/stuff/la/container/pattern.hh>

#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/projections.hh>
#include <dune/gdt/spaces/fv/default.hh>


using namespace Dune::GDT;

/**
 * Writes a DiscreteFunction step representing the solution at time t to filename.csv. Each row represents one time
 * step. The values in each row are put in order by the iterator of the grid_view of step. Using a 1D YaspGrid the
 * values thus are sorted as you would expect with the value for the leftmost entity in the leftmost column and the
 * value for the rightmost entity in the rightmost column. For 2D or other 1D grids the ordering may be different.
 * The corresponding time t is written to filename_timesteps.csv. append controls if the new data is appended to
 * an old file, if existing, or if the old file is removed.
 **/
template< class DiscreteFunctionType >
void write_step_to_csv(const double t,
                       const DiscreteFunctionType& step,
                       const std::string filename,
                       const bool append = true)
{
  // get grid_view
  const auto& grid_view = step.space().grid_view();
  //remove file if already existing
  const std::string filenamecsv = filename + ".csv";
  const std::string timefilename = filename + "_timesteps" + ".csv";
  if (!append) {
    remove(filenamecsv.c_str());
    remove(timefilename.c_str());
  }
  // open file
  std::ofstream output_file(filenamecsv, std::ios_base::app);
  std::ofstream time_output_file(timefilename, std::ios_base::app);
  time_output_file << DSC::toString(t) << std::endl;
  const auto it_end = grid_view.template end< 0 >();
  const auto const_it_begin = grid_view.template begin< 0 >();
  for (auto it = grid_view.template begin< 0 >(); it != it_end; ++it) {
    const auto& entity = *it;
    if (it != const_it_begin)
      output_file << ", ";
    output_file << DSC::toString(step.local_discrete_function(entity)->evaluate(entity.geometry().local(entity.geometry().center()))[0]);
  }
  output_file << std::endl;
  output_file.close();
}

/**
 * Problem definitions, notation as in Schneider/Alldredge 2014, dx.doi.org/10.1137/130934210
 * To add a test case, add its name to the TestCase enum and provide a specialization of Problem.
 */
enum class TestCase { TwoBeams, SourceBeam };

template< class EntityType2D, TestCase test_case >
class Problem
{
public:
  typedef typename DS::Functions::Constant< EntityType2D, double, 2, double, 1, 1 > ConstantFunctionType;

  static inline double sigma_a(const double /*t*/,const double /*x*/);

  static inline double T(const double /*t*/, const double /*x*/);

  static inline double Q(const double /*t*/, const double /*x*/, const double /*mu*/);

  static inline double boundary_conditions_left(const double psi,
                                         const double mu,
                                         const bool on_top_boundary,
                                         const double dmu);

  static inline double boundary_conditions_right(const double psi,
                                          const double mu,
                                          const bool on_bottom_boundary,
                                          const double dmu);

  static ConstantFunctionType initial_values();

  static DSC::Configuration default_grid_config();
}; // unspecialized Problem

// TwoBeams
template< class EntityType2D >
class Problem< EntityType2D, TestCase::TwoBeams >
{
public:
  typedef typename DS::Functions::Constant< EntityType2D, double, 2, double, 1, 1 > ConstantFunctionType;

  static inline double sigma_a(const double /*t*/,const double /*x*/)
  {
    return 4.0;
  }

  static inline double T(const double /*t*/, const double /*x*/)
  {
    return 0;
  }

  static inline double Q(const double /*t*/, const double /*x*/, const double /*mu*/)
  {
    return 0;
  }

  static inline double boundary_conditions_left(const double psi,
                                         const double mu,
                                         const bool on_top_boundary,
                                         const double dmu)
  {
    if (mu > 0)
      return on_top_boundary ? 50.0/dmu : 0.0;
    else
      return psi;
  }

  static inline double boundary_conditions_right(const double psi,
                                          const double mu,
                                          const bool on_bottom_boundary,
                                          const double dmu)
  {
    if (mu < 0)
      return on_bottom_boundary ? 50.0/dmu : 0.0;
    else
      return psi;
  }

  static ConstantFunctionType initial_values()
  {
    return ConstantFunctionType(0.0001);
  }

  static DSC::Configuration default_grid_config()
  {
    DSC::Configuration grid_config_2d;
    grid_config_2d["type"] = "provider.cube";
    grid_config_2d["lower_left"] = "[-0.5 -1.0]";
    grid_config_2d["upper_right"] = "[0.5 1.0]";
    grid_config_2d["num_elements"] = "[100 200]";
    return grid_config_2d;
  }
}; // Problem< ..., TwoBeams >

// SourceBeam
template< class EntityType2D >
class Problem< EntityType2D, TestCase::SourceBeam >
{
public:
  typedef typename DS::Functions::Constant< EntityType2D, double, 2, double, 1, 1 > ConstantFunctionType;

  static inline double sigma_a(const double /*t*/, const double x)
  {
    return x > 2.0 ? 0.0 : 1.0;
  }

  static inline double T(const double /*t*/, const double x)
  {
    if (x > 2.0)
      return 10.0;
    else if (x > 1.0)
      return 2.0;
    else
      return 0.0;
  }

  static inline double Q(const double /*t*/, const double x, const double /*mu*/)
  {
    return x < 1 || x > 1.5 ? 0.0 : 1.0;
  }

  static inline double boundary_conditions_left(const double psi,
                                         const double mu,
                                         const bool on_top_boundary,
                                         const double dmu)
  {
    if (mu > 0)
      return on_top_boundary ? 0.5/dmu : 0.0;
    else
      return psi;
  }

  static inline double boundary_conditions_right(const double psi,
                                          const double mu,
                                          const bool /*on_bottom_boundary*/,
                                          const double /*dmu*/)
  {
    if (mu < 0)
      return 0.0001;
    else
      return psi;
  }

  static ConstantFunctionType initial_values()
  {
    return ConstantFunctionType(0.0001);
  }

  static DSC::Configuration default_grid_config()
  {
    DSC::Configuration grid_config_2d;
    grid_config_2d["type"] = "provider.cube";
    grid_config_2d["lower_left"] = "[0 -1.0]";
    grid_config_2d["upper_right"] = "[3 1.0]";
    grid_config_2d["num_elements"] = "[300 200]";
    return grid_config_2d;
  }
}; // Problem< ..., SourceBeam >


//! forward or backward finite difference depending on the sign of mu
inline double finite_difference(const double psi_iplus1,
                                const double psi_i,
                                const double psi_iminus1,
                                const double dx,
                                const double mu)
{
 return mu < 0 ? (psi_iplus1-psi_i)/dx : (psi_i-psi_iminus1)/dx;
}

/**
 * Creates a map from entity indices to a pair containing a vector of indices of its neighbors and the
 * entity's coordinates. The order of the neighbors in the first part of the pair is left, right, top, bottom. If the
 * entity is on the boundary, the indices of the non-existing neighbors are set to IndexType(-1), which is, as IndexType
 * is unsigned, the highest number IndexType can represent.
 * This map is created in the first time step in walk_grid_parallel (see below) and used in all subsequent steps instead
 * of the grid_view. Without this map, the neighbors had to be found in every time step which is quite slow.
 */
template< class DiscreteFunctionType, class EntityRange >
auto create_map(const DiscreteFunctionType& psi, const EntityRange& entity_range)
             -> std::map< typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType,
                          typename std::pair<
             std::vector< typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType >,
             typename DiscreteFunctionType::SpaceType::GridViewType::template Codim< 0 >::Geometry::GlobalCoordinate > >
{
  typedef typename DiscreteFunctionType::SpaceType::GridViewType GridViewType;
  typedef typename GridViewType::IndexSet::IndexType IndexType;
  typedef typename GridViewType::template Codim< 0 >::Geometry::GlobalCoordinate CoordinateType;
  typedef typename std::map< IndexType, typename std::pair< std::vector< IndexType >, CoordinateType > > MapType;
  MapType index_map;

  const auto& mapper = psi.space().mapper();
  const auto& grid_view = psi.space().grid_view();
  IndexType left_index, right_index, top_index, bottom_index, entity_index;
  for (const auto& entity : entity_range) {
    std::vector< IndexType > indices(4);
    const CoordinateType entity_coords = entity.geometry().center();
    entity_index = mapper.mapToGlobal(entity, 0);
    const auto& x = entity_coords[0];
    const auto& mu = entity_coords[1];
    // find neighbors
    const auto i_it_end = grid_view.iend(entity);
    for (auto i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
      const auto& intersection = *i_it;
      const auto intersection_coords = intersection.geometry().center();
      if (DSC::FloatCmp::eq(intersection_coords[0], x)) {
        if (intersection_coords[1] > mu) {
          if (intersection.neighbor())
            top_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            top_index = IndexType(-1);
        } else {
          if (intersection.neighbor())
            bottom_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            bottom_index = IndexType(-1);
        }
      } else if (DSC::FloatCmp::eq(intersection_coords[1], mu)) {
        if (intersection_coords[0] > x) {
          if (intersection.neighbor())
            right_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            right_index = IndexType(-1);
        } else {
          if (intersection.neighbor())
            left_index = mapper.mapToGlobal(intersection.outside(), 0);
          else
            left_index = IndexType(-1);
        }
      } else {
        DUNE_THROW(Dune::InvalidStateException, "This should not happen!");
      }
    }
    indices[0] = left_index;
    indices[1] = right_index;
    indices[2] = top_index;
    indices[3] = bottom_index;
    index_map.insert(std::make_pair(entity_index, std::make_pair(indices, entity_coords)));
  }
  return index_map;
}

/**
 * Computes the update vector for the finite difference scheme. If stage is 0, it also computes the negative jacobian
 * that is needed in the Rosenbrock-type schemes.
 * At the first call (per thread), a map is created that maps the index of each entity in entity_range to its
 * coordinates and the indices of its neighbors (see create_map). After that, the entity_range is ignored and the grid
 * walk is done by iterating over the list. Thus, the partitioning of the grid with respect to the threads will be the
 * same for all time steps. If you want to change the partitioning over time or if the index map becomes to
 * memory-intensive, use the commented code.
 */
template< class EntityRange, class DiscreteFunctionType, class ProblemType >
void walk_grid_parallel_rosenbrock(const EntityRange& entity_range,
                                   const DiscreteFunctionType& psi,
                                   DiscreteFunctionType& psi_update,
                                   const double t,
                                   const double dx,
                                   const double dmu,
                                   DS::LA::EigenRowMajorSparseMatrix< double >& negative_jacobian,
                                   const size_t stage)
{
  // use this code if index_map becomes too memory-intensive
//  const auto& mapper = psi.space().mapper();
//  const auto& grid_view = psi.space().grid_view();
//  const auto& psi_vector = psi.vector();
//  auto& psi_update_vector = psi_update.vector();
//  typedef typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType IndexType;

//  IndexType left_index, right_index, top_index, bottom_index, entity_index;
//  for (const auto& entity : entity_range) {
//    bool on_left_boundary(false);
//    bool on_right_boundary(false);
//    bool on_top_boundary(false);
//    bool on_bottom_boundary(false);
//    const auto entity_coords = entity.geometry().center();
//    entity_index = mapper.mapToGlobal(entity, 0);
//    const auto& x = entity_coords[0];
//    const auto& mu = entity_coords[1];
//    // find neighbors
//    const auto i_it_end = grid_view.iend(entity);
//    for (auto i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
//      const auto& intersection = *i_it;
//      const auto intersection_coords = intersection.geometry().center();
//      if (DSC::FloatCmp::eq(intersection_coords[0], x)) {
//        if (intersection_coords[1] > mu) {
//          if (intersection.neighbor())
//            top_index = mapper.mapToGlobal(intersection.outside(), 0);
//          else
//            on_top_boundary = true;
//        } else {
//          if (intersection.neighbor())
//            bottom_index = mapper.mapToGlobal(intersection.outside(), 0);
//          else
//            on_bottom_boundary = true;
//        }
//      } else if (DSC::FloatCmp::eq(intersection_coords[1], mu)) {
//        if (intersection_coords[0] > x) {
//          if (intersection.neighbor())
//            right_index = mapper.mapToGlobal(intersection.outside(), 0);
//          else
//            on_right_boundary = true;
//        } else {
//          if (intersection.neighbor())
//            left_index = mapper.mapToGlobal(intersection.outside(), 0);
//          else
//            on_left_boundary = true;
//        }
//      } else {
//        DUNE_THROW(Dune::InvalidStateException, "This should not happen!");
//      }
//    }

//    const auto psi_i_k = psi_vector[entity_index];
//    const auto psi_iplus1_k = on_right_boundary ? boundary_conditions_right(psi_i_k, mu, on_bottom_boundary, dmu) : psi_vector[right_index];
//    const auto psi_iminus1_k = on_left_boundary ? boundary_conditions_left(psi_i_k, mu, on_top_boundary, dmu) : psi_vector[left_index];
//    const auto psi_i_kplus1 = on_top_boundary ? psi_i_k : psi_vector[top_index];
//    const auto psi_i_kminus1 = on_bottom_boundary ? psi_i_k : psi_vector[bottom_index];
//    // finite difference scheme update formula
//    psi_update_vector[entity_index] = -mu * finite_difference(psi_iplus1_k, psi_i_k, psi_iminus1_k, dx, mu)
//                                      - ProblemType::sigma_a(t,x)*psi_i_k
//                                      + ProblemType::Q(t,x,mu)
//                                      + 0.5*ProblemType::T(t,x) *
//                                      ( (1.0-mu*mu)*(psi_i_kplus1 - 2.0*psi_i_k + psi_i_kminus1)/(dmu*dmu)
//                                        - 2.0*mu*finite_difference(psi_i_kplus1, psi_i_k, psi_i_kminus1, dmu, mu) );

//    // in the first step of the Rosenbrock-type schemes, we have to assemble the jacobian
//    if (stage == 0) {
//      negative_jacobian.set_entry(entity_index, entity_index,
//                                  mu < 0
//                                  ? -mu/dx + ProblemType::sigma_a(t,x)
//                                    + ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) - mu/dmu)
//                                  :  mu/dx + ProblemType::sigma_a(t,x)
//                                    + ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) + mu/dmu));
//      if (!on_right_boundary)
//        negative_jacobian.set_entry(entity_index, right_index, mu < 0 ? mu/dx : 0.0);
//      if (!on_left_boundary)
//        negative_jacobian.set_entry(entity_index, left_index, mu < 0 ? 0.0 : -mu/dx);
//      if (!on_bottom_boundary)
//        negative_jacobian.set_entry(entity_index, bottom_index,
//                                    mu < 0
//                                    ? -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu))
//                                    : -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) + 2.0*mu/dmu));
//      if (!on_top_boundary)
//        negative_jacobian.set_entry(entity_index, top_index,
//                                    mu < 0
//                                    ? -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) - 2.0*mu/dmu)
//                                    : -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu)));
//    }
//  }

  const auto& psi_vector = psi.vector();
  auto& psi_update_vector = psi_update.vector();

  typedef typename DiscreteFunctionType::SpaceType::GridViewType::IndexSet::IndexType IndexType;

  thread_local static auto index_map = create_map(psi, entity_range);

  for (const auto& pair : index_map) {
    const auto& entity_index = pair.first;
    const auto& indices_pair = pair.second;
    const auto& indices = indices_pair.first;
    const auto& left_index = indices[0];
    const auto& right_index = indices[1];
    const auto& top_index = indices[2];
    const auto& bottom_index = indices[3];
    const auto& coords = indices_pair.second;
    const auto& x = coords[0];
    const auto& mu = coords[1];
    const bool on_left_boundary = left_index == IndexType(-1);
    const bool on_right_boundary = right_index == IndexType(-1);
    const bool on_top_boundary = top_index == IndexType(-1);
    const bool on_bottom_boundary = bottom_index == IndexType(-1);

    // i indices refer to the x coordinate, k indices to mu
    const auto psi_i_k = psi_vector[entity_index];
    // apply boundary conditions from problem at left and right boundary
    const auto psi_iplus1_k = on_right_boundary
                              ? ProblemType::boundary_conditions_right(psi_i_k, mu, on_bottom_boundary, dmu)
                              : psi_vector[right_index];
    const auto psi_iminus1_k = on_left_boundary
                               ? ProblemType::boundary_conditions_left(psi_i_k, mu, on_top_boundary, dmu)
                               : psi_vector[left_index];
    // zeroth order interpolation on top and bottom boundary
    const auto psi_i_kplus1 = on_top_boundary ? psi_i_k : psi_vector[top_index];
    const auto psi_i_kminus1 = on_bottom_boundary ? psi_i_k : psi_vector[bottom_index];
    // finite difference scheme update formula
    psi_update_vector[entity_index] = -mu * finite_difference(psi_iplus1_k, psi_i_k, psi_iminus1_k, dx, mu)
                                      - ProblemType::sigma_a(t,x)*psi_i_k
                                      + ProblemType::Q(t,x,mu)
                                      + 0.5*ProblemType::T(t,x) *
                                          ( (1.0-mu*mu)*(psi_i_kplus1 - 2.0*psi_i_k + psi_i_kminus1)/(dmu*dmu)
                                           - 2.0*mu*finite_difference(psi_i_kplus1, psi_i_k, psi_i_kminus1, dmu, mu) );

    // in the first step of the Rosenbrock-type schemes, we have to assemble the jacobian
    if (stage == 0) {
      negative_jacobian.set_entry(entity_index, entity_index,
                                  mu < 0
                                  ? -mu/dx + ProblemType::sigma_a(t,x)
                                    + ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) - mu/dmu)
                                  :  mu/dx + ProblemType::sigma_a(t,x)
                                    + ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) + mu/dmu));
      if (!on_right_boundary)
        negative_jacobian.set_entry(entity_index, right_index, mu < 0 ? mu/dx : 0.0);
      if (!on_left_boundary)
        negative_jacobian.set_entry(entity_index, left_index, mu < 0 ? 0.0 : -mu/dx);
      if (!on_bottom_boundary)
        negative_jacobian.set_entry(entity_index, bottom_index,
                                    mu < 0
                                    ? -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu))
                                    : -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) + 2.0*mu/dmu));
      if (!on_top_boundary)
        negative_jacobian.set_entry(entity_index, top_index,
                                    mu < 0
                                    ? -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu) - 2.0*mu/dmu)
                                    : -0.5 * ProblemType::T(t,x) * ((1.0 - mu * mu)/(dmu * dmu)));
    }
  } // iterate over index_map
} // walk_grid_parallel_rosenbrock


//! for Runge-Kutta time stepping, we do not need to assemble the matrix
template< class EntityRange, class DiscreteFunctionType, class ProblemType >
void walk_grid_parallel(const EntityRange& entity_range,
                        const DiscreteFunctionType& psi,
                        DiscreteFunctionType& psi_update,
                        const double t,
                        const double dx,
                        const double dmu)
{
  static DS::LA::EigenRowMajorSparseMatrix< double > fake_jacobian;
  // the 1 as last argument is arbitrary chosen not to be zero such that the jacobian is not assembled
  walk_grid_parallel_rosenbrock< EntityRange, DiscreteFunctionType, ProblemType >(entity_range,
                                                                                   psi,
                                                                                   psi_update,
                                                                                   t,
                                                                                   dx,
                                                                                   dmu,
                                                                                   fake_jacobian,
                                                                                   1);
}



#if HAVE_TBB
  template< class PartitioningType, class DiscreteFunctionType >
  struct Body
  {
    Body(PartitioningType& partitioning,
         const DiscreteFunctionType& psi,
         DiscreteFunctionType& psi_update,
         const double t,
         const double dx,
         const double dmu)
      : partitioning_(partitioning)
      , psi_(psi)
      , psi_update_(psi_update)
      , t_(t)
      , dx_(dx)
      , dmu_(dmu)
    {}

    Body(Body& other, tbb::split /*split*/)
      : partitioning_(other.partitioning_)
      , psi_(other.psi_)
      , psi_update_(other.psi_update_)
      , t_(other.t_)
      , dx_(other.dx_)
      , dmu_(other.dmu_)
    {}

    void operator()(const tbb::blocked_range< std::size_t > &range) const
    {
      // for all partitions in tbb-range
      for(std::size_t p = range.begin(); p != range.end(); ++p) {
        auto partition = partitioning_.partition(p);
        walk_grid_parallel(partition, psi_, psi_update_, t_, dx_, dmu_);
      }
    }

    void join(Body& /*other*/)
    {}

  PartitioningType& partitioning_;
  const DiscreteFunctionType& psi_;
  DiscreteFunctionType& psi_update_;
  const double t_;
  const double dx_;
  const double dmu_;
  }; // struct Body
#endif //HAVE_TBB

#if HAVE_TBB
  template< class PartitioningType, class DiscreteFunctionType, class ProblemType >
  struct Body_rosenbrock
  {
    Body_rosenbrock(PartitioningType& partitioning,
                    const DiscreteFunctionType& psi,
                    DiscreteFunctionType& psi_update,
                    const double t,
                    const double dx,
                    const double dmu,
                    DS::LA::EigenRowMajorSparseMatrix< double >& jacobian,
                    const size_t stage)
      : partitioning_(partitioning)
      , psi_(psi)
      , psi_update_(psi_update)
      , t_(t)
      , dx_(dx)
      , dmu_(dmu)
      , jacobian_(jacobian)
      , stage_(stage)
    {}

    Body_rosenbrock(Body_rosenbrock& other, tbb::split /*split*/)
      : partitioning_(other.partitioning_)
      , psi_(other.psi_)
      , psi_update_(other.psi_update_)
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
        walk_grid_parallel_rosenbrock< decltype(partition), DiscreteFunctionType, ProblemType >(partition, psi_, psi_update_, t_, dx_, dmu_, jacobian_, stage_);
      }
    }

    void join(Body_rosenbrock& /*other*/)
    {}

  PartitioningType& partitioning_;
  const DiscreteFunctionType& psi_;
  DiscreteFunctionType& psi_update_;
  const double t_;
  const double dx_;
  const double dmu_;
  DS::LA::EigenRowMajorSparseMatrix< double >& jacobian_;
  const size_t stage_;
  }; // struct Body_rosenbrock
#endif //HAVE_TBB


template <class DiscreteFunctionType>
void apply_finite_difference(const DiscreteFunctionType& psi,
                             DiscreteFunctionType& psi_update,
                             const double t,
                             const double dx,
                             const double dmu)
{
  typedef typename DiscreteFunctionType::SpaceType::GridViewType GridViewType;
#if DUNE_VERSION_NEWER(DUNE_COMMON,3,9) && HAVE_TBB //EXADUNE
    static const auto num_partitions = DSC_CONFIG_GET("threading.partition_factor", 1u)
                                * DS::threadManager().current_threads();
    static const auto partitioning = DSC::make_unique< Dune::RangedPartitioning< GridViewType, 0 > >(psi.space().grid_view(), num_partitions);
    static tbb::blocked_range< std::size_t > blocked_range(0, partitioning->partitions());
    Body< Dune::RangedPartitioning< GridViewType, 0 >, DiscreteFunctionType > body(*partitioning,
                                                                                   psi,
                                                                                   psi_update,
                                                                                   t,
                                                                                   dx,
                                                                                   dmu);
    tbb::parallel_reduce(blocked_range, body);
#endif
}

template <class DiscreteFunctionType, class ProblemType >
void apply_finite_difference_rosenbrock(const DiscreteFunctionType& psi,
                                        DiscreteFunctionType& psi_update,
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
    const auto partitioning = DSC::make_unique< Dune::RangedPartitioning< GridViewType, 0 > >(psi.space().grid_view(), num_partitions);
    tbb::blocked_range< std::size_t > blocked_range(0, partitioning->partitions());
    Body_rosenbrock< Dune::RangedPartitioning< GridViewType, 0 >, DiscreteFunctionType, ProblemType > body(*partitioning,
                                                                                              psi,
                                                                                              psi_update,
                                                                                              t,
                                                                                              dx,
                                                                                              dmu,
                                                                                              jacobian,
                                                                                              stage);
    tbb::parallel_reduce(blocked_range, body);
#endif
}

template< class DiscreteFunctionType >
DiscreteFunctionType create_first_last_stage(const DiscreteFunctionType& psi,
                                             const double t,
                                             const double dx,
                                             const double dmu)
{
  DiscreteFunctionType last_stage_of_last_step = psi;
  last_stage_of_last_step.vector() *= 0.0;
  apply_finite_difference(psi, last_stage_of_last_step, t, dx, dmu);
  return last_stage_of_last_step;
}

template< class DiscreteFunctionType >
double step(double& t,
            const double initial_dt,
            const double dx,
            const double dmu,
            DiscreteFunctionType& psi_n,
            Dune::DynamicMatrix< double >& A,
            Dune::DynamicVector< double >& b_1,
            Dune::DynamicVector< double >& b_2,
            Dune::DynamicVector< double >& c,
            const double TOL = 0.0001)
{
    static DiscreteFunctionType last_stage_of_last_step =  create_first_last_stage(psi_n, t, dx, dmu);
    static const auto num_stages = A.rows();
    static std::vector< DiscreteFunctionType > psi_intermediate_stages(num_stages, last_stage_of_last_step);
    static const auto b_diff = b_2 - b_1;
    static auto psi_n_tmp = psi_n;
    double mixed_error = 10.0;
    double dt = initial_dt;
    static double scale_max = 6;
    double scale_factor = 1.0;
    static auto diff_vector = psi_n.vector();
    static auto diff_vector_size = diff_vector.size();

    auto& psi_n_vector = psi_n.vector();

    while (mixed_error > TOL) {
      dt *= scale_factor;

      psi_intermediate_stages[0].vector() = last_stage_of_last_step.vector();
      for (size_t ii = 1; ii < num_stages; ++ii) {
//        std::fill(psi_intermediate_stages[ii].vector().begin(), psi_intermediate_stages[ii].vector().end(), 0.0);
        psi_n_tmp.vector() = psi_n_vector;
        for (size_t jj = 0; jj < ii; ++jj)
          psi_n_tmp.vector().axpy(dt*(A[ii][jj]), psi_intermediate_stages[jj].vector());
        apply_finite_difference(psi_n_tmp, psi_intermediate_stages[ii], t+c[ii]*dt, dx, dmu);
      }

      diff_vector = psi_intermediate_stages[0].vector()*b_diff[0];
      for (size_t ii = 1; ii < num_stages; ++ii)
        diff_vector.axpy(b_diff[ii], psi_intermediate_stages[ii].vector());
      diff_vector *= dt;
      // scaling, use absolute error if norm is less than 0.01, maybe should use psi_nplus1 instead of psi_n for relative error
      for (size_t ii = 0; ii < diff_vector_size; ++ii)
        diff_vector[ii] *= std::abs(psi_n_vector[ii]) > 0.01 ? std::abs(psi_n_vector[ii]) : 1.0;
      mixed_error = diff_vector.sup_norm();
//      std::cout << mixed_error << std::endl;
      scale_factor = std::min(std::max(0.9*std::pow(TOL/mixed_error, 1.0/5.0), 0.2), scale_max);
//      std::cout << "Scale: " << scale_factor << std::endl;
    }

    for (size_t ii = 0; ii < num_stages; ++ii) {
      psi_n_vector.axpy(dt*b_1[ii], psi_intermediate_stages[ii].vector());
    }

    last_stage_of_last_step.vector() = psi_intermediate_stages[num_stages - 1].vector();

    t += dt;

//    std::cout << t << " and dt " << dt << std::endl;

    return dt*scale_factor;
}

template< class DiscreteFunctionType, class ProblemType >
double step_rosenbrock(double& t,
                       const double initial_dt,
                       const double dx, const double dmu,
                       DiscreteFunctionType& psi_n,
                       Dune::DynamicMatrix< double >& A_new,
                       Dune::DynamicVector< double >& m_1,
                       Dune::DynamicVector< double >& m_2,
                       Dune::DynamicVector< double >& c,
                       Dune::DynamicVector< double >& /*d*/,
                       Dune::DynamicMatrix< double >& C,
                       const double gamma,
                       DS::LA::EigenRowMajorSparseMatrix< double >& negative_jacobian,
                       DS::LA::EigenRowMajorSparseMatrix< double >& system_matrix,
                       const double TOL)
{
  typedef typename DS::LA::Solver< typename DS::LA::EigenRowMajorSparseMatrix< double > > SolverType;
  static std::unique_ptr< SolverType > solver = DSC::make_unique< SolverType >(system_matrix);

  static const auto num_stages = A_new.rows();
  thread_local static std::vector< DiscreteFunctionType > psi_intermediate_stages(num_stages, psi_n);
  static const auto m_diff = m_2 - m_1;
  static auto psi_n_tmp = psi_n;
  static auto k_i_tmp = psi_n;
  static auto k_sum = psi_n;
  double mixed_error = 10.0;
  double dt = initial_dt;
  static double scale_max = 6;
  double scale_factor = 1.0;
  static auto diff_vector = psi_n.vector();
  static auto diff_vector_size = diff_vector.size();

  while (mixed_error > TOL) {
    dt *= scale_factor;

    for (size_t ii = 0; ii < num_stages; ++ii) {
      psi_n_tmp.vector() = psi_n.vector();
      for (size_t jj = 0; jj < ii; ++jj)
        psi_n_tmp.vector().axpy(A_new[ii][jj], psi_intermediate_stages[jj].vector());
      apply_finite_difference_rosenbrock< DiscreteFunctionType, ProblemType >(psi_n_tmp, k_i_tmp, t+c[ii]*dt, dx, dmu, negative_jacobian, ii);
      // as C_ii is the same for all i, we only need to calculate the matrix in the first step
      if (ii == 0) {
        // create solver
        system_matrix = negative_jacobian;
        for (size_t row = 0; row < system_matrix.rows(); ++row)
          system_matrix.add_to_entry(row, row, 1.0/(gamma*dt));
        solver = DSC::make_unique< SolverType >(system_matrix);
      }
      // fuer explizit zeitabhaengige Funktionen fehlt hier ein Term (siehe Wikipedia)
      // ...
      //
      for (size_t jj = 0; jj < ii; ++jj)
        k_i_tmp.vector().axpy(C[ii][jj]/dt, psi_intermediate_stages[jj].vector());
      // solve
      solver->apply(k_i_tmp.vector(), psi_intermediate_stages[ii].vector());
    }

    // calculate error
    diff_vector = psi_intermediate_stages[0].vector()*m_diff[0];
    for (size_t ii = 1; ii < num_stages; ++ii) {
      diff_vector.axpy(m_diff[ii], psi_intermediate_stages[ii].vector());
    }
    // scaling, use absolute error if norm is less than 0.01, maybe should use psi_nplus1 instead of psi_n for relative error
    for (size_t ii = 0; ii < diff_vector_size; ++ii)
      diff_vector[ii] *= std::abs(psi_n.vector()[ii]) > 0.01 ? std::abs(psi_n.vector()[ii]) : 1.0;
    mixed_error = diff_vector.sup_norm();
//    std::cout << rel_error << std::endl;
    scale_factor = std::min(std::max(0.9*std::pow(TOL/mixed_error, 1.0/5.0), 0.2), scale_max);
  }

  for (size_t ii = 0; ii < num_stages; ++ii) {
    psi_n.vector().axpy(m_1[ii], psi_intermediate_stages[ii].vector());
  }

  t += dt;

  std::cout << t << " and dt " << dt << std::endl;

  return dt*scale_factor;
}


template <class FDDiscreteFunction, class IntegratedDiscretFunctionType>
void integrate_over_mu(const FDDiscreteFunction& psi, IntegratedDiscretFunctionType& psi_integrated, const double dmu)
{
  // get grid views and mapper
  typedef typename FDDiscreteFunction::SpaceType::GridViewType GridViewType2D;
  typedef typename IntegratedDiscretFunctionType::SpaceType::GridViewType GridViewType1D;
  static const auto& grid_view_1d = psi_integrated.space().grid_view();
  static const auto& grid_view_2d = psi.space().grid_view();
  static const auto& mapper_1d = psi_integrated.space().mapper();
  static const auto& mapper_2d = psi.space().mapper();
  static const auto grid_view_2d_size = grid_view_2d.size(0);
  // create entitysearch on 1d grid
  static auto entity_search = DSG::EntityInlevelSearch< GridViewType1D >(grid_view_1d);
  // create vectors to store the x coordinate and the index of each entity on the 2d grid
  static std::vector< typename GridViewType2D::ctype > entity_x_vector(grid_view_2d_size);
  static std::vector< typename GridViewType2D::IndexSet::IndexType > entity_index_vector(grid_view_2d_size);
  // walk 2d grid to find x coordinate and index of each entity
  static const auto it_end = grid_view_2d.template end< 0 >();
  size_t counter = 0;
  for (auto it = grid_view_2d.template begin< 0 >(); it != it_end; ++it, ++counter) {
    const auto& entity = *it;
    entity_index_vector[counter] = mapper_2d.mapToGlobal(entity,0);
    const auto entity_coords = entity.geometry().center();
    entity_x_vector[counter] = entity_coords[0];
  }
  // find entities in 1d grid using the x coordinates we collected before
  const auto x_entities = entity_search(entity_x_vector);
  // integrate over all entities with the same x coordinate in the 2d grid
  for(size_t ii = 0; ii < grid_view_2d_size; ++ii) {
    assert(x_entities[ii] != nullptr);
    psi_integrated.vector()[mapper_1d.mapToGlobal(*x_entities[ii],0)] += psi.vector()[entity_index_vector[ii]];
  }
  psi_integrated.vector() *= dmu;
}


template <class DiscreteFunctionType, class XFVSpaceType>
void solve(DiscreteFunctionType& psi_n,
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
           const XFVSpaceType& x_fvspace,
           const double TOL)
{
  double t_ = 0;
  double dt = first_dt;
  assert(t_end - t_ >= dt);
  size_t time_step_counter = 0;

  const auto& x_grid_view = x_fvspace.grid_view();

  typedef DiscreteFunction< XFVSpaceType, Dune::Stuff::LA::CommonDenseVector< double > > XFVFunctionType;

  const double save_interval = DSC::FloatCmp::eq(save_step_length, 0.0) ? dt : save_step_length;
  double next_save_time = t_ + save_interval > t_end ? t_end : t_ + save_interval;
  size_t save_step_counter = 1;

  // clear solution
  if (save_solution) {
    solution.clear();
    solution.emplace_back(std::make_pair(t_, psi_n));
  }
  if (write_solution) {
//    psi_n.visualize(filename_prefix + "_0");
    XFVFunctionType psi_integrated(x_fvspace, "x_solution");
    integrate_over_mu(psi_n, psi_integrated, dmu);
    psi_integrated.visualize(filename_prefix + "_0");
    write_step_to_csv(x_grid_view, t_, psi_integrated, filename_prefix + ".csv", false);
  }

  while (t_ + dt < t_end)
  {
    // do a timestep
    dt = step(t_, dt, dx, dmu, psi_n, A, b_1, b_2, c, TOL);

    // check if data should be written in this timestep (and write)
    if (DSC::FloatCmp::ge(t_, next_save_time - 1e-10)) {
      if (save_solution)
        solution.emplace_back(std::make_pair(t_, psi_n));
      if (write_solution) {
//        psi_n.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
        XFVFunctionType psi_integrated(x_fvspace, "x_solution");
        integrate_over_mu(psi_n, psi_integrated, dmu);
        psi_integrated.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
        write_step_to_csv(x_grid_view, t_, psi_integrated, filename_prefix + ".csv");
        std::cout << t_ << " and dt " << dt << std::endl;
      }
      next_save_time += save_interval;
      ++save_step_counter;
    }

    // augment time step counter
    ++time_step_counter;
  } // while (t_ < t_end)

  // do last step s.t. it matches t_end exactly
  if (!DSC::FloatCmp::ge(t_, t_end - 1e-10)) {
    step(t_, t_end - t_, dx, dmu, psi_n, A, b_1, b_2, c, TOL);
    if (save_solution)
      solution.emplace_back(std::make_pair(t_, psi_n));
    if (write_solution) {
//        psi_n.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
      XFVFunctionType psi_integrated(x_fvspace, "x_solution");
      integrate_over_mu(psi_n, psi_integrated, dmu);
      psi_integrated.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
      write_step_to_csv(x_grid_view, t_, psi_integrated, filename_prefix + ".csv");
      std::cout << t_ << " and dt " << dt << std::endl;
    }
  }
} // ... solve(...)

template< class DiscreteFunctionType >
Dune::Stuff::LA::SparsityPatternDefault assemble_pattern(DiscreteFunctionType& psi_n)
{
  const auto& mapper = psi_n.space().mapper();
  const auto& grid_view = psi_n.space().grid_view();
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



template <class DiscreteFunctionType, class XFVSpaceType, class ProblemType >
void solve_rosenbrock(DiscreteFunctionType& psi_n,
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
                      Dune::DynamicMatrix< double > Gamma,
                      const XFVSpaceType& x_fvspace,
                      const double TOL)
{
  double t_ = 0;
  double dt = first_dt;
  assert(t_end - t_ >= dt);
  size_t time_step_counter = 0;

  const auto& x_grid_view = x_fvspace.grid_view();

  typedef DiscreteFunction< XFVSpaceType, Dune::Stuff::LA::CommonDenseVector< double > > XFVFunctionType;

  const double save_interval = DSC::FloatCmp::eq(save_step_length, 0.0) ? dt : save_step_length;
  double next_save_time = t_ + save_interval > t_end ? t_end : t_ + save_interval;
  size_t save_step_counter = 1;

  // clear solution
  if (save_solution) {
    solution.clear();
    solution.emplace_back(std::make_pair(t_, psi_n));
  }

  if (write_solution) {
//    psi_n.visualize(filename_prefix + "_0");
    XFVFunctionType psi_integrated(x_fvspace, "x_solution");
    integrate_over_mu(psi_n, psi_integrated, dmu);
    psi_integrated.visualize(filename_prefix + "_0");
    write_step_to_csv(t_, psi_integrated, filename_prefix + ".csv", false);
  }

  const auto pattern = assemble_pattern(psi_n);

  const auto num_grid_elements = psi_n.space().grid_view().size(0);

  DS::LA::EigenRowMajorSparseMatrix< double > jacobian(num_grid_elements, num_grid_elements, pattern);
  DS::LA::EigenRowMajorSparseMatrix< double > system_matrix(num_grid_elements, num_grid_elements, pattern);

  // transform variables for faster calculations, C = diag(1/gamma, ... , 1/gamma) - Gamma^(-1),
  // A_new = A*Gamma^(-1), m = (b_1,...,b_n)*Gamma^(-1)
  auto Gamma_inv = Gamma;
  Gamma_inv.invert();
  auto C = Gamma_inv;
  C *= -1.0;
  for (size_t ii = 0; ii < C.rows(); ++ii)
    C[ii][ii] += 1.0/(Gamma[ii][ii]);
  A.rightmultiply(Gamma_inv);
//  std::cout << "A: " << DSC::toString(A) << std::endl;
//  std::cout << "C: " << DSC::toString(C) << std::endl;
  auto m_1 = b_1;
  Gamma_inv.mtv(b_1, m_1);
  auto m_2 = b_2;
  Gamma_inv.mtv(b_2, m_2);
  const auto gamma = Gamma[0][0];
//  std::cout << "m_1: " << DSC::toString(m_1) << std::endl;
//  std::cout << "m_2: " << DSC::toString(m_2) << std::endl;

  while (t_ + dt < t_end)
  {
    // do a timestep
    dt = step_rosenbrock< DiscreteFunctionType, ProblemType >(t_, dt, dx, dmu, psi_n, A, m_1, m_2, c, d, C, gamma, jacobian, system_matrix, TOL);

    // check if data should be written in this timestep (and write)
    if (DSC::FloatCmp::ge(t_, next_save_time - 1e-10)) {
      if (save_solution)
        solution.emplace_back(std::make_pair(t_, psi_n));
      if (write_solution) {
//        psi_n.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
        XFVFunctionType psi_integrated(x_fvspace, "x_solution");
        integrate_over_mu(psi_n, psi_integrated, dmu);
        psi_integrated.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
        write_step_to_csv(t_, psi_integrated, filename_prefix + ".csv");
      }
      next_save_time += save_interval;
      ++save_step_counter;
    }

    // augment time step counter
    ++time_step_counter;
  } // while (t_ < t_end)

  // do last step s.t. it matches t_end exactly
  if (!DSC::FloatCmp::ge(t_, t_end - 1e-10)) {
    step_rosenbrock< DiscreteFunctionType, ProblemType >(t_, dt, dx, dmu, psi_n, A, m_1, m_2, c, d, C, gamma, jacobian, system_matrix, TOL);
    if (save_solution)
      solution.emplace_back(std::make_pair(t_, psi_n));
    if (write_solution) {
      //        psi_n.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
      XFVFunctionType psi_integrated(x_fvspace, "x_solution");
      integrate_over_mu(psi_n, psi_integrated, dmu);
      psi_integrated.visualize(filename_prefix + "_" + DSC::toString(save_step_counter));
      write_step_to_csv(t_, psi_integrated, filename_prefix + ".csv");
    }
  }
} // ... solve_rosenbrock(...)

/**
 * main function. Choose problem (TwoBeams or SoureBeam) by uncommenting the right TestCase.
 */
int main(int argc, char* argv[])
{
  try {
    // setup MPI
    typedef Dune::MPIHelper MPIHelper;
    MPIHelper::instance(argc, argv);

    // parse options and use default values for options that are not provided
    if (argc == 1) {
      std::cout << "Usage: " << argv[0] << "-threading.max_count THREADS -filename NAME -gridsize_x GRIDSIZE -gridsize_mu GRIDSIZE "
                << "-TOL TOL -t_end TIME -num_save_steps NUMBER" << std::endl;
      std::cout << "Using default values." << std::endl;
    }

    // setup threadmanager
    DSC_CONFIG.set("threading.partition_factor", 1, true);
    // default values for options
    size_t num_threads = 1;
    std::string filename = "twobeams";
    std::string grid_size_x = "100";
    std::string grid_size_mu = "100";
    double TOL = 0.0001;
    double t_end = 0.02;
    double num_save_steps = 100;
    for (int i = 1; i < argc; i += 2) {
      if (std::string(argv[i]) == "-threading.max_count") {
        if (i + 1 < argc) { // Make sure we aren't at the end of argv!
          num_threads = DSC::fromString< size_t >(argv[i+1]); // Increment 'i' so we don't get the argument as the next argv[i].
          DS::threadManager().set_max_threads(num_threads);
        } else {
          std::cerr << "-threading.max_count option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-filename") {
        if (i + 1 < argc) {
          filename = argv[i+1];
        } else {
          std::cerr << "-filename option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-gridsize_x") {
        if (i + 1 < argc) {
          grid_size_x = argv[i+1];
        } else {
          std::cerr << "-gridsize_x option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-gridsize_mu") {
        if (i + 1 < argc) {
          grid_size_mu = argv[i+1];
        } else {
          std::cerr << "-gridsize_mu option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-TOL") {
        if (i + 1 < argc) {
          TOL = DSC::fromString<double>(argv[i+1]);
        } else {
          std::cerr << "-TOL option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-t_end") {
        if (i + 1 < argc) {
          t_end = DSC::fromString<double>(argv[i+1]);
        } else {
          std::cerr << "-t_end option requires one argument." << std::endl;
          return 1;
        }
      } else if (std::string(argv[i]) == "-num_save_steps") {
        if (i + 1 < argc) {
          num_save_steps = DSC::fromString<double>(argv[i+1]);
        } else {
          std::cerr << "-num_save_steps option requires one argument." << std::endl;
          return 1;
        }
      } else
        std::cerr << "Unrecognized option " << argv[i] << "has been ignored." << std::endl;
    }

    // set dimensions
    static const size_t dimDomain = 1;

    //choose GridTypes, 2D grid is used for calculations, 1D grid for the solution that is integrated over mu
    typedef Dune::YaspGrid< dimDomain, Dune::EquidistantOffsetCoordinates< double, dimDomain > >  GridType1D;
    typedef Dune::YaspGrid< 2*dimDomain, Dune::EquidistantOffsetCoordinates< double, 2*dimDomain > >  GridType2D;
    typedef typename GridType2D::Codim< 0 >::Entity EntityType2D;

    // choose ProblemType
    const TestCase test_case = TestCase::SourceBeam;
//    const TestCase test_case = TestCase::TwoBeams;
    typedef Problem< EntityType2D, test_case > ProblemType;

    //get grid configuration from problem and set the number of elements
    Dune::Stuff::Common::Configuration grid_config = ProblemType::default_grid_config();
    grid_config["num_elements"] = "[" + grid_size_x + " " + grid_size_mu + "]";

    //create grids
    std::cout << "Creating grids..." << std::endl;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType1D >  GridProviderType1D;
    typedef Dune::Stuff::Grid::Providers::Cube< GridType2D >  GridProviderType2D;
    GridProviderType1D grid_provider_1d = *(GridProviderType1D::create(grid_config));
    GridProviderType2D grid_provider_2d = *(GridProviderType2D::create(grid_config));
    const std::shared_ptr< const GridType1D > grid_1d = grid_provider_1d.grid_ptr();
    const std::shared_ptr< const GridType2D > grid_2d = grid_provider_2d.grid_ptr();

    // make finite volume spaces on the leaf grids
    std::cout << "Creating finite volume spaces..." << std::endl;
    typedef typename GridType1D::LeafGridView                                        GridViewType1D;
    typedef typename GridType2D::LeafGridView                                        GridViewType2D;
    const GridViewType1D grid_view_1d = grid_1d->leafGridView();
    const GridViewType2D grid_view_2d = grid_2d->leafGridView();
    typedef Spaces::FV::Default< GridViewType1D, double, dimDomain, 1 >              FVSpaceType1D;
    typedef Spaces::FV::Default< GridViewType2D, double, dimDomain, 1 >            FVSpaceType2D;
    const FVSpaceType1D fv_space_1d(grid_view_1d);
    const FVSpaceType2D fv_space_2d(grid_view_2d);

    // allocate discrete function for psi
    typedef DiscreteFunction< FVSpaceType2D, Dune::Stuff::LA::EigenDenseVector< double > > DiscreteFunctionType2D;
    DiscreteFunctionType2D psi(fv_space_2d, "solution");

    //project initial values
    const auto initial_values = ProblemType::initial_values();
    std::cout << "Projecting initial values..." << std::endl;
    project(initial_values, psi);

    //calculate dx and dmu
    const double x_grid_length = DSC::fromString< std::vector< double > >(grid_config["upper_right"], 2)[0]
                               - DSC::fromString< std::vector< double > >(grid_config["lower_left"], 2)[0];
    const double mu_grid_length = DSC::fromString< std::vector< double > >(grid_config["upper_right"], 2)[1]
                                - DSC::fromString< std::vector< double > >(grid_config["lower_left"], 2)[1];
    const double dx = x_grid_length/(DSC::fromString< size_t >(grid_size_x));
    const double dmu = mu_grid_length/(DSC::fromString< size_t >(grid_size_mu));
    std::cout << "dx: " << dx << " dmu: " << dmu << std::endl;

    // set initial time step length
    const double CFL = 0.1;
    double dt = CFL*dx;
    const double saveInterval = t_end/num_save_steps;

    std::vector< std::pair< double, DiscreteFunctionType2D > > solution;

    // use the following with solve
    // Bogacki-Shampine
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

    // Dormand–Prince
//    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
//                                    (std::string("[0 0 0 0 0 0 0;") +
//                                     " 0.2 0 0 0 0 0 0;" +
//                                     " 0.075 0.225 0 0 0 0 0;" +
//                                     " " + DSC::toString(44.0/45.0, 15) + " " + DSC::toString(-56.0/15.0, 15) + " " + DSC::toString(32.0/9.0, 15) + " 0 0 0 0;" +
//                                     " " + DSC::toString(19372.0/6561.0, 15) + " " + DSC::toString(-25360.0/2187.0, 15) + " " + DSC::toString(64448.0/6561.0, 15) + " " + DSC::toString(-212.0/729.0, 15) + " 0 0 0;" +
//                                     " " + DSC::toString(9017.0/3168.0, 15) + " " + DSC::toString(-355.0/33.0, 15) + " " + DSC::toString(46732.0/5247.0, 15) + " " + DSC::toString(49.0/176.0, 15) + " " + DSC::toString(-5103.0/18656.0, 15) + " 0 0;" +
//                                     " " + DSC::toString(35.0/384.0, 15) + " 0 " + DSC::toString(500.0/1113.0, 15) + " " + DSC::toString(125.0/192.0, 15) + " " + DSC::toString(-2187.0/6784.0, 15) + " " + DSC::toString(11.0/84.0, 15) + " 0]"));

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


    // use the following with solve_rosenbrock
    // GRK4T
//    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
//                                    (std::string("[0 0 0 0;") +
//                                     " 0.462 0 0 0;" +
//                                     " -0.0815668168327 0.961775150166 0 0;" +
//                                     " -0.0815668168327 0.961775150166 0 0]"));
//    Dune::DynamicVector< double > b_1(DSC::fromString< Dune::DynamicVector< double > >("[0.217487371653 0.486229037990 0 0.296283590357]"));
//    Dune::DynamicVector< double > b_2(DSC::fromString< Dune::DynamicVector< double > >("[-0.717088504499 1.77617912176 -0.0590906172617 0]"));
//    Dune::DynamicVector< double > c(DSC::fromString< Dune::DynamicVector< double > >("[0 0 0" + DSC::toString(2.0/3.0, 15) + "]"));
//    Dune::DynamicMatrix< double > Gamma(DSC::fromString< Dune::DynamicMatrix< double > >
//                                        (std::string("[0.231 0 0 0;") +
//                                         " -0.270629667752 0.231 0 0;" +
//                                         " 0.311254483294 0.00852445628482 0.231 0;" +
//                                         " 0.282816832044 -0.457959483281 -0.111208333333 0.231]"));


    // GRK4A
    Dune::DynamicMatrix< double > A(DSC::fromString< Dune::DynamicMatrix< double > >
                                    (std::string("[0 0 0 0;") +
                                     " 0.438 0 0 0;" +
                                     " 0.796920457938 0.0730795420615 0 0;" +
                                     " 0.796920457938 0.0730795420615 0 0]"));
    Dune::DynamicVector< double > b_1(DSC::fromString< Dune::DynamicVector< double > >("[0.199293275701 0.482645235674 0.0680614886256 0.25]"));
    Dune::DynamicVector< double > b_2(DSC::fromString< Dune::DynamicVector< double > >("[0.346325833758  0.285693175712 0.367980990530 0]"));
    Dune::DynamicVector< double > c(DSC::fromString< Dune::DynamicVector< double > >("[0 0 0" + DSC::toString(2.0/3.0, 15) + "]"));
    Dune::DynamicMatrix< double > Gamma(DSC::fromString< Dune::DynamicMatrix< double > >
                                        (std::string("[0.395  0 0 0;") +
                                         " -0.767672395484 0.395  0 0;" +
                                         " -0.851675323742  0.522967289188 0.395  0;" +
                                         " 0.288463109545 0.0880214273381 -0.337389840627 0.395]"));

    // this is unused by now, but you need to uncomment it whenever you use solve_rosenbrock and comment otherwise
    Dune::DynamicVector< double > d(Gamma.rows());
    for (size_t ii = 0; ii < Gamma.rows(); ++ii) {
      d[ii] = 0.0;
      for (size_t jj = 0; jj <= ii; ++jj)
        d[ii] += Gamma[ii][jj];
    }





    DSC_PROFILER.startTiming("fd.solve");

//    solve(psi,
//          t_end,
//          dt,
//          dx,
//          dmu,
//          saveInterval,
//          false,
//          true,
//          filename,
//          solution,
//          A,
//          b_1,
//          b_2,
//          c,
//          fv_space_1D,
//          TOL);


    solve_rosenbrock< DiscreteFunctionType2D, FVSpaceType1D, ProblemType >(psi,
                                                                           t_end,
                                                                           dt,
                                                                           dx,
                                                                           dmu,
                                                                           saveInterval,
                                                                           false,
                                                                           true,
                                                                           filename,
                                                                           solution,
                                                                           A,
                                                                           b_1,
                                                                           b_2,
                                                                           c,
                                                                           d,
                                                                           Gamma,
                                                                           fv_space_1d,
                                                                           TOL);

    DSC_PROFILER.stopTiming("fd.solve");

    std::cout << "took: " << DSC_PROFILER.getTiming("fd.solve")/1000.0 << " seconds." << std::endl;


    std::cout << " done" << std::endl;
    return 0;
  } catch (Dune::Exception& e) {
    std::cerr << "Dune reported: " << e.what() << std::endl;
    std::abort();
  }
} // ... main(...)

// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include "config.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>               // for input/output to shell
#include <fstream>                // for input/output to files

#include <dune/common/parallel/mpihelper.hh> // include mpi helper class

#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container/common.hh>

#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/playground/spaces/finitevolume/default.hh>
#include <dune/gdt/discretefunction/default.hh>
#include <dune/gdt/operators/projections.hh>

#include <dune/hdd/hyperbolic/problems/burgers.hh>
#include <dune/hdd/hyperbolic/problems/default.hh>

using namespace Dune::GDT;
using namespace Dune::HDD;

int main()
{
    static const int dimDomain = 1;
    static const int dimRange = 1;
    //choose GridType
    typedef Dune::YaspGrid< dimDomain >                     GridType;
    typedef typename GridType::Codim< 0 >::Entity           EntityType;

    //configure Problem
//    typedef Dune::HDD::Hyperbolic::Problems::Burgers< EntityType, double, dimDomain, double, dimRange > ProblemType;
    typedef Dune::HDD::Hyperbolic::Problems::Default< EntityType, double, dimDomain, double, dimRange > ProblemType;
    typedef typename ProblemType::FunctionType BoundaryValueFunctionType;
    typedef typename ProblemType::ConfigType ConfigType;
    ConfigType problem_config = ProblemType::default_config();
    //set boundary type ("periodic" or "dirichlet")
    ConfigType boundary_config;
    boundary_config["type"] = "dirichlet";
//    boundary_config["type"] = "periodic";
    problem_config.add(boundary_config, "boundary_info", true);
    //set boundary values (ignored if boundary is periodic)
    ConfigType boundary_value_config = ProblemType::DefaultFunctionType::default_config();
    boundary_value_config["type"] = ProblemType::DefaultFunctionType::static_id();
    boundary_value_config["variable"] = "x";
    boundary_value_config["expression"] = "x[0]";
    boundary_value_config["order"] = "1";
    problem_config.add(boundary_value_config, "boundary_values", true);

    //create Problem
    ProblemType problem = *(ProblemType::create(problem_config));

    //get grid configuration from problem
    typedef typename ProblemType::ConfigType ConfigType;
    const auto grid_config = problem.grid_config();

    //create grid
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config, "grid"));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    //get analytical flux and initial values
    typedef typename ProblemType::FluxType         AnalyticalFluxType;
    typedef typename ProblemType::FunctionType     FunctionType;
    typedef typename FunctionType::DomainFieldType DomainFieldType;
    typedef typename ProblemType::RangeFieldType   RangeFieldType;
    typedef typename Dune::Stuff::Functions::Indicator < EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 > IndicatorFunctionType;
    typedef typename IndicatorFunctionType::DomainType DomainType;
    const std::shared_ptr< const AnalyticalFluxType > analytical_flux = problem.flux();
    const std::shared_ptr< const FunctionType > initial_values = problem.initial_values();
//    const std::shared_ptr< const FunctionType > initial_values = std::make_shared< const IndicatorFunctionType >
//            (std::vector< std::tuple < DomainType, DomainType, RangeFieldType > > (1, std::make_tuple< DomainType, DomainType, RangeFieldType >(DomainType(0.5), DomainType(1), RangeFieldType(1))));

    // make a finite volume space on the leaf grid
    typedef typename GridType::LeafGridView                                     GridViewType;
    typedef Spaces::FiniteVolume::Default< GridViewType, RangeFieldType, 1 >    FVSpaceType;
    const FVSpaceType fv_space(grid->leafGridView());
    const auto& grid_view = fv_space.grid_view();

    // allocate a discrete function for the concentration and another one to temporary store the update in each step
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< RangeFieldType > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");
    FVFunctionType u_update(fv_space, "solution");

    Operators::apply_projection(*initial_values, u);
    u.visualize("concentration_0", false);

    // now do the time steps
    double t=0;
    const double dt=0.0005;
    int time_step_counter=0;
    const double saveInterval = 0.01;
    double saveStep = 0.01;
    int save_step_counter = 1;
    const double t_end = 5;

    //calculate dx and create lambda = dt/dx for the Lax-Friedrichs flux
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 > ConstantFunctionType;
    ConstantFunctionType lambda(dt/dx);

    //get numerical flux and local operator
    typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Inner< ConstantFunctionType > NumericalFluxType;
    typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Dirichlet< ConstantFunctionType, BoundaryValueFunctionType > NumericalBoundaryFluxType;
    typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType > LocalOperatorType;
    typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType > LocalBoundaryOperatorType;
    LocalOperatorType local_operator(*analytical_flux, lambda);
    const std::shared_ptr< const BoundaryValueFunctionType > boundary_values = problem.boundary_values();
    LocalBoundaryOperatorType local_boundary_operator(*analytical_flux, lambda, *boundary_values);

    while (t<t_end)
    {
      u_update.vector() *= RangeFieldType(0);
      // augment time step counter
      ++time_step_counter;

      // define IntersectionIterator
      typedef typename GridViewType::IntersectionIterator IntersectionIteratorType;

      //hack for periodic boundary
      int left_boundary_entity_offset = 0;
      int right_boundary_entity_offset = 0;
      auto it_end = fv_space.grid_view().template end< 0 >();
      int offset = 0;

      //matrices for the local operator
      Dune::DynamicMatrix< RangeFieldType > update(1, 1, RangeFieldType(0));
      Dune::DynamicMatrix< RangeFieldType > uselessmatrix(1, 1, RangeFieldType(0));
      std::vector< Dune::DynamicMatrix< RangeFieldType > > uselesstmplocalmatrix{};

      //walk the grid
      for (auto it = grid_view.template begin< 0 >(); it != it_end; ++it,  ++offset) {
        const auto& entity = *it;
        const auto u_i_n = u.local_discrete_function(entity);
        auto u_update_i_n = u_update.local_discrete_function(entity);

        IntersectionIteratorType i_it_end = grid_view.iend(entity);

        //walk intersections of the current entity
        for (IntersectionIteratorType i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
          const auto& intersection = *i_it;

          //handle inner intersections
          if (intersection.neighbor()) {
            const auto neighbor_ptr = intersection.outside();
            const auto& neighbor = *neighbor_ptr;
            const auto u_j_n = u.local_function(neighbor);
            update[0][0] = RangeFieldType(0);
            local_operator.apply(*u_i_n, *u_i_n, *u_j_n, *u_j_n, intersection, uselessmatrix, uselessmatrix, update, uselessmatrix, uselesstmplocalmatrix);
            u_update_i_n->vector().add(0, -1.0*dt*update[0][0]);
          }

          //hack for periodic boundary in 1D
          if (intersection.boundary()) {
            if (problem.boundary_info().get< std::string >("type") == "periodic") {
              if (Dune::FloatCmp::eq(intersection.geometry().center()[0], 1.0)) {
                right_boundary_entity_offset = offset;
              } else if (Dune::FloatCmp::eq(intersection.geometry().center()[0], 0.0)) {
                left_boundary_entity_offset = offset;
              } else DUNE_THROW(Dune::NotImplemented, "Strange boundary intersection");
            } else if (problem.boundary_info().get< std::string >("type") == "dirichlet") {
              update[0][0] = RangeFieldType(0);
              local_boundary_operator.apply(*u_i_n, *u_i_n, intersection, update, uselesstmplocalmatrix);
              u_update_i_n->vector().add(0, -1.0*dt*update[0][0]);
            } else DUNE_THROW(Dune::NotImplemented, "Only periodic or dirichlet boundary types are implemented!");
          }
        } // Intersection Walk
      } // Entity Grid Walk

      //handle boundary intersections (periodic boundary)
      if (problem.boundary_info().get< std::string >("type") == "periodic") {
        auto it_left = grid_view.template begin< 0 >();
        for (int ii = 0; ii < left_boundary_entity_offset; ++ii)
          ++it_left;
        const EntityType& left_boundary_entity = *it_left;
        //std::cout << "center left" << left_boundary_entity.geometry().center()[0] << std::endl;
        auto it_right = grid_view.template begin< 0 >();
        for (int ii = 0; ii < right_boundary_entity_offset; ++ii)
          ++it_right;
        const EntityType& right_boundary_entity = *it_right;
        //std::cout << "center right" << right_boundary_entity.geometry().center()[0] << std::endl;
        const auto u_left_n = u.local_discrete_function(left_boundary_entity);
        const auto u_right_n = u.local_discrete_function(right_boundary_entity);
        auto u_update_left_n = u_update.local_discrete_function(left_boundary_entity);
        auto u_update_right_n = u_update.local_discrete_function(right_boundary_entity);
        // left boundary entity
        IntersectionIteratorType i_it_end = grid_view.iend(left_boundary_entity);
        update[0][0] = RangeFieldType(0);
        for (IntersectionIteratorType i_it = grid_view.ibegin(left_boundary_entity); i_it != i_it_end; ++i_it) {
          const auto& intersection = *i_it;
          if (intersection.boundary()) {
            local_operator.apply(*u_left_n, *u_left_n, *u_right_n, *u_right_n, intersection, uselessmatrix, uselessmatrix, update, uselessmatrix, uselesstmplocalmatrix);
            u_update_left_n->vector().add(0, -1.0*dt*update[0][0]);
          }
        }
        // right boundary entity
        i_it_end = grid_view.iend(right_boundary_entity);
        update[0][0] = RangeFieldType(0);
        for (IntersectionIteratorType i_it = grid_view.ibegin(right_boundary_entity); i_it != i_it_end; ++i_it) {
          const auto& intersection = *i_it;
          if (intersection.boundary()) {
            local_operator.apply(*u_right_n, *u_right_n, *u_left_n, *u_left_n, intersection, uselessmatrix, uselessmatrix, update, uselessmatrix, uselesstmplocalmatrix);
            u_update_right_n->vector().add(0, -1.0*dt*update[0][0]);
          }
        }
      } // if ( ... == "periodic")

      //update u
      u.vector() += u_update.vector();

      // augment time
      t += dt;

      // check if data should be written
      if (t >= saveStep)
      {
        // write data
        u.visualize("concentration_" + DSC::toString(save_step_counter), false);

        // increase counter and saveStep for next interval
        saveStep += saveInterval;
        ++save_step_counter;
      }

      // print info about time, timestep size and counter
      std::cout << "s=" << grid->size(0)
                << " k=" << time_step_counter << " t=" << t << " dt=" << dt << std::endl;
    }    // while (t < t_end)

    std::cout << "Finished!!\n";

    return 0;
} // ... main(...)

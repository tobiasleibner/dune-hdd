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

#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/la/container/common.hh>

#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/localevaluation/hyperbolic.hh>

#include <dune/hdd/hyperbolic/problems/burgers.hh>

#include <dune/grid/common/mcmgmapper.hh> // mapper class
#include <dune/grid/io/file/vtk/vtkwriter.hh>

#include <dune/common/parallel/mpihelper.hh> // include mpi helper class


// __DUNE_GRID_HOWTO_VTKOUT_HH__
#include <stdio.h>
template< class GridType, class FunctionType >
void vtkout (const GridType& grid , const FunctionType& u, const char* name , int k, double time=0.0, int rank=0)
{
  Dune::VTKWriter < typename GridType::LeafGridView > vtkwriter(grid.leafGridView());
  char fname[128];
  char sername [128];
  sprintf(fname ,"%s-%05d",name ,k);
  sprintf(sername ,"%s.series",name);
  vtkwriter.addCellData(u,"celldata");
  vtkwriter.write( fname , Dune::VTK::ascii );
  if ( rank == 0) {
    std::ofstream serstream(sername , (k==0 ? std::ios_base::out : std::ios_base::app));
    serstream << k << " " << fname << ".vtu " << time << std::endl;
    serstream.close();
  }
}
// __DUNE_GRID_HOWTO_VTKOUT_HH__

// integrate_entity_HH
#include <dune/common/exceptions.hh>
#include <dune/geometry/quadraturerules.hh>

//! compute integral of function over entity with given order
template<class Entity, class Function>
double integrateEntity (const Entity &entity, const Function &f, int p)
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
  rule = Dune::QuadratureRules<ctype,dim>::rule(gt,p);

  // ensure that rule has at least the requested order
  if (rule.order()<p)
    DUNE_THROW(Dune::Exception,"order not available");

  // compute approximate integral
  double result=0;
  for (typename Dune::QuadratureRule<ctype,dim>::const_iterator i=rule.begin();
       i!=rule.end(); ++i)
  {
    double fval = f.evaluate(geometry.global(i->position()));
    double weight = i->weight();
    double detjac = geometry.integrationElement(i->position());
    result += fval * weight * detjac;
  }

  // return result
  return result;
}
// integrate_entity_HH


int main()
{
    //choose GridType
    typedef Dune::YaspGrid< 1 >                             GridType;
    typedef typename GridType::Codim< 0 >::Entity           EntityType;

    //get problem
    typedef Dune::HDD::Hyperbolic::Problems::Burgers< EntityType, double, 1, double, 1 > ProblemType;
    ProblemType problem{};

    //get grid configuration from problem
    typedef typename ProblemType::ConfigType ConfigType;
    const std::shared_ptr< const ConfigType > grid_config = problem.grid_config();

    //create grid
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(*grid_config, "grid"));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    //get AnalyticFlux and initial values
    typedef typename ProblemType::FluxType AnalyticFluxType;
    typedef typename ProblemType::FunctionType FunctionType;
    const std::shared_ptr< const AnalyticFluxType > analytic_flux = problem.flux();
    Dune::FieldVector<double, 1> ergebnis;
    (*analytic_flux).evaluate(0.55, ergebnis);
    std::cout << "analytic_flux " << ergebnis;
    const std::shared_ptr< const FunctionType > initial_values = problem.initial_values();

    // make a mapper for codim 0 entities in the leaf grid
    typedef typename Dune::LeafMultipleCodimMultipleGeomTypeMapper<GridType, Dune::MCMGElementLayout > MapperType;
    typedef typename MapperType::Index IndexType;
    MapperType mapper(*grid);

    // allocate a vector for the concentration and a vector to temporary store the update in each step
    typedef typename ProblemType::RangeFieldType RangeFieldType;
    Dune::Stuff::LA::CommonDenseVector< RangeFieldType > u(mapper.size());
    Dune::Stuff::LA::CommonDenseVector< RangeFieldType > u_update_vector(mapper.size());

    //get LeafGrid and Iterator for LeafGrid, set initial values
    typedef typename GridType::LeafGridView GridViewType;
    typedef typename GridViewType::template Codim< 0 >::Iterator IteratorType;
    GridViewType grid_view = grid->leafGridView();
    IteratorType it_end = grid_view.template end< 0 >();
    const int quadrature_order = 2;
    for (IteratorType it = grid_view.template begin< 0 >(); it != it_end; ++it) {
      const auto& entity = *it;
      const auto& local_function = initial_values->local_function(entity);
      double u_i = integrateEntity(entity, *local_function, quadrature_order);
      u[mapper.index(entity)] = u_i/entity.geometry().volume();
    }

    // write initial grid
    vtkout(*grid,u,"concentration",0,0.0);

    // now do the time steps
    double t=0;
    const double dt=0.05;
    int time_step_counter=0;
    const double saveInterval = 0.1;
    double saveStep = 0.1;
    int save_step_counter = 1;
    const double t_end = 1;

    //dx and lambda
    typedef typename FunctionType::DomainFieldType DomainFieldType;
    Dune::Stuff::Grid::Dimensions< GridViewType > dimensions(grid_view);
    const double delta_x = dimensions.entity_width.max();
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, 1 /*dimDomain*/, RangeFieldType, 1, 1 > ConstantFunctionType;
    std::cout << (dt/delta_x);
    const double delta_t_durch_delta_x = dt/delta_x;
    ConstantFunctionType lambda(delta_t_durch_delta_x);

    //get numerical flux and local operator
    typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichsFlux< ConstantFunctionType > NumericalFluxType;
    typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType > LocalOperatorType;
    LocalOperatorType local_operator(*analytic_flux, lambda);

    while (t<t_end)
    {
      u_update_vector *= RangeFieldType(0);
      // augment time step counter
      ++time_step_counter;

      // define IntersectionIterator
      typedef typename GridViewType::IntersectionIterator IntersectionIteratorType;

      //hack for periodic boundary
      int left_boundary_entity_offset = 0;
      int right_boundary_entity_offset = 0;
      it_end = grid_view.template end< 0 >();
      int offset = 0;

      //matrices for the local operator
      Dune::DynamicMatrix< RangeFieldType > update(1, 1, RangeFieldType(0));
      Dune::DynamicMatrix< RangeFieldType > uselessmatrix(1, 1, RangeFieldType(0));
      std::vector< Dune::DynamicMatrix< RangeFieldType > > uselesstmplocalmatrix{};

      //do the grid walk
      for (IteratorType it = grid_view.template begin< 0 >(); it != it_end; ++it,  ++offset) {
        const auto& entity = *it;
        ConstantFunctionType u_i_n_global(u[mapper.index(entity)]);
        const std::unique_ptr< Dune::Stuff::LocalfunctionInterface< EntityType, DomainFieldType, 1, RangeFieldType, 1, 1 > >& u_i_n = u_i_n_global.local_function(entity);

        IntersectionIteratorType i_it_end = grid_view.iend(entity);

        //walk intersections of the current entity
        for (IntersectionIteratorType i_it = grid_view.ibegin(entity); i_it != i_it_end; ++i_it) {
          const auto& intersection = *i_it;

          //handle inner intersections
          if (intersection.neighbor()) {
            const auto& neighbor = intersection.outside();
            ConstantFunctionType u_j_n_global(u[mapper.index(*neighbor)]);
            const auto u_j_n = u_j_n_global.local_function(*neighbor);
            update[0][0] = RangeFieldType(0);
                    //std::cout << "localfunc " << (*u_i_n).evaluate(entity.geometry().center()) << std::endl;
            local_operator.apply(*u_i_n, *u_i_n, *u_j_n, *u_j_n, intersection, uselessmatrix, uselessmatrix, update, uselessmatrix, uselesstmplocalmatrix);
            //std::cout << dt*update[0][0] << std::endl;
            u_update_vector[mapper.index(entity)] -= dt*update[0][0];
          }

          //hack for periodic boundary in 1D
          if (intersection.boundary()) {
            if (Dune::FloatCmp::eq(intersection.geometry().center()[0], 1.0)) {
              right_boundary_entity_offset = offset;
              //std::cout << "right offset" << right_boundary_entity_offset << std::endl;
            } else if (Dune::FloatCmp::eq(intersection.geometry().center()[0], 0.0)) {
              left_boundary_entity_offset = offset;
              //std::cout << "left offset" << left_boundary_entity_offset << std::endl;
            } else DUNE_THROW(Dune::NotImplemented, "Strange boundary intersection");
          }
        } // Intersection Walk
      } // Entity Grid Walk

      //handle boundary intersections (periodic boundary)
      IteratorType it_left = grid_view.template begin< 0 >();
      for (int ii = 0; ii < left_boundary_entity_offset; ++ii)
        ++it_left;
      const EntityType& left_boundary_entity = *it_left;
      std::cout << "center left" << left_boundary_entity.geometry().center()[0] << std::endl;
      IteratorType it_right = grid_view.template begin< 0 >();
      for (int ii = 0; ii < right_boundary_entity_offset; ++ii)
        ++it_right;
      const EntityType& right_boundary_entity = *it_right;
      std::cout << "center right" << right_boundary_entity.geometry().center()[0] << std::endl;
      ConstantFunctionType u_left_n_global(u[mapper.index(left_boundary_entity)]);
      ConstantFunctionType u_right_n_global(u[mapper.index(right_boundary_entity)]);
      const auto& u_left_n = u_left_n_global.local_function(left_boundary_entity);
      const auto& u_right_n = u_right_n_global.local_function(right_boundary_entity);
      // left boundary entity
      IntersectionIteratorType i_it_end = grid_view.iend(left_boundary_entity);
      update[0][0] = RangeFieldType(0);
      for (IntersectionIteratorType i_it = grid_view.ibegin(left_boundary_entity); i_it != i_it_end; ++i_it) {
        const auto& intersection = *i_it;
        if (intersection.boundary()) {
          local_operator.apply(*u_left_n, *u_left_n, *u_right_n, *u_right_n, intersection, uselessmatrix, uselessmatrix, update, uselessmatrix, uselesstmplocalmatrix);
          u_update_vector[mapper.index(left_boundary_entity)] -= dt*update[0][0];
        }
      }
      // right boundary entity
      i_it_end = grid_view.iend(right_boundary_entity);
      update[0][0] = RangeFieldType(0);
      for (IntersectionIteratorType i_it = grid_view.ibegin(right_boundary_entity); i_it != i_it_end; ++i_it) {
        const auto& intersection = *i_it;
        if (intersection.boundary()) {
          local_operator.apply(*u_right_n, *u_right_n, *u_left_n, *u_left_n, intersection, uselessmatrix, uselessmatrix, update, uselessmatrix, uselesstmplocalmatrix);
          u_update_vector[mapper.index(right_boundary_entity)] -= dt*update[0][0];
        }
      }

      //update u
      u += u_update_vector;

      // augment time
      t += dt;

      // check if data should be written
      if (t >= saveStep)
      {
        // write data
        vtkout(*grid,u,"concentration",save_step_counter,t);

        // increase counter and saveStep for next interval
        saveStep += saveInterval;
        ++save_step_counter;
      }

      // print info about time, timestep size and counter
      std::cout << "s=" << grid->size(0)
                << " k=" << time_step_counter << " t=" << t << " dt=" << dt << std::endl;
    }    // while loop

    std::cout << "Finished!!\n";

    return 0;
} // ... main(...)

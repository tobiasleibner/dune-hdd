// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Schindler
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include "config.h"

#include <string>
#include <vector>
#include <memory>
#include <iostream>               // for input/output to shell

#include <dune/common/parallel/mpihelper.hh> // include mpi helper class

#include <dune/stuff/grid/provider/cube.hh>
#include <dune/stuff/grid/information.hh>
#include <dune/stuff/grid/periodicview.hh>
#include <dune/stuff/la/container/common.hh>

#include <dune/gdt/assembler/local/codim1.hh>
#include <dune/gdt/assembler/system.hh>
#include <dune/gdt/localoperator/codim1.hh>
#include <dune/gdt/localevaluation/laxfriedrichs.hh>
#include <dune/gdt/spaces/fv/default.hh>
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

    //create grid
    typedef Dune::Stuff::Grid::Providers::Cube< GridType >  GridProviderType;
    GridProviderType grid_provider = *(GridProviderType::create(grid_config, "grid"));
    const std::shared_ptr< const GridType > grid = grid_provider.grid_ptr();

    // make a finite volume space on the leaf grid
    typedef typename GridType::LeafGridView                                     GridViewType;
    typedef typename Dune::Stuff::Grid::PeriodicGridView< GridViewType >        PeriodicGridViewType;
    typedef Spaces::FV::Default< PeriodicGridViewType, RangeFieldType, 1 >      FVSpaceType;
    const GridViewType grid_view = grid->leafGridView();
    const PeriodicGridViewType periodic_grid_view(grid_view);
    const FVSpaceType fv_space(periodic_grid_view);
    //const auto& grid_view = fv_space.grid_view();

    // allocate a discrete function for the concentration and another one to temporary store the update in each step
    typedef DiscreteFunction< FVSpaceType, Dune::Stuff::LA::CommonDenseVector< RangeFieldType > > FVFunctionType;
    FVFunctionType u(fv_space, "solution");
    FVFunctionType u_update(fv_space, "solution");
    std::cout << "u vector " << u.vector().size() << std:: endl;

    //visualize initial values
    Operators::apply_projection(*initial_values, u);
    u.visualize("concentration_0", false);

    // now do the time steps
    double t=0;
    const double dt=0.005;
    int time_step_counter=0;
    const double saveInterval = 0.01;
    double saveStep = 0.01;
    int save_step_counter = 1;
    const double t_end = 5;

    //calculate dx and create lambda = dt/dx for the Lax-Friedrichs flux
    Dune::Stuff::Grid::Dimensions< PeriodicGridViewType > dimensions(fv_space.grid_view());
    const double dx = dimensions.entity_width.max();
    typedef typename Dune::Stuff::Functions::Constant< EntityType, DomainFieldType, dimDomain, RangeFieldType, 1, 1 > ConstantFunctionType;
    ConstantFunctionType lambda(dt/dx);

    //get numerical flux and local operator
    typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Inner< ConstantFunctionType > NumericalFluxType;
    typedef typename Dune::GDT::LocalEvaluation::LaxFriedrichs::Dirichlet< ConstantFunctionType, BoundaryValueFunctionType > NumericalBoundaryFluxType;
    typedef typename Dune::GDT::LocalOperator::Codim1FV< NumericalFluxType > LocalOperatorType;
    typedef typename Dune::GDT::LocalOperator::Codim1FVBoundary< NumericalBoundaryFluxType > LocalBoundaryOperatorType;
    const LocalOperatorType local_operator(*analytical_flux, lambda);
    const std::shared_ptr< const BoundaryValueFunctionType > boundary_values = problem.boundary_values();
    const LocalBoundaryOperatorType local_boundary_operator(*analytical_flux, lambda, *boundary_values);


    //get system assembler
    typedef SystemAssembler< FVSpaceType > SystemAssemblerType;
    SystemAssemblerType systemAssembler(fv_space);
    const LocalAssembler::Codim1CouplingFV< LocalOperatorType > inner_assembler(local_operator);
    const LocalAssembler::Codim1BoundaryFV< LocalBoundaryOperatorType > boundary_assembler(local_boundary_operator);

    //time loop
    while (t<t_end)
    {
      //clear update vector
      u_update.vector() *= RangeFieldType(0);

      //add local assemblers
      systemAssembler.add(inner_assembler, u, u_update);
      systemAssembler.add(boundary_assembler, u, u_update);

      //walk the grid
      systemAssembler.assemble();

      //update u
      u.vector() += u_update.vector()*(-1.0*dt);

      // augment time step counter
      ++time_step_counter;

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
    } // while (t < t_end)

    std::cout << "Finished!!\n";

    return 0;
} // ... main(...)

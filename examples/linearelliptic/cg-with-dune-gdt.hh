// This file is part of the dune-hdd project:
//   http://users.dune-project.org/projects/dune-hdd
// Copyright holders: Felix Albrecht
// License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#include <string>
#include <vector>

#include <dune/stuff/common/string.hh>

#include <dune/pymor/common/exceptions.hh>
#include <dune/pymor/parameters/base.hh>
#include <dune/pymor/la/container/eigen.hh>

#include <dune/hdd/linearelliptic/discretization/cg_gdt.hh>
#include "problem.hh"


class LinearellipticExampleCG
{
  typedef Problem<>                               ProblemType;
  typedef ProblemType::SettingsType               SettingsType;
  typedef ProblemType::GridPartType               GridPartType;
  typedef ProblemType::RangeFieldType             RangeFieldType;
  static const int DUNE_UNUSED(                   dimDomain) = ProblemType::dimDomain;
  static const int DUNE_UNUSED(                   dimRange) = ProblemType::dimRange;
  typedef ProblemType::ModelType                  ModelType;
  typedef Dune::HDD::LinearElliptic::ContinuousGalerkinDiscretizationGDT< GridPartType,
                                                                          RangeFieldType,
                                                                          dimRange,
                                                                          1,
                                                                          true >
                                                  DiscretizationType;

public:
  static const std::string static_id();

  static void writeSettingsFile(const std::string filename);

  LinearellipticExampleCG(const std::vector< std::string > arguments = std::vector< std::string >())
    throw (Dune::Pymor::Exception::this_does_not_make_any_sense);

  bool parametric() const;

  Dune::Pymor::ParameterType parameter_type() const;

  Dune::Pymor::LA::EigenDenseVector* solve(const Dune::Pymor::Parameter mu = Dune::Pymor::Parameter()) const
    throw (Dune::Pymor::Exception::wrong_parameter_type);

//  void visualize(const Dune::Pymor::LA::EigenDenseVector* vector,
//                 const std::string filename,
//                 const std::string name) const;

private:
  ProblemType problem_;
  std::shared_ptr< DiscretizationType > discretization_;
}; // class LinearellipticExampleCG




//int run(int argc, char** argv)
//{
//  try {
//    // init problem
//    ProblemType problem(argc, argv);
//    const bool debugLogging = problem.debugLogging();
//    Stuff::Common::LogStream& info  = Stuff::Common::Logger().info();
//    Stuff::Common::LogStream& debug = Stuff::Common::Logger().debug();
//    Dune::Timer timer;
//    const SettingsType& settings = problem.settings();
//    const std::string filename = problem.filename();
//    // check
//    if (problem.model()->parametric() && !problem.model()->affineparametric())
//      DUNE_THROW(Dune::NotImplemented,
//                 "\n" << Dune::Stuff::Common::colorStringRed("ERROR:")
//                 << " only implemented for nonparametric or affineparametric models!");
//    // grid part
//    const std::shared_ptr< const GridPartType > gridPart(new GridPartType(*(problem.grid())));

//    info << "initializing solver";
//    if (!debugLogging)
//      info << "... " << std::flush;
//    else
//      info << ":" << std::endl;
//    timer.reset();
//    typedef Elliptic::SolverContinuousGalerkinGDT< GridPartType, RangeFieldType, dimRange, 1 > SolverType;
//    SolverType solver(gridPart,
//                      problem.boundaryInfo(),
//                      problem.model());
//    solver.init(debug, "  ");
//    if (!debugLogging)
//      info << "done (took " << timer.elapsed() << " sec)" << std::endl;

//    typedef typename SolverType::VectorType VectorType;
//    if (!problem.model()->parametric()) {
//      info << "solving";
//      if (!debugLogging)
//        info << "... " << std::flush;
//      else
//        info << ":" << std::endl;
//      timer.reset();
//      std::shared_ptr< VectorType > solutionVector = solver.createVector();
//      solver.solve(solutionVector,
//                   settings.sub("linearsolver"),
//                   debug,
//                   "  ");
//      if (!debugLogging)
//        info << "done (took " << timer.elapsed() << " sec)" << std::endl;

//      info << "writing solution to disc";
//      if (!debugLogging)
//        info << "... " << std::flush;
//      else
//        info << ":" << std::endl;
//      timer.reset();
//      solver.visualize(solutionVector,
//                       filename + ".solution",
//                       id() + ".solution",
//                       debug,
//                       "  ");
//      if (!debugLogging)
//        info << "done (took " << timer.elapsed() << " sec)" << std::endl;
//    } else { // if (!model->parametric())
//      typedef typename ModelType::ParamFieldType  ParamFieldType;
//      typedef typename ModelType::ParamType       ParamType;
//      const size_t paramSize = problem.model()->paramSize();
//      const SettingsType& parameterSettings = settings.sub("parameter");
//      const size_t numTestParams = parameterSettings.get< size_t >("test.size");
//      // loop over all test parameters
//      for (size_t ii = 0; ii < numTestParams; ++ii) {
//        const std::string iiString = Dune::Stuff::Common::toString(ii);
//        const ParamType testParameter
//            = parameterSettings.getDynVector< ParamFieldType >("test." + iiString, paramSize);
//        // after this, testParameter is at least as long as paramSize, but it might be too long
//        const ParamType mu = Dune::Stuff::Common::resize(testParameter, paramSize);
//        info << "solving for parameter [" << mu << "]";
//        if (!debugLogging)
//          info << "... " << std::flush;
//        else
//          info << ":" << std::endl;
//        timer.reset();
//        std::shared_ptr< VectorType > solutionVector = solver.createVector();
//        solver.solve(solutionVector,
//                     mu,
//                     settings.sub("linearsolver"),
//                     debug,
//                     "  ");
//        if (!debugLogging)
//          info << "done (took " << timer.elapsed() << " sec)" << std::endl;

//        info << "writing solution for parameter [" << mu << "] to disc";
//        if (!debugLogging)
//          info << "... " << std::flush;
//        else
//          info << ":" << std::endl;
//        timer.reset();
//        std::stringstream name;
//        name << id() << ".solution." << iiString << " (parameter [" << mu << "])";
//        solver.visualize(solutionVector,
//                         filename + ".solution." + iiString,
//                         name.str(),
//                         debug,
//                         "  ");
//        if (!debugLogging)
//          info << "done (took " << timer.elapsed() << " sec)" << std::endl;
//      } // loop over all test parameters
//    } // if (!model->parametric())

//  } catch(Dune::Exception& e) {
//    std::cerr << "Dune reported error: " << e.what() << std::endl;
//  } catch(std::exception& e) {
//    std::cerr << e.what() << std::endl;
//  } catch( ... ) {
//    std::cerr << "Unknown exception thrown!" << std::endl;
//  } // try

//  // if we came that far we can as well be happy about it
//  return 0;
//} // run
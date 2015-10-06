# This file is part of the dune-hdd project:
#   https://github.com/pymor/dune-hdd
# Copyright Holders: Felix Schindler
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

add_library(dunehdd-hyperbolic-burgers
            STATIC
            "burgers.cc")
add_library(dunehdd-hyperbolic-twobeams
            STATIC
            "twobeams.cc")

ADD_EXECUTABLE(example_hyperbolic_burgers "burgers.cc" ${COMMON_HEADER})
ADD_EXECUTABLE(example_hyperbolic_twobeams "twobeams.cc" ${COMMON_HEADER})

TARGET_LINK_LIBRARIES(example_hyperbolic_burgers
                      ${COMMON_LIBS} ${Boost_FILESYSTEM_LIBRARY})
TARGET_LINK_LIBRARIES(example_hyperbolic_twobeams
                      ${COMMON_LIBS} ${Boost_FILESYSTEM_LIBRARY})

foreach(target
example_hyperbolic_burgers
example_hyperbolic_twobeams
        )
  add_dune_parmetis_flags(${target})
  add_dune_superlu_flags(${target})
  add_dune_umfpack_flags(${target})
  add_dune_tbb_flags(${target})
  add_dune_alugrid_flags(${target})
  add_dune_mpi_flags(${target})
  target_link_libraries(${target} parmetis )
  add_dune_alugrid_flags(${target})
endforeach(target)

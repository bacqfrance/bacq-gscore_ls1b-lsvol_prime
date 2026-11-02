cmake_minimum_required(VERSION 3.26)
project(BACQBench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # Mandatory for .so / .dylib

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/libs" CACHE PATH "Default install prefix" FORCE)
endif()

# ==================== Options ====================
option(WITH_OPENMP "Activate support for OpenMP" ON)

# ==================== OpenMP ====================
if(WITH_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found and activated : ${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP enabled but not found → desactivated")
        set(WITH_OPENMP OFF CACHE BOOL "OpenMP" FORCE)
    endif()
endif()

# ==================== Shared libraries ====================
add_library(BacqBench SHARED
    hardware/src/simpleSimulatedAnnealing.cpp
    # add other cpp source files here
    hardware/src/factorizeOnCPU.cpp
)

# 1. Definition for preprocessing macro (-D__C_TYPES_LIB_)
target_compile_definitions(BacqBench PRIVATE
    __C_TYPES_LIB_
)

# 2. Optimisation options for compilers (GCC / Clang)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(BacqBench PRIVATE
        -O3
        -march=native
        -ffast-math
        ${OpenMP_CXX_FLAGS} # brute force the use of -fopenmp on linux
    )
endif()

# Conditional link of OpenMP 
if(WITH_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(BacqBench PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(BacqBench PRIVATE USE_OPENMP)
endif()

# Optimization Flags depending on OS
if(MSVC)
    target_compile_options(BacqBench PRIVATE /O2 /W4)
else()
    target_compile_options(BacqBench PRIVATE -O3 -march=native -Wall -Wextra)
endif()

# Set target names
set_target_properties(BacqBench PROPERTIES
    OUTPUT_NAME "BacqBench"
    DEBUG_POSTFIX "d"                     # libBacqBench_d.so for debug
    VISIBILITY_INLINES_HIDDEN ON
)

# Installation
install(TARGETS BacqBench
    LIBRARY  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/libs" COMPONENT Runtime
    RUNTIME  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/libs" COMPONENT Runtime   # Windows .dll
    ARCHIVE  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/libs"  COMPONENT Development
)

#install(DIRECTORY include/ DESTINATION include)

### =================================================
### First unitary test and integration test (Binary)
### =================================================
add_executable(sa_test_bin
    hardware/src/simpleSimulatedAnnealing.cpp
)

# Applies flag -D__UNITARY_TEST_BINARY_ only to sa_test_bin
target_compile_definitions(sa_test_bin PRIVATE
    __UNITARY_TEST_BINARY_
)

# Generate binary program
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sa_test_bin PRIVATE -O3 -march=native -ffast-math)
endif()

target_link_libraries(sa_test_bin PRIVATE OpenMP::OpenMP_CXX)

### =================================================
### 2. unitary test and integration test (Spin)
### =================================================
add_executable(sa_test_spin
    hardware/src/simpleSimulatedAnnealing.cpp
)

# Applies flag -D__UNITARY_TEST_BINARY_ only to sa_test_spin
target_compile_definitions(sa_test_spin PRIVATE
    __UNITARY_TEST_SPIN_
)

### =================================================
### 2.b unitary test factorization on CPU
### =================================================
add_executable(factorize_test
    hardware/src/factorizeOnCPU.cpp
)

# Applies flag -D__UNITARY_TEST_BINARY_ only to sa_test_spin
target_compile_definitions(factorize_test PRIVATE
    __UNITARY_TEST_
)


# Generate binary program
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sa_test_spin PRIVATE -O3 -march=native -ffast-math)
endif()

target_link_libraries(sa_test_spin PRIVATE OpenMP::OpenMP_CXX)

# ==============================================================================
# 3. Install unitary test programs in bin/
# ==============================================================================
install(TARGETS sa_test_bin sa_test_spin factorize_test
    RUNTIME DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

# ==============================================================================
# 4. Add 'make check' to run the tests
# ==============================================================================
enable_testing()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# Record checks in CTest engine: first cpp tests (C++)
add_test(NAME IntegrationTestBin COMMAND sa_test_bin)
add_test(NAME IntegrationTestSpin COMMAND sa_test_spin)
add_test(NAME IntegrationTestPrime COMMAND factorize_test)

# --- Python tests ---
add_test(
    NAME PythonGnBenchLibTest
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/gscore/GnBenchLib.py" 1
)
add_test(
    NAME PythonLinBenchLibTest
    COMMAND Python3::Interpreter "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/linscore/LinBench2.py" 
)

# Force the root for sources in the PYTHONPATH for make check smooth execution
set_tests_properties(PythonGnBenchLibTest PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}"
)

# Create "make check" to launch ctest
add_custom_target(check
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS sa_test_bin sa_test_spin
    COMMENT "Execute unitary tests..."
)

# ==============================================================================
# 5. Generate documentation (make doc)
# ==============================================================================
find_package(Doxygen REQUIRED)
find_package(LATEX REQUIRED COMPONENTS PDFLATEX)
find_program(PANDOC_EXECUTABLE pandoc)

if(DOXYGEN_FOUND AND LATEX_PDFLATEX_FOUND AND PANDOC_EXECUTABLE)
    # Doxyfile source file and generated file in directory build/
    set(DOXYFILE_IN "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
    set(DOXYFILE_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
    set(README_MD "${CMAKE_SOURCE_DIR}/README.md")
    set(README_TEX "${CMAKE_CURRENT_SOURCE_DIR}/doc/readme_section.tex")
    # Rule to generate LaTeX section from README.md
    add_custom_command(
        OUTPUT ${README_TEX}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/docs"
        COMMAND ${PANDOC_EXECUTABLE} ${README_MD} -o ${README_TEX} --listings
        DEPENDS ${README_MD}
        COMMENT "Convert README.md into LaTeX section with Pandoc"
    )
    
    # If no Doxyfile.in provided, let cmake generate it
    # use doxyfile_default for commands
    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    # Target 'make doc' that launches Doxygen then compile the PDF report with pdflatex
    add_custom_target(doc
        # 1. Generate doc (HTML + LaTeX source)
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        # 2. Compile Makefile generated by Doxygen in sub-directory latex/
        COMMAND ${CMAKE_COMMAND} -E chdir "${CMAKE_CURRENT_BINARY_DIR}/doc/latex" make
        # 3. Optionnal : Copy final PDF in doc/ repertory of the project
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_SOURCE_DIR}/doc"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/doc/latex/refman.pdf" "${CMAKE_CURRENT_SOURCE_DIR}/doc/documentation.pdf"
        DEPENDS ${README_TEX}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generate Doxygen documentation and compile PDF documentation (pdflatex)..."
        VERBATIM
    )
else()
    message(WARNING "Doxygen or LaTeX or pandoc not found. Target 'make doc' not available.")
endif()

cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

# Macro to convert a CMake list variable to a SLEPc configure list
macro(OC_TO_SLEPc_LIST CMAKE_LIST SLEPc_LIST)
  set(${SLEPc_LIST} "[")
  list(LENGTH ${CMAKE_LIST} _CMAKE_LIST_LENGTH)
  if(${_CMAKE_LIST_LENGTH} GREATER 0)
    list(GET ${CMAKE_LIST} 0 _ITEM)
    string(APPEND ${SLEPc_LIST} "${_ITEM}")
    if(${_CMAKE_LIST_LENGTH} GREATER 1)
      math(EXPR _STOP_IDX "${_CMAKE_LIST_LENGTH}-1")
      foreach(_ITEM_IDX RANGE 1 ${_STOP_IDX})
        list(GET ${CMAKE_LIST} ${_ITEM_IDX} _ITEM)
        string(APPEND ${SLEPc_LIST} ",${_ITEM}")
      endforeach()
      unset(_STOP_IDX)
    endif()
    unset(_ITEM)
  endif()
  string(APPEND ${SLEPc_LIST} "]")
  unset(_CMAKE_LIST_LENGTH)
  message(STATUS "CMAKE_LIST = '${${CMAKE_LIST}}'")
  message(STATUS "SLEPc_LIST = '${${SLEPc_LIST}}'")
endmacro()

file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/slepcversion.h _SLEPc_VERSION_H)
string(REGEX REPLACE ".*#define[ \t]+SLEPC_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" SLEPc_VERSION_MAJOR ${_SLEPc_VERSION_H})
string(REGEX REPLACE ".*#define[ \t]+SLEPC_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" SLEPc_VERSION_MINOR ${_SLEPc_VERSION_H})
string(REGEX REPLACE ".*#define[ \t]+SLEPC_VERSION_SUBMINOR[ \t]+([0-9]+).*" "\\1" SLEPc_VERSION_SUBMINOR ${_SLEPc_VERSION_H})
string(REGEX REPLACE ".*#define[ \t]+SLEPC_VERSION_RELEASE[ \t]+([0-9]+).*" "\\1" SLEPc_VERSION_RELEASE ${_SLEPc_VERSION_H})

set(SLEPc_VERSION "${SLEPc_VERSION_MAJOR}.${SLEPc_VERSION_MINOR}.${SLEPc_VERSION_SUBMINOR}")

message(STATUS "SLEPc_VERSION '${SLEPc_VERSION}'")

project(SLEPc
  LANGUAGES C CXX Fortran
  VERSION ${SLEPc_VERSION}
)

set(SLEPc_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

get_property(OC_HAVE_MULTICONFIG_ENV GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

list(APPEND SLEPc_CONFIG "--prefix=${SLEPc_PREFIX}")

message(STATUS "SLEPc configure options: '${SLEPc_CONFIG}'")

add_custom_target(SLEPc
  ALL
  COMMAND ${CMAKE_COMMAND} -E env PETSC_DIR=${SLEPc_PREFIX} ${SLEPc_DIR}/configure ${SLEPc_CONFIG}
  COMMAND ${CMAKE_COMMAND} -E env SLEPC_DIR=${SLEPc_DIR} PETSC_DIR=${SLEPc_PREFIX} make
  COMMAND ${CMAKE_COMMAND} -E env SLEPC_DIR=${SLEPc_DIR} PETSC_DIR=${SLEPc_PREFIX} make install
  COMMAND_EXPAND_LISTS
  WORKING_DIRECTORY ${SLEPc_DIR}
  COMMENT "Configuring, building and installing SLEPc...."
)

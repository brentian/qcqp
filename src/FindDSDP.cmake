#set(DSDP_HOME $ENV{DSDP_HOME} CACHE PATH "DSDP root directory.")
#message("Looking for DSDP in ${DSDP_HOME}")
#if (APPLE)
#    set(DSDP_INCLUDE_DIR "${DSDP_HOME}/include")
#    find_library(DSDP_LIBRARY libdsdp.lib HINTS "${DSDP_HOME}/lib/stable")
#    find_library(DSDP_BETA_LIBRARY libdsdp.lib HINTS "${DSDP_HOME}/lib/latest")
#elseif (UNIX)
#    set(DSDP_INCLUDE_DIR "${DSDP_HOME}/include")
#    find_library(DSDP_LIBRARY libdsdp.lib HINTS "${DSDP_HOME}/lib/stable")
#    find_library(DSDP_BETA_LIBRARY libdsdp.lib HINTS "${DSDP_HOME}/lib/latest")
#endif ()

set(DSDP_HOME "$ENV{DSDP_HOME}")
message("Looking for DSDP in ${DSDP_HOME}")
set(DSDP_INCLUDE_DIR "${DSDP_HOME}/include")

find_library(DSDP_LIBRARY
        NAMES dsdp
        HINTS "${DSDP_HOME}/lib/")

find_library(DSDP_LIBRARY_mx
        NAMES mx
        HINTS "${DSDP_HOME}/thirdparty/prebuild/osx/")

find_library(DSDP_LIBRARY_mat
        NAMES mat
        HINTS "${DSDP_HOME}/thirdparty/prebuild/osx/")

find_library(DSDP_LIBRARY_mex
        NAMES mex
        HINTS "${DSDP_HOME}/thirdparty/prebuild/osx/")

# MKLs
# clang -Wl,-rpath ${MKL_HOME}/lib  -I ../include/ -L../build/src -L../thirdparty/prebuild/osx  -ldsdp -lmx -lmex -lmat -lm
# $MKL_HOME/lib/libmkl_intel_lp64.a $MKL_HOME/lib/libmkl_sequential.a $MKL_HOME/lib/libmkl_core.a
find_library(
        DSDP_LIBRARY_MKL1
        NAMES mkl_intel_lp64
        HINTS "$ENV{MKL_HOME}/lib/"
)
find_library(
        DSDP_LIBRARY_MKL2
        NAMES mkl_sequential
        HINTS "$ENV{MKL_HOME}/lib/"
)
find_library(
        DSDP_LIBRARY_MKL3
        NAMES mkl_core
        HINTS "$ENV{MKL_HOME}/lib/"
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DSDP DEFAULT_MSG DSDP_LIBRARY DSDP_INCLUDE_DIR)

if (DSDP_FOUND)
    set(DSDP_INCLUDE_DIRS ${DSDP_INCLUDE_DIR})
    set(DSDP_LIBRARIES
            ${DSDP_LIBRARY}
            ${DSDP_LIBRARY_mex}
            ${DSDP_LIBRARY_mat}
            ${DSDP_LIBRARY_mx}
            ${DSDP_LIBRARY_MKL1}
            ${DSDP_LIBRARY_MKL2}
            ${DSDP_LIBRARY_MKL3}
            )
    message("—- Set DSDP lib  ${DSDP_LIBRARIES}")
    message("—- Set DSDP includes  ${DSDP_INCLUDE_DIRS}")
endif (DSDP_FOUND)

mark_as_advanced(DSDP_LIBRARY DSDP_INCLUDE_DIR)


set(COPT_HOME "$ENV{COPT_HOME}")
message("Looking for COPT in ${COPT_HOME}")
set(COPT_INCLUDE_DIR "${COPT_HOME}/include")

find_library(COPT_LIBRARY
        NAMES COPT
        HINTS "${COPT_HOME}/lib/")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(COPT DEFAULT_MSG COPT_LIBRARY COPT_INCLUDE_DIR)

if (COPT_FOUND)
    set(COPT_INCLUDE_DIRS ${COPT_INCLUDE_DIR})
    set(COPT_LIBRARIES ${COPT_LIBRARY})
    message("—- Set COPT lib  ${COPT_LIBRARIES}")
    message("—- Set COPT includes  ${COPT_INCLUDE_DIRS}")
endif (COPT_FOUND)

mark_as_advanced(COPT_LIBRARY COPT_INCLUDE_DIR)
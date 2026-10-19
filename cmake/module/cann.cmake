message(STATUS "--------------cann--------------")


if (NOT DEFINED CANN_ROOT)
  message(FATAL_ERROR "Cannot get definition for 'CANN_ROOT'")
endif ()


if (NOT EXISTS ${CANN_ROOT})
  message(FATAL_ERROR "Cannot found directory ${CANN_ROOT}")
endif ()
if (NOT EXISTS ${CANN_ROOT}/include)
  message(FATAL_ERROR "Cannot found directory ${CANN_ROOT}/include")
endif ()

add_library(cann INTERFACE)

include_directories(${CANN_ROOT}/include)
target_include_directories(cann INTERFACE ${CANN_ROOT}/include)


# CANN up to 8 keeps the link stubs in runtime/lib64/stub; CANN 9 ships the
# libraries in lib64 of the architecture directory, linked as <root>/lib64.
set(cann_lib_suffixes runtime/lib64/stub lib64)

list(APPEND CANN_LIBS_NAME ascendcl)
list(APPEND CANN_LIBS_NAME acl_dvpp_mpi)

foreach (cann_name ${CANN_LIBS_NAME})
  find_library(${cann_name}_lib NAMES ${cann_name} HINTS ${CANN_ROOT} PATH_SUFFIXES ${cann_lib_suffixes})
  if (NOT ${cann_name}_lib)
    set(CANN_FOUND OFF)
    message(FATAL_ERROR "Cannot found lib${cann_name}.so under ${CANN_ROOT} needed by CANN")
  endif ()

  add_library(cann::${cann_name} SHARED IMPORTED)
  set_target_properties(cann::${cann_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${CANN_ROOT}/include
      IMPORTED_LOCATION ${${cann_name}_lib}
  )
endforeach ()

set(CANN_FOUND ON)

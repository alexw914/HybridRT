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


if (NOT EXISTS ${CANN_ROOT}/runtime/lib64/stub)
  message(FATAL_ERROR "Cannot found directory ${CANN_ROOT}/runtime/lib64/stub")
endif ()
set(cann_lib_dir ${CANN_ROOT}/runtime/lib64/stub)


list(APPEND CANN_LIBS_NAME ascendcl)
list(APPEND CANN_LIBS_NAME acl_dvpp_mpi)

foreach (cann_name ${CANN_LIBS_NAME})
  find_library(${cann_name}_lib NAMES ${cann_name} HINTS ${CANN_ROOT} PATH_SUFFIXES runtime/lib64/stub)
  if (NOT ${cann_name}_lib)
    set(CANN_FOUND OFF)
    message(FATAL_ERROR "Cannot found ${${cann_name}_lib} needed by CANN")
  endif ()

  add_library(cann::${cann_name} SHARED IMPORTED)
  set_target_properties(cann::${cann_name} PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${CANN_ROOT}/include
      IMPORTED_LOCATION ${${cann_name}_lib}
  )
endforeach ()


#set(CMAKE_FIND_ROOT_PATH ${CMAKE_FIND_ROOT_PATH} ${cann_lib_dir})
#target_link_directories(cann INTERFACE ${cann_lib_dir})
#
#target_link_libraries(cann INTERFACE
#    ${cann_lib_dir}/libascendcl.so
#    ${cann_lib_dir}/libacl_dvpp_mpi.so
#)

#add_library(cann::cann ALIAS cann)
set(CANN_FOUND ON)

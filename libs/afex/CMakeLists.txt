if(NOT IDF_TARGET)
  option(AFEX_EMBEDDED "Build the minimal embedded afex core." OFF)
endif()

add_subdirectory(core)

if(NOT IDF_TARGET AND NOT AFEX_EMBEDDED)
  add_subdirectory(host)
endif()

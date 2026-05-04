add_subdirectory(core)

if(NOT IDF_TARGET)
  add_subdirectory(host)
endif()

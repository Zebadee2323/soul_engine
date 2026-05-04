find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

add_library(pyafex_native MODULE
    src/pyafex_module.cpp
)

target_compile_features(pyafex_native PRIVATE cxx_std_20)
target_link_libraries(pyafex_native PRIVATE afex Python3::Module)

set(pyafex_package_dir "${CMAKE_BINARY_DIR}/python/lib/pyafex/pyafex")

set_target_properties(pyafex_native PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_pyafex"
    LIBRARY_OUTPUT_DIRECTORY "${pyafex_package_dir}"
)

if(DEFINED Python3_SOABI)
    set_target_properties(pyafex_native PROPERTIES SUFFIX ".${Python3_SOABI}.so")
endif()

add_custom_command(TARGET pyafex_native POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/pyafex/__init__.py"
        "${pyafex_package_dir}/__init__.py"
)

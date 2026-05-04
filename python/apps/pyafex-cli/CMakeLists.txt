find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(pyafex_cli "${CMAKE_CURRENT_BINARY_DIR}/pyafex-cli")
set(pyafex_pythonpath "${CMAKE_BINARY_DIR}/python/lib/pyafex")

file(GENERATE
    OUTPUT "${pyafex_cli}"
    CONTENT "#!/usr/bin/env bash\nPYTHONPATH=${pyafex_pythonpath}:$PYTHONPATH exec ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/main.py \"$@\"\n"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

add_custom_target(pyafex_cli ALL
    DEPENDS pyafex_native "${pyafex_cli}"
)

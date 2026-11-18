cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED CMAKE_BUILD_TYPE AND NOT DEFINED ENV{CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type")
endif ()

project(runtimeviewer VERSION 0.1.0 LANGUAGES CXX)

# The Dusklight version to build against: a release tag (e.g. "v2.0.0") or a commit SHA.
if (NOT DUSKLIGHT_VERSION)
    set(DUSKLIGHT_VERSION "v2.0.2")
endif ()
# Fetches the Dusklight sources into ./dusklight and provides DUSKLIGHT_DIR.
# Use -DDUSKLIGHT_DIR= to point at an existing Dusklight checkout.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FetchDusklight.cmake")
add_subdirectory("${DUSKLIGHT_DIR}/sdk" dusklight-sdk EXCLUDE_FROM_ALL)

configure_file(mod.json.in "${CMAKE_CURRENT_BINARY_DIR}/mod.json" @ONLY NEWLINE_STYLE LF)

add_library(mi_game_abi INTERFACE)
target_link_libraries(mi_game_abi INTERFACE dusklight_mod_api dusklight_game_headers)
target_compile_definitions(mi_game_abi INTERFACE DUSK_MOD_FEATURE_GAME=1)
if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(mi_game_abi INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/FIglobal.h>")
else ()
    target_compile_options(mi_game_abi INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include global.h>")
endif ()

function(mi_add_component target_name)
    add_library(${target_name} STATIC ${ARGN})
    target_include_directories(${target_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
    set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

mi_add_component(mi_shared src/shared/core.cpp)
target_link_libraries(mi_shared PUBLIC dusklight_mod_api)

mi_add_component(mi_parser
        src/parser/dlparse.cpp
        src/parser/gctex.cpp
)

mi_add_component(mi_j3d_importer src/parser/j3d_importer.cpp)
target_link_libraries(mi_j3d_importer
        PUBLIC mi_parser
        PRIVATE mi_game_abi
)

mi_add_component(mi_game_runtime src/game/j3d_hook.cpp)
target_link_libraries(mi_game_runtime
        PUBLIC mi_j3d_importer mi_shared
        PRIVATE mi_game_abi
)

mi_add_component(mi_preview src/preview/preview.cpp)
target_link_libraries(mi_preview
        PUBLIC mi_shared
        PRIVATE dusklight_mod_feature_webgpu
)
_mod_add_webgpu_headers(mi_preview)

mi_add_component(mi_app src/app/inspector_service.cpp)
target_link_libraries(mi_app PUBLIC mi_shared mi_preview)

mi_add_component(mi_gui src/gui/dusklight_ui.cpp)
target_link_libraries(mi_gui PUBLIC mi_app dusklight_mod_api)

add_mod(runtimeviewer
        FEATURES game fmt webgpu
        SOURCES src/app/mod.cpp
        MOD_JSON "${CMAKE_CURRENT_BINARY_DIR}/mod.json"
        RES_DIR res
        OVERLAY_DIR overlay
        TEXTURES_DIR textures
)
target_include_directories(runtimeviewer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(runtimeviewer PRIVATE
        mi_game_runtime
        mi_gui
        mi_preview
        mi_shared
)

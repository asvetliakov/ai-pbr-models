cmake_minimum_required(VERSION 3.20)
project(AIPBRGui LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If using vcpkg manifest mode, it will discover header-only stb automatically.
# Prefer to find OpenCV first to avoid Protobuf target clashes when Torch also uses Protobuf
find_package(OpenCV REQUIRED core imgcodecs imgproc)
# Torch: set Torch_DIR to your LibTorch path (e.g., C:/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

# Workaround: if protobuf is present via OpenCV, ensure consistent linkage to avoid duplicate target exports
set(Protobuf_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)

add_executable(ai_pbr_gui WIN32
    src/main.cpp
)

# For Windows common controls and shell dialogs
if (WIN32)
    target_link_libraries(ai_pbr_gui PRIVATE comctl32 ole32 shlwapi)
    target_compile_definitions(ai_pbr_gui PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# Torch
# Note: Torch may require /bigobj on MSVC
if (MSVC)
    target_compile_options(ai_pbr_gui PRIVATE /bigobj /EHsc /W3 /permissive-)
else()
    target_compile_options(ai_pbr_gui PRIVATE -Wall -Wextra -Wno-unknown-pragmas)
endif()

# Link Torch libraries
# Torch suggests these properties to avoid MSVC runtime conflicts
set_property(TARGET ai_pbr_gui PROPERTY CXX_STANDARD 17)
target_link_libraries(ai_pbr_gui PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# Ensure runtime can locate the Torch DLLs at run-time (Windows)
if (WIN32)
    # Copy LibTorch DLLs next to the executable so PATH isn't required
    # Derive torch bin dir from one of the torch libraries
    if (TORCH_LIBRARIES)
        list(GET TORCH_LIBRARIES 0 TORCH_FIRST_LIB)
        get_filename_component(TORCH_LIB_DIR "${TORCH_FIRST_LIB}" DIRECTORY)
        get_filename_component(TORCH_ROOT "${TORCH_LIB_DIR}" DIRECTORY)
        set(TORCH_BIN_DIR "${TORCH_ROOT}/bin")
        if (EXISTS "${TORCH_BIN_DIR}")
            file(GLOB TORCH_DLLS "${TORCH_BIN_DIR}/*.dll")
            add_custom_command(TARGET ai_pbr_gui POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:ai_pbr_gui>
            )
        endif()
    endif()

    # Encourage vcpkg to deploy app-local dependencies (like OpenCV) automatically
    # Note: This takes effect when using vcpkg toolchain.
    set(VCPKG_APPLOCAL_DEPS ON CACHE BOOL "" FORCE)
    # Avoid accidental OpenCV dnn dependency; ensure only core/imgcodecs/imgproc are in use
    message(STATUS "OpenCV components: ${OpenCV_LIBS}")
endif()

# Include stb from vcpkg (header-only)
find_path(STB_INCLUDE_DIRS "stb_image.h")
if (STB_INCLUDE_DIRS)
    target_include_directories(ai_pbr_gui PRIVATE ${STB_INCLUDE_DIRS})
endif()

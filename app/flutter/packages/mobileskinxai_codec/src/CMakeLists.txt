cmake_minimum_required(VERSION 3.16)
project(mobileskinxai_codec LANGUAGES C)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(ENABLE_STATIC ON CACHE BOOL "" FORCE)
set(WITH_TURBOJPEG ON CACHE BOOL "" FORCE)
set(WITH_SIMD OFF CACHE BOOL "" FORCE)
set(WITH_TESTS OFF CACHE BOOL "" FORCE)
set(WITH_TOOLS OFF CACHE BOOL "" FORCE)
include(ExternalProject)
set(JPEG_BUILD ${CMAKE_CURRENT_BINARY_DIR}/vendor_build)
set(JPEG_ARGS -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DWITH_TURBOJPEG=ON
    -DWITH_SIMD=OFF -DWITH_TESTS=OFF -DWITH_TOOLS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_C_COMPILER_ARG1=${CMAKE_C_COMPILER_ARG1}
    -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM})
if(CMAKE_TOOLCHAIN_FILE)
  list(APPEND JPEG_ARGS -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE})
endif()
foreach(KEY ANDROID_ABI ANDROID_PLATFORM CMAKE_SYSTEM_NAME CMAKE_SYSTEM_PROCESSOR CMAKE_OSX_ARCHITECTURES CMAKE_OSX_SYSROOT CMAKE_OSX_DEPLOYMENT_TARGET)
  if(${KEY})
    list(APPEND JPEG_ARGS -D${KEY}=${${KEY}})
  endif()
endforeach()
ExternalProject_Add(jpeg_native SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/libjpeg-turbo
  BINARY_DIR ${JPEG_BUILD} CMAKE_ARGS ${JPEG_ARGS}
  BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target turbojpeg-static --parallel 4
  INSTALL_COMMAND "" BUILD_BYPRODUCTS ${JPEG_BUILD}/libturbojpeg.a)
add_library(turbojpeg_import STATIC IMPORTED)
set_target_properties(turbojpeg_import PROPERTIES IMPORTED_LOCATION ${JPEG_BUILD}/libturbojpeg.a)
add_library(mobileskinxai_codec SHARED codec.c)
target_include_directories(mobileskinxai_codec PRIVATE vendor/libjpeg-turbo/src ${CMAKE_CURRENT_BINARY_DIR}/vendor_build)
target_link_libraries(mobileskinxai_codec PRIVATE turbojpeg_import)
add_dependencies(mobileskinxai_codec jpeg_native)
if(NOT MSVC)
  target_compile_options(mobileskinxai_codec PRIVATE -ffp-contract=off)
  target_link_libraries(mobileskinxai_codec PRIVATE m)
endif()
if(ANDROID)
  target_link_options(mobileskinxai_codec PRIVATE "-Wl,-z,max-page-size=16384")
endif()
if(MSX_IOS_FRAMEWORK)
  set_target_properties(mobileskinxai_codec PROPERTIES FRAMEWORK TRUE
    FRAMEWORK_VERSION A OUTPUT_NAME MobileSkinXaiCodec
    MACOSX_FRAMEWORK_IDENTIFIER org.mobileskinxai.codec)
endif()

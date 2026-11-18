cmake_minimum_required(VERSION 3.16)
project(mobileskinxai_codec LANGUAGES C)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "codec_build")
set(mobileskinxai_codec_bundled_libraries $<TARGET_FILE:mobileskinxai_codec> PARENT_SCOPE)

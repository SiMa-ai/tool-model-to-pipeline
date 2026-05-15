#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2025 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************

cmake_minimum_required(VERSION 3.16)

set(plugin_version "1.7")
set(plugin_name "simaaiboxdecodebf16")
string(SUBSTRING ${plugin_name} 0 1 FIRST_CHAR)
string(TOUPPER ${FIRST_CHAR} FIRST_CHAR)
string(SUBSTRING ${plugin_name} 1 -1 REST_OF_STRING)
set(name_camel "${FIRST_CHAR}${REST_OF_STRING}")

project("${plugin_name}"
    VERSION 0.1
    DESCRIPTION "SiMa.ai ${plugin_name} plugin"
    HOMEPAGE_URL "https://bitbucket.org/sima-ai/gst-simaai-plugins-base/"
    LANGUAGES C CXX
)

if (NOT "$ENV{DEPENDENCIES_DIR_1}" STREQUAL "")
  set(TEMPLATES_DIR $ENV{DEPENDENCIES_DIR_1})
  set(CORE_ALLOCATOR_DIR $ENV{DEPENDENCIES_DIR_1}/allocator)
  set(CORE_BUFFER_POOL_DIR $ENV{DEPENDENCIES_DIR_1}/buffer-pool)
  set(CORE_CAPS_DIR $ENV{DEPENDENCIES_DIR_1}/caps)
  set(CORE_BUS_LOGGER_DIR $ENV{DEPENDENCIES_DIR_1}/bus-logger)
  set(CORE_SIMAAI_METADATA_DIR $ENV{DEPENDENCIES_DIR_1}/simaaimetadata)
else()
  set(TEMPLATES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../templates)
  set(CORE_ALLOCATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../core/allocator)
  set(CORE_BUFFER_POOL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../core/buffer-pool)
  set(CORE_CAPS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../core/caps)
  set(CORE_BUS_LOGGER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../core/bus-logger)
  set(CORE_SIMAAI_METADATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../core/simaaimetadata)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0)
pkg_check_modules(GSTREAMER_BASE REQUIRED IMPORTED_TARGET gstreamer-base-1.0)
pkg_check_modules(GLIB REQUIRED IMPORTED_TARGET glib-2.0)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)
pkg_check_modules(JSON_GLIB json-glib-1.0)

add_definitions(
    -DVERSION=\"${plugin_version}\"
    -DGST_LICENSE=\"LGPL\"
    -DGST_PACKAGE_NAME=\"SiMa.ai\ ${plugin_name}\ plugin\"
    -DGST_PACKAGE_ORIGIN=\"https://bitbucket.org/sima-ai/gst-simaai-plugins-base/\"
    -DPACKAGE=\"gst-simaai-plugins-base\"
    -DPLUGIN_NAME_LOWER=${plugin_name}
    -DPLUGIN_NAME_CAMEL=${name_camel}
)

add_library(${PROJECT_NAME}
    SHARED
    payload.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TEMPLATES_DIR}
    PUBLIC
    ${JSON_GLIB_INCLUDE_DIRS}
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
    ${CORE_ALLOCATOR_DIR}
    ${CORE_BUFFER_POOL_DIR}
    ${CORE_CAPS_DIR}
    ${CORE_BUS_LOGGER_DIR}
    ${CORE_SIMAAI_METADATA_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
    PkgConfig::GSTREAMER
    PkgConfig::GSTREAMER_BASE
    PkgConfig::GLIB
    PkgConfig::OPENCV
    simaaitrace
    simaailog
    gstsimaallocator
    gstsimaaibufferpool
    gstsimaaibuslogger
    gstsimaaicaps
    simaai_genboxdecode
    simaaimetadata
)

include(GNUInstallDirs)

INSTALL(TARGETS "${PROJECT_NAME}"  DESTINATION ${CMAKE_INSTALL_LIBDIR}/gstreamer-1.0)


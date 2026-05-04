set(afex_core_sources
  src/afex.cpp
  src/builtin_extractors.cpp
  src/rms_extractor.cpp
  src/zcr_extractor.cpp
)

if(IDF_TARGET)
  set(afex_enable_spectral_extractors_default OFF)
else()
  set(afex_enable_spectral_extractors_default ON)
endif()

option(AFEX_ENABLE_SPECTRAL_EXTRACTORS "Enable heavier spectral, pitch, onset, and voice extractors." ${afex_enable_spectral_extractors_default})

if(AFEX_ENABLE_SPECTRAL_EXTRACTORS)
  list(APPEND afex_core_sources src/spectral_voice_extractors.cpp)
endif()

if(IDF_TARGET)
  idf_component_register(
    SRCS ${afex_core_sources}
    INCLUDE_DIRS include
  )

  target_compile_features(${COMPONENT_LIB} PUBLIC cxx_std_20)
  target_compile_definitions(${COMPONENT_LIB}
    PUBLIC
      AFEX_ENABLE_EXCEPTIONS=0
      AFEX_ENABLE_SPECTRAL_EXTRACTORS=$<BOOL:${AFEX_ENABLE_SPECTRAL_EXTRACTORS}>
      AFEX_EMBEDDED=1
  )
else()
  option(AFEX_ENABLE_EXCEPTIONS "Enable throwing compatibility wrappers in afex host APIs." ON)

  add_library(afex
    ${afex_core_sources}
  )

  target_compile_features(afex PUBLIC cxx_std_20)
  set_target_properties(afex PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(afex
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

  target_compile_definitions(afex
    PUBLIC
      AFEX_ENABLE_EXCEPTIONS=$<BOOL:${AFEX_ENABLE_EXCEPTIONS}>
      AFEX_ENABLE_SPECTRAL_EXTRACTORS=$<BOOL:${AFEX_ENABLE_SPECTRAL_EXTRACTORS}>
  )
endif()

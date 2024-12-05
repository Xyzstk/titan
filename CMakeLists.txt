cmake_minimum_required(VERSION 3.15)

project(titan)

# Packages
find_package(fmt CONFIG REQUIRED)
find_package(LLVM CONFIG REQUIRED)
find_package(triton CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)

file(GLOB_RECURSE TITAN_SOURCES  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE TITAN_INCLUDES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp")
source_group(TREE ${PROJECT_SOURCE_DIR} FILES ${TITAN_SOURCES} ${TITAN_INCLUDES})

add_executable(${PROJECT_NAME} ${TITAN_SOURCES} ${TITAN_INCLUDES})

target_include_directories(${PROJECT_NAME} PRIVATE
  "src/"
)
target_compile_features(${PROJECT_NAME} PRIVATE 
  cxx_std_20
)
target_link_libraries(${PROJECT_NAME} PRIVATE 
  fmt::fmt
  range-v3 
  triton::triton
  LLVMWindowsManifest
  LLVMWindowsDriver
  LLVMXRay
  LLVMLibDriver
  LLVMDlltoolDriver
  LLVMCoverage
  LLVMLineEditor
  LLVMX86TargetMCA
  LLVMX86Disassembler
  LLVMX86AsmParser
  LLVMX86CodeGen
  LLVMX86Desc
  LLVMX86Info
  LLVMOrcJIT
  LLVMMCJIT
  LLVMJITLink
  LLVMInterpreter
  LLVMExecutionEngine
  LLVMRuntimeDyld
  LLVMOrcTargetProcess
  LLVMOrcShared
  LLVMDWP
  LLVMDebugInfoGSYM
  LLVMOption
  LLVMObjectYAML
  LLVMObjCopy
  LLVMMCA
  LLVMMCDisassembler
  LLVMLTO
  LLVMPasses
  LLVMCFGuard
  LLVMCoroutines
  LLVMObjCARCOpts
  LLVMipo
  LLVMVectorize
  LLVMLinker
  LLVMInstrumentation
  LLVMFrontendOpenMP
  LLVMFrontendOpenACC
  LLVMExtensions
  LLVMDWARFLinker
  LLVMGlobalISel
  LLVMMIRParser
  LLVMAsmPrinter
  LLVMSelectionDAG
  LLVMCodeGen
  LLVMIRReader
  LLVMAsmParser
  LLVMInterfaceStub
  LLVMFileCheck
  LLVMFuzzMutate
  LLVMTarget
  LLVMScalarOpts
  LLVMInstCombine
  LLVMAggressiveInstCombine
  LLVMTransformUtils
  LLVMBitWriter
  LLVMAnalysis
  LLVMProfileData
  LLVMSymbolize
  LLVMDebugInfoPDB
  LLVMDebugInfoMSF
  LLVMDebugInfoDWARF
  LLVMObject
  LLVMTextAPI
  LLVMMCParser
  LLVMMC
  LLVMDebugInfoCodeView
  LLVMBitReader
  LLVMFuzzerCLI
  LLVMCore
  LLVMRemarks
  LLVMBitstreamReader
  LLVMBinaryFormat
  LLVMTableGen
  LLVMSupport
  LLVMDemangle
)

add_subdirectory(intrinsics)

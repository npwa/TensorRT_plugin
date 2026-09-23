# Phase 5 step 1: the kernel -> plugin -> engine validation ladder (requirements.md §6)
# as an actual, repeatable test suite.
#
# Kernel-level: the pytest suite (tests/test_dequant_gemm_*_kernel.py) already IS this
# rung -- standalone kernel output vs. the PyTorch reference, run via `pytest tests/`.
# Not duplicated here as a ctest case: it needs the project venv's torch/pytest, and
# wiring a separate Python-environment activation into ctest for no benefit beyond what
# `pytest tests/` already gives would be process for its own sake.
#
# Plugin-level and engine-level are compiled C++ binaries already built by this same
# CMake project, so ctest is the natural runner for those two rungs.
set(VECTORS_DIR ${CMAKE_BINARY_DIR}/plugin_test_vectors)
set(ENGINE_PLAN ${VECTORS_DIR}/engine.plan)

add_test(NAME gen_test_vectors
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python3
            ${CMAKE_SOURCE_DIR}/python/reference/gen_plugin_test_vectors.py
            --out-dir ${VECTORS_DIR})
set_tests_properties(gen_test_vectors PROPERTIES FIXTURES_SETUP test_vectors)

# Plugin-level: the kernel wrapped in a minimal TensorRT test engine vs. the PyTorch
# reference (requirements.md §6, rung 2). Also produces engine.plan for the next rung.
add_test(NAME plugin_level_validation
    COMMAND $<TARGET_FILE:validate_plugin_engine> ${VECTORS_DIR} ${ENGINE_PLAN})
set_tests_properties(plugin_level_validation PROPERTIES
    FIXTURES_REQUIRED test_vectors
    FIXTURES_SETUP engine_plan)

# Engine-level: the full assembled engine (loaded through the Phase 4 harness, the same
# path a real deployment would use) vs. the PyTorch reference (requirements.md §6, rung 3).
add_test(NAME engine_level_validation
    COMMAND $<TARGET_FILE:run_single> ${ENGINE_PLAN} ${VECTORS_DIR})
set_tests_properties(engine_level_validation PROPERTIES FIXTURES_REQUIRED engine_plan)

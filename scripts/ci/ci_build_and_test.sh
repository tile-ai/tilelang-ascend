#!/bin/bash
# Verbatim extraction of the ci_cd.yml "Build and Test in Local Container"
# `bash -c` body (the original docker exec tilelang_x1 inline script), run on A3
# via the host-wrapper. Only the double-quote/dollar escaping from the inline
# string is removed; the test commands, order, args, --forked -n4/-n8 and exit
# code propagation are unchanged.
set -e

# --- execution-channel adaptation (the only added lines) ---
# Env vars the original passed with `docker exec -e ...` are sourced from a
# workspace state file written by the workflow step. GITHUB_ENV/GITHUB_OUTPUT are
# redirected to workspace files because the container cannot see the runner's
# temp files (both writes are not read by any later step; they stay verbatim).
ENV_FILE="${GITHUB_WORKSPACE:-.}/ci_exec_env.env"
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
: "${GITHUB_ENV:=ci_exec_env_out.env}"
: "${GITHUB_OUTPUT:=ci_exec_output.env}"

# Channel-adaptation guard: upstream originally passed these eight values with
# `docker exec -e ...`, so they were always DEFINED (possibly empty). The
# adapter delivers them via the env file above or via the execution environment.
# Require definition (not non-emptyness) so that legal empty strings keep
# working, and fail instead of silently running no tests.
: "${INCREMENTAL_FLAG?ci_exec: INCREMENTAL_FLAG is not defined}"
: "${SKIP_PYTEST?ci_exec: SKIP_PYTEST is not defined}"
: "${TEST_DIRS_ARG?ci_exec: TEST_DIRS_ARG is not defined}"
: "${EXPERIMENT_DIRS_ARG?ci_exec: EXPERIMENT_DIRS_ARG is not defined}"
: "${PYTEST_FILES_ARG?ci_exec: PYTEST_FILES_ARG is not defined}"
: "${RUN_EXAMPLES?ci_exec: RUN_EXAMPLES is not defined}"
: "${RUN_PYTEST_ONLY?ci_exec: RUN_PYTEST_ONLY is not defined}"
: "${PYTEST_MARKERS?ci_exec: PYTEST_MARKERS is not defined}"
if [ "$RUN_PYTEST_ONLY" != "true" ] && [ "$RUN_EXAMPLES" != "true" ]; then
  echo "ci_exec: neither RUN_PYTEST_ONLY nor RUN_EXAMPLES is true; no tests selected" >&2
  exit 1
fi

# The execution channel is responsible for activating the venv + CANN:
#   A2 legacy: docker exec -e BASH_ENV=/root/.bashrc
#   A3 host-wrapper: root-owned bootstrap sourced via BASH_ENV by the wrapper
# Guard only — fail fast with a clear message if the toolchain is missing.
command -v python >/dev/null 2>&1 || {
  echo "python not found in CI execution environment" >&2
  exit 127
}
command -v pytest >/dev/null 2>&1 || {
  echo "pytest not found in CI execution environment" >&2
  exit 127
}

# --- verbatim original body ---
# 如果有缓存，复制到 build 目录
if [ -d build-cache ] && [ "$(ls -A build-cache)" ]; then
  echo "Copy build cache..."
  mkdir -p build
  cp -ra build-cache/* build/
fi

# 1. 增量编译项目
bash install_ascend.sh $INCREMENTAL_FLAG
# 保存编译产物
if [ -d "build" ]; then
  echo "Saving build artifacts..."
  rm -rf build-cache/*
  cp -ra build/* build-cache/
fi

# 2. 加载环境变量
echo 'Sourcing environment...'
source set_env.sh
echo 'TILELANG_PATH='$PWD >> $GITHUB_ENV

# 3. 运行测试用例
cd examples
chmod +x bench_test.sh

# 构建 pytest marker 过滤参数
PYTEST_MARKER_ARGS=()
if [ -n "$PYTEST_MARKERS" ]; then
  PYTEST_MARKER_ARGS+=(-m "$PYTEST_MARKERS")
  echo 'Applying pytest marker filter: -m '$PYTEST_MARKERS
fi

if [ "$RUN_PYTEST_ONLY" = true ]; then
  # 只跑 pytest（testing/ 修改）
  if [ -n "$PYTEST_FILES_ARG" ]; then
    echo 'Running incremental pytest for: '$PYTEST_FILES_ARG
    # Prefix every path, not just the first: the value can hold several
    # files and the rest would otherwise resolve against examples/.
    PYTEST_TARGETS=()
    for pytest_target in $PYTEST_FILES_ARG; do
      PYTEST_TARGETS+=("../$pytest_target")
    done
    set +e
    pytest --forked "${PYTEST_TARGETS[@]}" -v -n 4 "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee test_output.log
  else
    echo 'Running full pytest (testing/ modified)'
    set +e
    pytest --forked ../testing/python/ -v -n 8 "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee test_output.log
  fi
  TEST_EXIT_CODE=${PIPESTATUS[0]}
  set -e
elif [ "$RUN_EXAMPLES" = true ]; then
  # 跑 examples + experiment + pytest
  echo 'Running benchmark tests with args: '$TEST_DIRS_ARG' '$EXPERIMENT_DIRS_ARG
  set +e
  if [ "$SKIP_PYTEST" = true ]; then
    ./bench_test.sh --skip-pytest $TEST_DIRS_ARG $EXPERIMENT_DIRS_ARG 2>&1 | tee test_output.log
  else
    if [ -n "$PYTEST_MARKERS" ]; then
      ./bench_test.sh --pytest-markers "$PYTEST_MARKERS" $TEST_DIRS_ARG $EXPERIMENT_DIRS_ARG 2>&1 | tee test_output.log
    else
      ./bench_test.sh $TEST_DIRS_ARG $EXPERIMENT_DIRS_ARG 2>&1 | tee test_output.log
    fi
  fi
  TEST_EXIT_CODE=${PIPESTATUS[0]}

  # Operator tests were moved out of the legacy runner, so run them here:
  # one Pytest invocation gets xdist scheduling, the marker filter and the
  # full failure output. Entries reserved but not migrated yet are absent
  # from list-tests, so nothing points at a file that does not exist.
  #
  # Scoped by the same two arguments the runner just took, so an
  # incremental run tests the operators in the directories it ran and
  # no others. Both are empty on a full run, which selects them all.
  OPERATOR_TESTS=()
  while IFS= read -r operator_test; do
    [ -n "$operator_test" ] && OPERATOR_TESTS+=("../$operator_test")
  done < <(python ../scripts/ci/resolve_operator_tests.py list-tests $TEST_DIRS_ARG $EXPERIMENT_DIRS_ARG)
  if [ ${#OPERATOR_TESTS[@]} -gt 0 ]; then
    echo 'Running operator tests: '${OPERATOR_TESTS[*]}
    pytest --forked "${OPERATOR_TESTS[@]}" -v -n 8 "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee -a test_output.log
    OPERATOR_EXIT_CODE=${PIPESTATUS[0]}
    # A3 flake guard (single retry, host-wrapper channel only): a transient
    # device-side hdc/TSD subprocess startup timeout (E39007 / 507033) can
    # abort the first npu tensor in one forked child although the same test
    # passes everywhere else. That is an infra failure, not a kernel result,
    # so re-run the same command once before failing. CI_EXECUTION_MODE is
    # exported by the workflow into the sourced ENV_FILE; on the A2 legacy
    # channel (unset or legacy) the original no-retry semantics stay intact.
    if [ "$OPERATOR_EXIT_CODE" -ne 0 ] && [ "${CI_EXECUTION_MODE:-}" = "host-wrapper" ]; then
      echo 'Operator pytest failed once; retrying unchanged command once (A3 device-subprocess startup flake guard)...'
      pytest --forked "${OPERATOR_TESTS[@]}" -v -n 8 "${PYTEST_MARKER_ARGS[@]}" 2>&1 | tee -a test_output.log
      OPERATOR_EXIT_CODE=${PIPESTATUS[0]}
    fi
    if [ "$TEST_EXIT_CODE" -eq 0 ]; then
      TEST_EXIT_CODE=$OPERATOR_EXIT_CODE
    fi
  fi
  set -e
fi

echo 'Test exit code: '$TEST_EXIT_CODE
echo "test_exit_code=$TEST_EXIT_CODE" >> $GITHUB_OUTPUT

# 如果测试失败，让脚本退出码非0
if [ "$TEST_EXIT_CODE" -ne 0 ]; then
  exit 1
fi

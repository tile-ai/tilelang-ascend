ensure_torch_tl_ascend() {
    local output
    if ! output=$(pip show torch_tl_ascend 2>&1); then
        echo "torch_tl_ascend is not installed. Installing with python setup.py install ..."
        python setup.py install || { echo "Failed to install package"; exit 1; }
        echo "Installed torch_tl_ascend"
        return 0
    fi

    local expected_version=$(tr -d ' \t\r\n' < src/torch_tl_ascend/VERSION)  # read local VERSION file and remove whitespace/newlines
    local installed_version=$(echo "$output" | awk -F': ' '/^Version:/{print $2}' | tr -d ' \t\r\n')  # find installed version

    if [ "$installed_version" != "$expected_version" ]; then
        echo "torch_tl_ascend v$installed_version is installed. Expected v$expected_version. Reinstalling ..."
        echo "Cleaning previous build artifacts ..."
        rm -rf build/ dist/
        rm -rf src/*.egg-info/
        pip uninstall -y torch_tl_ascend || { echo "Failed to uninstall outdated package"; exit 1; }
        python setup.py install || { echo "Failed to install package"; exit 1; }
        echo "Installed torch_tl_ascend v$expected_version"
        return 0
    fi

    echo "torch_tl_ascend v$installed_version is already installed"
    return 0
}

ensure_torch_tl_ascend

echo "Running test_torch.py"
python test_torch.py || { echo "python test_torch.py failed"; exit 1; }
echo "Running test_source.py"
python test_source.py || { echo "python test_source.py failed"; exit 1; }

echo "All Test Passed!"
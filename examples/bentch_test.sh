#!/bin/bash

echo "Starting Python script execution..."
echo "====================================="

total_scripts=0
passed_scripts=0
failed_scripts=0
failed_files=()

# Find all Python files in current directory
python_files=$(find . -maxdepth 2 -name "*.py" | sort)

if [ -z "$python_files" ]; then
    echo "No Python files found in current directory."
    exit 0
fi

echo "Found Python files:"
for file in $python_files; do
    echo "  - $file"
done
echo

for file in $python_files; do
    echo "Executing: $file"
    total_scripts=$((total_scripts + 1))

    # Execute the Python script and capture output
    output=$(python "$file" 2>&1)
    exit_code=$?

    # Get the last line of output
    last_line=$(echo "$output" | tail -n 1)

    # Check if last line contains 'Kernel Output Match' OR 'Test Passed!' (case insensitive)
    if [[ "$last_line" =~ [Kk][Ee][Rr][Nn][Ee][Ll][[:space:]][Oo][Uu][Tt][Pp][Uu][Tt][[:space:]][Mm][Aa][Tt][Cc][Hh] ]] || [[ "$last_line" =~ [Tt][Ee][Ss][Tt][[:space:]][Pp][Aa][Ss][Ss][Ee][Dd][!] ]]; then
        echo "  Status: PASSED"
        echo "  Last line: $last_line"
        passed_scripts=$((passed_scripts + 1))
    else
        echo "  Status: FAILED"
        echo "  Exit code: $exit_code"
        echo "  Last line: $last_line"
        echo "  Full output (last 10 lines):"
        echo "  ----------------------------------------"
        echo "$output" | tail -n 10 | sed 's/^/  /'
        echo "  ----------------------------------------"
        
        if [ $exit_code -eq 0 ]; then
            echo "  Note: Script executed successfully but didn't output expected pass phrase"
        else
            echo "  Error: Script execution failed with exit code $exit_code"
        fi
        failed_scripts=$((failed_scripts + 1))
        failed_files+=("$file")
    fi
    echo
done

echo "====================================="
echo "Execution Summary"
echo "====================================="
echo "Total Python scripts executed: $total_scripts"
echo "Scripts with expected pass output: $passed_scripts"
echo "Scripts without expected pass output: $failed_scripts"
echo

if [ $failed_scripts -gt 0 ]; then
    echo "Failed scripts:"
    for file in "${failed_files[@]}"; do
        echo "  - $file"
    done
    echo
fi
# Calculate and display percentage
if [ $total_scripts -gt 0 ]; then
    percentage=$((passed_scripts * 100 / total_scripts))
    echo "Pass rate: $percentage% ($passed_scripts/$total_scripts)"
else
    echo "Pass rate: 0% (0/0)"
fi

echo "====================================="
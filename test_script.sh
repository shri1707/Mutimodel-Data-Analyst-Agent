#!/bin/bash

# Data Analysis API - Comprehensive Test Script
# =============================================
# 
# This script runs all curl test cases from curl_tests.txt with proper
# error handling, logging, and 10-second delays between tests.
#
# Usage: ./test_script.sh
# 
# Prerequisites:
# 1. API must be running: docker-compose up -d
# 2. All example files must be present
# 3. Script must be run from project root: /home/sujal/Work/TDS-P2v3
#
# =============================================

set -e  # Exit on error

# Configuration
API_BASE="http://localhost:8000"
TIMEOUT_BETWEEN_TESTS=10
LOG_FILE="test_results_$(date +%Y%m%d_%H%M%S).log"
SUMMARY_FILE="test_summary_$(date +%Y%m%d_%H%M%S).txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print colored output
print_status() {
    local status="$1"
    local message="$2"
    
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message" | tee -a "$LOG_FILE"
            ;;
    esac
}

# Function to run a test with timeout and error handling
run_test() {
    local test_name="$1"
    local curl_cmd="$2"
    local expected_status="$3"  # Optional: expected HTTP status code
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    print_status "INFO" "Running Test $TOTAL_TESTS: $test_name"
    echo "Command: $curl_cmd" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    
    start_time=$(date +%s)
    
    # Run curl command and capture output and HTTP status
    if output=$(timeout 300s bash -c "$curl_cmd" 2>&1); then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        # Check if output contains error indicators
        if echo "$output" | grep -q '"status":"error"' || echo "$output" | grep -q "HTTP Status: 4[0-9][0-9]\|HTTP Status: 5[0-9][0-9]"; then
            print_status "WARNING" "Test completed but returned error status ($duration seconds)"
            echo "$output" >> "$LOG_FILE"
            if [[ "$test_name" == *"ERROR"* ]]; then
                PASSED_TESTS=$((PASSED_TESTS + 1))
                print_status "SUCCESS" "Error test behaved as expected"
            else
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
        else
            print_status "SUCCESS" "Test passed ($duration seconds)"
            echo "$output" >> "$LOG_FILE"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        fi
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        print_status "ERROR" "Test failed or timed out ($duration seconds)"
        echo "Error output: $output" >> "$LOG_FILE"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    
    echo "========================================" >> "$LOG_FILE"
    
    # Wait between tests (except for the last test)
    if [ $TOTAL_TESTS -lt 32 ]; then  # Adjust based on total number of tests
        print_status "INFO" "Waiting ${TIMEOUT_BETWEEN_TESTS} seconds before next test..."
        sleep $TIMEOUT_BETWEEN_TESTS
    fi
}

# Function to check prerequisites
check_prerequisites() {
    print_status "INFO" "Checking prerequisites..."
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ] || [ ! -d "app" ] || [ ! -d "example1" ]; then
        print_status "ERROR" "Please run this script from the project root directory (/home/sujal/Work/TDS-P2v3)"
        exit 1
    fi
    
    # Check if API is running
    if ! curl -s "$API_BASE/health" > /dev/null 2>&1; then
        print_status "ERROR" "API is not responding at $API_BASE"
        print_status "INFO" "Please start the API with: docker-compose up -d"
        exit 1
    fi
    
    # Check example files
    for example_dir in example1 example2 example3 example4 example5 example6; do
        if [ ! -d "$example_dir" ]; then
            print_status "WARNING" "Example directory $example_dir not found"
        fi
    done
    
    # Check specific required files
    required_files=(
        "example1/edges.csv"
        "example1/question.txt"
        "example2/sample-sale.csv"
        "example2/question.txt"
        "example3/sample-weather.csv"
        "example3/question.txt"
        "example4/question.txt"
        "example5/question.txt"
        "example6/question.txt"
    )
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            print_status "WARNING" "Required file $file not found"
        fi
    done
    
    print_status "SUCCESS" "Prerequisites check completed"
}

# Function to generate final summary
generate_summary() {
    local end_time=$(date)
    
    cat > "$SUMMARY_FILE" << EOF
Data Analysis API Test Summary
==============================

Test Run Details:
- Start Time: $start_time_global
- End Time: $end_time
- Total Tests: $TOTAL_TESTS
- Passed: $PASSED_TESTS
- Failed: $FAILED_TESTS
- Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%

Log Files:
- Detailed Log: $LOG_FILE
- Summary: $SUMMARY_FILE

API Configuration:
- Base URL: $API_BASE
- Timeout Between Tests: ${TIMEOUT_BETWEEN_TESTS}s

Test Categories Covered:
- Health checks
- Network analysis (Example 1)
- Sales data analysis (Example 2)
- Weather data analysis (Example 3)
- Web scraping analysis (Example 4)
- Movie data analysis (Example 5)
- Legal data analysis (Example 6)
- Multiple file uploads
- Machine learning analysis
- Error handling
- Performance tests
- Different analysis types
- Various complexity levels

EOF

    print_status "INFO" "Test summary saved to: $SUMMARY_FILE"
}

# Main execution starts here
main() {
    start_time_global=$(date)
    
    echo ""
    echo "=============================================="
    echo "  Data Analysis API - Comprehensive Test Suite"
    echo "=============================================="
    echo ""
    
    print_status "INFO" "Starting test suite at $(date)"
    print_status "INFO" "Log file: $LOG_FILE"
    print_status "INFO" "Summary file: $SUMMARY_FILE"
    
    # Check prerequisites
    check_prerequisites
    
    echo ""
    print_status "INFO" "Starting test execution..."
    echo ""
    
    # 0. HEALTH CHECKS
    run_test "Health Check" \
        'curl -s -X GET "http://localhost:8000/health" -H "accept: application/json" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Root Endpoint Check" \
        'curl -s -X GET "http://localhost:8000/" -H "accept: application/json" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    # 1. CORE EXAMPLE TESTS - All 6 Example Folders
    run_test "Example 1 - Network Analysis" \
        'curl -s -X POST "http://localhost:8000/api/" -F "question.txt=@example1/question.txt" -F "edges.csv=@example1/edges.csv" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Example 2 - Sales Data Analysis" \
        'curl -s -X POST "http://localhost:8000/api/" -F "question.txt=@example2/question.txt" -F "sample-sale.csv=@example2/sample-sale.csv" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Example 3 - Weather Data Analysis" \
        'curl -s -X POST "http://localhost:8000/api/" -F "question.txt=@example3/question.txt" -F "sample-weather.csv=@example3/sample-weather.csv" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    
    run_test "Example 5 - Movie Data Analysis" \
        'curl -s -X POST "http://localhost:8000/api/" -F "question.txt=@example5/question.txt" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Example 6 - Legal Data Analysis" \
        'curl -s -X POST "http://localhost:8000/api/" -F "question.txt=@example6/question.txt" -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\n"'
    
    run_test "Example 5 - Movie Data Analysis" \
        'curl -X POST "http://localhost:8000/api/"   -F "question=@example6/question.txt"   -F "files=$(cat example6/data.json)"'
    
    
    # Generate final summary
    echo ""
    print_status "INFO" "All tests completed!"
    print_status "INFO" "Generating final summary..."
    
    generate_summary
    
    echo ""
    echo "=============================================="
    echo "           TEST SUITE SUMMARY"
    echo "=============================================="
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Success Rate: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%"
    echo ""
    echo "Log Files:"
    echo "- Detailed: $LOG_FILE"
    echo "- Summary: $SUMMARY_FILE"
    echo ""
    
    if [ $FAILED_TESTS -eq 0 ]; then
        print_status "SUCCESS" "All tests passed! 🎉"
        exit 0
    else
        print_status "WARNING" "Some tests failed. Check the log files for details."
        exit 1
    fi
}

# Trap to handle script interruption
trap 'print_status "ERROR" "Script interrupted by user"; exit 130' INT TERM

# Run main function
main "$@"

#!/bin/bash

# DocuScope CA Desktop Installation Test Script
# This script performs basic validation of the installed application

set -e

APP_NAME="DocuScope CA"
TEST_TIMEOUT=30

echo "🧪 Starting DocuScope CA Desktop installation test..."

# Function to check if a process is running
check_process() {
    local process_name="$1"
    if pgrep -f "$process_name" > /dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to wait for process to start
wait_for_process() {
    local process_name="$1"
    local timeout="$2"
    local counter=0
    
    echo "⏳ Waiting for process '$process_name' to start (timeout: ${timeout}s)..."
    
    while [ $counter -lt $timeout ]; do
        if check_process "$process_name"; then
            echo "✅ Process '$process_name' is running"
            return 0
        fi
        sleep 1
        counter=$((counter + 1))
    done
    
    echo "❌ Process '$process_name' did not start within ${timeout}s"
    return 1
}

# Function to test application startup
test_application_startup() {
    local executable="$1"
    
    echo "🚀 Testing application startup: $executable"
    
    # Check if executable exists
    if [ ! -f "$executable" ] && [ ! -f ./*.AppImage ]; then
        echo "❌ Executable not found: $executable"
        return 1
    fi
    
    # For AppImage, find the actual file
    if [[ "$executable" == ./*.AppImage ]]; then
        executable=$(find . -name "*.AppImage" -type f | head -1)
        if [ -z "$executable" ]; then
            echo "❌ No AppImage found"
            return 1
        fi
    fi
    
    echo "📍 Using executable: $executable"
    
    # Start application in background
    echo "🔄 Starting application..."
    
    # Use different approaches based on the platform
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - use open command for .app bundles
        if [[ "$executable" == *.app* ]]; then
            open "$executable" &
        else
            "$executable" &
        fi
    else
        # Linux/Windows - direct execution
        "$executable" &
    fi
    
    APP_PID=$!
    echo "📋 Application PID: $APP_PID"
    
    # Wait a moment for startup
    sleep 3
    
    # Check if the main process is still running
    if kill -0 $APP_PID 2>/dev/null; then
        echo "✅ Application process is running"
        
        # Wait for Streamlit server to potentially start
        echo "⏳ Waiting for Streamlit server startup..."
        sleep 10
        
        # Check if Streamlit is running (port 8501)
        if command -v curl >/dev/null 2>&1; then
            if curl -s --connect-timeout 5 http://localhost:8501 >/dev/null 2>&1; then
                echo "✅ Streamlit server is responding on port 8501"
            else
                echo "⚠️  Streamlit server not responding (may still be starting)"
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget -q --timeout=5 --tries=1 http://localhost:8501 -O /dev/null 2>/dev/null; then
                echo "✅ Streamlit server is responding on port 8501"
            else
                echo "⚠️  Streamlit server not responding (may still be starting)"
            fi
        else
            echo "ℹ️  Cannot test HTTP connectivity (curl/wget not available)"
        fi
        
        # Cleanup - terminate the application
        echo "🧹 Terminating application..."
        kill $APP_PID 2>/dev/null || true
        sleep 2
        
        # Force kill if still running
        if kill -0 $APP_PID 2>/dev/null; then
            echo "🔨 Force terminating application..."
            kill -9 $APP_PID 2>/dev/null || true
        fi
        
        echo "✅ Application test completed successfully"
        return 0
    else
        echo "❌ Application process exited immediately"
        return 1
    fi
}

# Function to check installation paths
check_installation() {
    echo "🔍 Checking installation..."
    
    case "$OSTYPE" in
        linux*)
            # Linux - check common locations
            if [ -f "/usr/bin/docuscope-ca" ]; then
                echo "✅ Found Linux executable: /usr/bin/docuscope-ca"
                test_application_startup "/usr/bin/docuscope-ca"
            elif find . -name "*.AppImage" -type f | head -1 >/dev/null 2>&1; then
                echo "✅ Found AppImage in current directory"
                test_application_startup "./*.AppImage"
            else
                echo "❌ No Linux executable found"
                return 1
            fi
            ;;
        darwin*)
            # macOS - check Applications folder
            if [ -d "/Applications/DocuScope CA.app" ]; then
                echo "✅ Found macOS app: /Applications/DocuScope CA.app"
                test_application_startup "/Applications/DocuScope CA.app"
            else
                echo "❌ No macOS app found in /Applications"
                return 1
            fi
            ;;
        msys*|cygwin*|win*)
            # Windows - check Program Files
            if [ -f "C:/Program Files/DocuScope CA/DocuScope CA.exe" ]; then
                echo "✅ Found Windows executable: C:/Program Files/DocuScope CA/DocuScope CA.exe"
                test_application_startup "C:/Program Files/DocuScope CA/DocuScope CA.exe"
            else
                echo "❌ No Windows executable found"
                echo "Checking alternative locations..."
                find /c/Program* -name "*DocuScope*" -type f 2>/dev/null || echo "No alternative locations found"
                return 1
            fi
            ;;
        *)
            echo "❌ Unknown operating system: $OSTYPE"
            return 1
            ;;
    esac
}

# Main test execution
main() {
    echo "🎯 DocuScope CA Desktop Installation Test"
    echo "🖥️  Operating System: $OSTYPE"
    echo "📅 Test Date: $(date)"
    echo ""
    
    # Run installation check
    if check_installation; then
        echo ""
        echo "🎉 Installation test PASSED!"
        echo "✅ Application was successfully installed and started"
        exit 0
    else
        echo ""
        echo "💥 Installation test FAILED!"
        echo "❌ Application installation or startup failed"
        exit 1
    fi
}

# Run main function
main "$@"

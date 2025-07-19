#!/usr/bin/env python3
"""
Test runner script for the stock analysis application.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle output."""
    print(f"\n🔧 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    
    if result.returncode != 0:
        print(f"❌ {description} failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def main():
    """Main test runner."""
    print("🧪 Stock Analysis Application Test Suite")
    print("=" * 60)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Check if we're in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Warning: Not running in a virtual environment")
        print("   Consider activating your virtual environment first")
    
    # Install test dependencies
    install_success = run_command(
        "pip install -r tests/requirements.txt",
        "Installing test dependencies"
    )
    
    if not install_success:
        print("❌ Failed to install test dependencies")
        return 1
    
    # Run different test categories
    test_commands = [
        # Unit tests
        ("pytest tests/unit/ -v --tb=short", "Running unit tests"),
        
        # Integration tests
        ("pytest tests/integration/ -v --tb=short", "Running integration tests"),
        
        # All tests with coverage
        ("pytest tests/ --cov=app --cov-report=term-missing --cov-report=html", 
         "Running all tests with coverage"),
        
        # Security tests
        ("bandit -r app/ -f json -o security_report.json", "Running security analysis"),
        
        # Safety check for dependencies
        ("safety check --json --output safety_report.json", "Checking dependency safety"),
    ]
    
    results = []
    
    for command, description in test_commands:
        success = run_command(command, description)
        results.append((description, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {description}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} test categories passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
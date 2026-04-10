# Development Workflow

Always perform all tasks on a feature branch. When starting a task, check whether you are already on a feature branch. If not, please create a new feature branch with a descriptive name starting with claude/. For example, claude/my-new-feature.

After you complete a task, please commit your changes and push them to Github. Open a PR if one does not already exist.

Pre-commit checks will run when you try to commit. Please fix all issues surfaced by pre-commit checks.

# Testing

Please write comprehensive unit and integration tests for new changes.

Test all nodes and tools as well as the overall graph.
Test all logic branches and common error cases.

Put all the test files in the tests directory. This directory should have the same subdirectory structure as the original repo. For example, to test the file agent/graph.py the test_graph.py file should be in tests/agent/test_graph.py

Do not test private methods in a module.

# Python Package management

uv is installed locally. Please use it for Python package management.

# Contributing to Vietnamese ASR

Thank you for considering contributing to the Vietnamese ASR project! This document provides guidelines and best practices for contributing to the codebase.

## Development Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tuandung222/Convert-PhoWhisper-ASR-from-encdec-to-ctc.git
   cd Convert-PhoWhisper-ASR-from-encdec-to-ctc
   ```

2. Install the package with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Style and Quality

This project uses several tools to ensure code quality and consistency:

- **Black**: For code formatting
- **isort**: For import sorting
- **Flake8**: For linting
- **MyPy**: For type checking
- **Bandit**: For security checks

These tools run automatically through pre-commit hooks when you commit changes. You can also run them manually:

```bash
# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run specific hooks
pre-commit run black --all-files
pre-commit run flake8 --all-files
```

## Development Workflow

1. **Create a branch** from the `main` branch for your changes.
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit them, ensuring pre-commit hooks pass:
   ```bash
   git add .
   git commit -m "Your descriptive commit message"
   ```

3. **Push your branch** to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Submit a pull request** to the `main` branch.

## Adding Dependencies

If you need to add new dependencies:

1. Add them to the appropriate requirements file:
   - `api/requirements.txt` for API dependencies
   - `ui/requirements.txt` for UI dependencies

2. Update `setup.py` to include the new dependencies in both:
   - `install_requires` for core dependencies
   - The appropriate section of `extras_require`

## Testing

Before submitting a pull request, make sure your changes pass all tests:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest
```

## Documentation

Please update documentation when making changes, including:

- Code docstrings (using Google style)
- README.md updates for user-facing changes
- Comments in configuration files when appropriate

## Docker Changes

If you modify Docker-related files:

1. Ensure the changes are consistent across all Docker Compose files
2. Test the changes locally using the appropriate Docker Compose command
3. Update the Docker documentation in `docker/README.md` as needed

## Submitting Pull Requests

When submitting a pull request, please:

1. Provide a clear description of the changes
2. Reference any related issues
3. Ensure all checks pass (pre-commit, tests, etc.)
4. Update relevant documentation

## License

By contributing to this project, you agree that your contributions will be licensed under the project's MIT License.

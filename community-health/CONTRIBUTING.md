# Contributing to GESL

GESL is an open source project. We welcome contributions via pull requests as well as questions, feature requests, or bug reports via issues. Contact our team with any questions. Please also refer to our [code of conduct](CODE_OF_CONDUCT.md).

If you aren't a GESL developer at LLNL, you won't have permission to push new branches to the repository. First, you should create a fork. This will create your copy of the GESL repository and ensure you can push your changes up to GitHub and create PRs.

## Quick Start

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GESL.git
   cd GESL
   ```
3. Set up your development environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
4. Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Contribution Guidelines

* **Create your branches off the `main` branch.** Always branch from the latest `main` to ensure your work is based on the most recent code.
* **Clearly name your branches, commits, and PRs** as this will help us manage queued work in a timely manner.
  * Branch names: Use descriptive names like `feature/add-shap-visualization` or `bugfix/fix-tertiary-classifier`
  * Commit messages: Write in the imperative (e.g., "Add new LIME explanation method" or "Fix feature scaling in inference pipeline")
* **Commit your work in logically organized commits**, and group commits together logically in a PR.
* **Title each PR clearly** and give it an unambiguous description. Include:
  * What problem does this PR solve?
  * What approach did you take?
  * Any breaking changes or dependencies?
  * Screenshots/examples if applicable (especially for UI/visualization changes)
* **Review existing issues before opening a new one.** Your issue might already be under development or discussed by others. Feel free to add to any outstanding issue/bug.
* **Be explicit when opening issues and reporting bugs.** Include:
  * Expected behavior vs. actual behavior
  * Steps to reproduce the issue
  * Your environment (Python version, OS, etc.)
  * Relevant error messages or logs
  * Sample data or minimal reproducible example if applicable

## Development Best Practices

### Code Style

* Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines for Python code
* Use meaningful variable and function names
* Add docstrings to functions and classes
* Keep functions focused and modular

### Testing

* Test your changes thoroughly before submitting a PR
* For classifier changes, verify:
  * Training pipeline completes without errors
  * Inference produces expected results
  * XAI methods (SHAP, LIME, feature importance) work correctly
* For server changes, test with both `server.py` and `demo_server.py`
* For UI changes, test `viewer.html` in multiple browsers if possible

### Documentation

* Update relevant documentation when adding new features
* Add comments for complex logic or non-obvious implementation details
* Update README.md if you're adding new dependencies or changing setup steps

## Pull Request Checklist

Before submitting your PR, make sure you have:

- [ ] Tested your changes locally
- [ ] Updated documentation as needed
- [ ] Written clear commit messages
- [ ] Verified your code follows existing style conventions
- [ ] Checked that your changes don't break existing functionality
- [ ] Signed off your commits with `git commit --signoff`

## Commit Sign-off

All commits must be signed off, indicating that you agree to the [Developer Certificate of Origin](https://developercertificate.org/). Use the `--signoff` flag:

```bash
git commit --signoff -m "Add new feature"
```

## Types of Contributions

We welcome various types of contributions:

* **Bug fixes**: Help us identify and fix issues
* **New features**: Add new XAI methods, classifiers, or functionality
* **Documentation**: Improve README, add examples, or clarify existing docs
* **Performance improvements**: Optimize training, inference, or visualization
* **Testing**: Add test coverage or improve existing tests
* **UI/UX**: Enhance the web viewer interface
* **Data processing**: Improve data augmentation or feature engineering

## Questions?

If you have questions about contributing, please contact the project team (see Authors section in README.md) or open a GitHub issue with the `question` label.

Thank you for contributing to GESL!

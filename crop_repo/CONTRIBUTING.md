# 🤝 Contributing to Crop Recommendation System

Thank you for considering contributing! Every contribution helps improve this project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

---

## Code of Conduct

Be respectful, constructive, and inclusive. This project is open to everyone.

---

## How to Contribute

### 🐛 Reporting Bugs
1. Check [existing issues](../../issues) first
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs. actual behaviour
   - Your OS, Python version, and library versions

### 💡 Suggesting Features
Open an issue with the label `enhancement` and describe:
- The problem it solves
- Your proposed solution
- Any alternatives considered

### 🔧 Submitting Code

**Good first issues:** Look for issues tagged `good first issue` or `help wanted`.

---

## Development Setup

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/crop-recommendation.git
cd crop-recommendation

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a feature branch
git checkout -b feature/your-feature-name
```

---

## Pull Request Process

1. Ensure your code runs without errors
2. Update `README.md` if you've changed functionality
3. Add/update docstrings for any new functions
4. Keep PRs focused — one feature or fix per PR
5. Write a clear PR description explaining *what* and *why*

```bash
git add .
git commit -m "feat: add SHAP explainability section"
git push origin feature/your-feature-name
# Then open a Pull Request on GitHub
```

### Commit Message Convention

```
feat:     New feature
fix:      Bug fix
docs:     Documentation changes
style:    Formatting, no logic change
refactor: Code restructure
test:     Adding tests
chore:    Build process or tooling
```

---

## Style Guidelines

- Follow **PEP 8** for Python code
- Use **meaningful variable names** (no single letters except loop counters)
- Add **docstrings** to all functions
- Keep **notebook cells focused** — one concept per cell
- Use `# ── Section Header ──` style comments for readability

---

Thank you for making this project better! 🌾

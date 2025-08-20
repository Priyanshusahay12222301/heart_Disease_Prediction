# Contributing to Heart Disease Prediction System

Thank you for your interest in contributing to the Enhanced Heart Disease Prediction System! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Types of Contributions](#types-of-contributions)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)
- [Community](#community)

## 🤝 Code of Conduct

### Our Pledge
We are committed to making participation in this project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Expected Behavior
- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with Flask web development
- Knowledge of HTML/CSS/JavaScript for frontend contributions

### Quick Start
1. Fork the repository
2. Clone your fork locally
3. Set up the development environment
4. Make your changes
5. Test your changes
6. Submit a pull request

## 🛠 Development Setup

### 1. Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/heart_Disease_Prediction.git
cd heart_Disease_Prediction-main
```

### 2. Set Up Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
./.venv/Scripts/Activate.ps1

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install production dependencies
pip install -r enhanced_requirements.txt

# For development, also install testing tools
pip install pytest pytest-flask coverage black flake8
```

### 4. Set Up Pre-commit Hooks (Optional but Recommended)
```bash
pip install pre-commit
pre-commit install
```

### 5. Verify Installation
```bash
cd "heart disease predictor"
python syntax_test.py
```

## 📝 Contributing Guidelines

### Before You Start
1. **Check existing issues** - Look for existing issues or feature requests
2. **Create an issue** - If none exists, create one describing your proposed changes
3. **Wait for approval** - For major changes, wait for maintainer approval
4. **Assign yourself** - Assign the issue to yourself when starting work

### Branch Naming Convention
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/improvements

Examples:
- `feature/add-diabetes-prediction`
- `bugfix/fix-validation-error`
- `docs/update-api-documentation`

## 🎯 Types of Contributions

### 🧠 Machine Learning Improvements
- **Model Enhancements**: New algorithms, feature engineering, hyperparameter tuning
- **Data Processing**: Better data validation, preprocessing, feature selection
- **Performance Optimization**: Model compression, inference speed improvements
- **Interpretability**: SHAP, LIME, or other explainability features

**Guidelines:**
- Include performance metrics comparison
- Document new dependencies
- Provide validation on test data
- Include feature importance analysis

### 🎨 Frontend Improvements
- **UI/UX Enhancements**: Better user interface, accessibility improvements
- **Responsive Design**: Mobile optimization, cross-browser compatibility
- **Interactive Features**: New visualizations, real-time feedback
- **Performance**: Faster loading, optimized assets

**Guidelines:**
- Test on multiple browsers
- Ensure mobile responsiveness
- Follow accessibility standards (WCAG 2.1)
- Optimize for performance

### 🔧 Backend Improvements
- **API Enhancements**: New endpoints, better error handling
- **Security**: Input validation, authentication, authorization
- **Performance**: Caching, database optimization, async processing
- **Monitoring**: Logging, metrics, health checks

**Guidelines:**
- Follow REST API best practices
- Include comprehensive error handling
- Add appropriate tests
- Document API changes

### 📚 Documentation
- **API Documentation**: OpenAPI/Swagger specs
- **User Guides**: Setup instructions, usage examples
- **Developer Docs**: Architecture, deployment guides
- **Medical Context**: Clinical background, interpretation guides

**Guidelines:**
- Use clear, concise language
- Include code examples
- Keep documentation up-to-date
- Add diagrams where helpful

### 🧪 Testing
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end functionality
- **Performance Tests**: Load testing, benchmarks
- **Security Tests**: Vulnerability assessments

**Guidelines:**
- Maintain >80% code coverage
- Include edge cases
- Test error conditions
- Document test scenarios

## 🔄 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow coding standards
- Write/update tests
- Update documentation
- Test locally

### 3. Commit Changes
```bash
# Use conventional commit format
git add .
git commit -m "feat: add new risk factor analysis

- Implement SHAP value calculation
- Add interactive feature importance chart
- Update API documentation
- Add unit tests for new functionality

Closes #123"
```

### 4. Push and Create PR
```bash
git push origin feature/your-feature-name
# Create pull request on GitHub
```

## 📐 Coding Standards

### Python Code Style
- **PEP 8 compliance** - Use `black` for formatting
- **Type hints** - Add type annotations for functions
- **Docstrings** - Follow Google or NumPy style
- **Error handling** - Use specific exceptions with clear messages

Example:
```python
def calculate_risk_score(
    age: int, 
    cholesterol: float, 
    blood_pressure: int
) -> Dict[str, Union[float, str]]:
    """Calculate heart disease risk score.
    
    Args:
        age: Patient age in years (must be 18-120)
        cholesterol: Total cholesterol in mg/dl
        blood_pressure: Systolic blood pressure in mmHg
    
    Returns:
        Dictionary containing risk score and category
        
    Raises:
        ValueError: If input values are out of valid range
    """
    if not 18 <= age <= 120:
        raise ValueError(f"Age must be between 18-120, got {age}")
    
    # Implementation here
    return {"score": 0.75, "category": "moderate"}
```

### JavaScript/HTML Standards
- **ES6+ syntax** - Use modern JavaScript features
- **Semantic HTML** - Proper HTML5 elements
- **Accessibility** - ARIA labels, keyboard navigation
- **Performance** - Minimize DOM manipulation, use efficient selectors

### CSS Standards
- **BEM methodology** - Block Element Modifier naming
- **Mobile-first** - Responsive design approach
- **CSS custom properties** - Use CSS variables for theming
- **Performance** - Minimize reflows, use transforms for animations

## 🧪 Testing Guidelines

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest test_enhanced_api.py

# Run API tests
python test_enhanced_api.py
```

### Writing Tests
- **Test file naming**: `test_*.py` or `*_test.py`
- **Test function naming**: `test_descriptive_name`
- **Arrange-Act-Assert** pattern
- **Mock external dependencies**

Example:
```python
def test_risk_calculation_high_risk():
    """Test risk calculation for high-risk patient."""
    # Arrange
    patient_data = {
        "age": 65,
        "cholesterol": 300,
        "blood_pressure": 180
    }
    
    # Act
    result = calculate_risk_score(**patient_data)
    
    # Assert
    assert result["category"] == "high"
    assert result["score"] > 0.7
```

### Test Coverage Requirements
- **Minimum 80% coverage** for new code
- **Critical paths 100%** - Authentication, data validation, model inference
- **Edge cases** - Invalid inputs, boundary conditions
- **Error scenarios** - Network failures, file not found, etc.

## 📖 Documentation Standards

### Code Documentation
- **Function docstrings** - Parameters, return values, exceptions
- **Class docstrings** - Purpose, usage examples
- **Module docstrings** - High-level overview
- **Inline comments** - Complex logic explanation

### API Documentation
- **OpenAPI/Swagger** - For REST endpoints
- **Request/response examples** - Include sample data
- **Error codes** - Document all possible errors
- **Rate limiting** - Usage restrictions

### User Documentation
- **Setup guides** - Step-by-step installation
- **Usage examples** - Common workflows
- **Troubleshooting** - Common issues and solutions
- **FAQ** - Frequently asked questions

## 📤 Submitting Changes

### Pull Request Checklist
- [ ] **Branch is up-to-date** with main branch
- [ ] **Tests pass** locally and in CI
- [ ] **Code follows** style guidelines
- [ ] **Documentation updated** for changes
- [ ] **Commit messages** follow conventional format
- [ ] **No merge conflicts** exist
- [ ] **PR description** explains changes clearly

### PR Description Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## 👀 Review Process

### For Contributors
1. **Create detailed PR** - Use the template above
2. **Respond to feedback** - Address reviewer comments promptly
3. **Update documentation** - Keep docs in sync with code changes
4. **Be patient** - Reviews may take time for thorough evaluation

### Review Criteria
- **Functionality** - Does it work as intended?
- **Code quality** - Is it readable, maintainable, efficient?
- **Testing** - Are there adequate tests?
- **Documentation** - Are changes documented?
- **Security** - Are there any security implications?
- **Performance** - Does it impact system performance?

### Merge Requirements
- **2 approving reviews** for significant changes
- **1 approving review** for minor changes/bug fixes
- **All CI checks passing**
- **No merge conflicts**
- **Up-to-date with main branch**

## 🏷️ Release Process

### Versioning
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR** - Incompatible API changes
- **MINOR** - New functionality (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

### Release Checklist
- [ ] Update version numbers
- [ ] Update CHANGELOG.md
- [ ] Tag release in Git
- [ ] Update documentation
- [ ] Announce release

## 🌟 Recognition

### Contributors
All contributors will be recognized in:
- **README.md** - Contributors section
- **CHANGELOG.md** - Release notes
- **GitHub releases** - Release descriptions

### Types of Recognition
- **Code contributors** - Direct code contributions
- **Documentation contributors** - Documentation improvements
- **Issue reporters** - Bug reports and feature requests
- **Community contributors** - Helping others, code reviews

## 📞 Community

### Getting Help
- **GitHub Issues** - Technical questions, bug reports
- **Discussions** - General questions, feature discussions
- **Documentation** - Check existing docs first

### Communication Guidelines
- **Be respectful** - Treat everyone with kindness
- **Be specific** - Provide detailed information
- **Be patient** - Allow time for responses
- **Search first** - Check for existing issues/discussions

### Maintainer Contact
For urgent matters or security issues, contact the maintainers directly through GitHub.

**Project Maintainer**: [Your GitHub Username]
- **GitHub**: [@your-username](https://github.com/your-username)
- **Issues**: Use GitHub Issues for technical questions
- **Discussions**: Use GitHub Discussions for general questions

## 🗺️ Project Roadmap

### Current Priorities (Help Wanted!)
1. **Model Improvements**
   - Integration with real UCI Heart Disease Dataset
   - Implementation of ensemble methods (Random Forest + XGBoost)
   - SHAP/LIME explainability features

2. **Frontend Enhancements**
   - Mobile app development (React Native/Flutter)
   - Advanced data visualizations
   - Multi-language support

3. **Backend Features**
   - User authentication and profiles
   - Prediction history tracking
   - RESTful API v2 with GraphQL

4. **Infrastructure**
   - Docker containerization
   - CI/CD pipeline setup
   - Cloud deployment guides

### Future Goals
- Integration with Electronic Health Records (EHR) systems
- Real-time monitoring dashboard for healthcare providers
- Multi-disease prediction capabilities
- Clinical validation studies

## 🎉 Thank You!

Thank you for contributing to the Heart Disease Prediction System! Your contributions help improve healthcare technology and make medical prediction tools more accessible and accurate.

Every contribution, no matter how small, makes a difference. Whether you're fixing a typo, adding a feature, or improving documentation, you're helping create better tools for healthcare professionals and researchers worldwide.

---

**Remember**: This is an educational project. Always consult with medical professionals for actual healthcare decisions. Our goal is to advance medical AI research and education while maintaining the highest standards of code quality and medical ethics.

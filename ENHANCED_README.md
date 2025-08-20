# Enhanced Heart Disease Prediction System

A comprehensive machine learning-powered web application for heart disease risk assessment with advanced features, improved accuracy, and professional-grade visualization.

## 🎯 **Major Improvements**

### ✨ **Enhanced Features**
- **13 Clinical Parameters**: Comprehensive analysis using all standard heart disease indicators
- **Advanced ML Models**: Logistic Regression and Random Forest with hyperparameter tuning
- **Feature Importance Analysis**: Shows which factors most influence risk assessment
- **Professional UI**: Modern, responsive design with real-time validation
- **Interactive Visualizations**: Risk gauges, charts, and feature impact analysis
- **Comprehensive API**: RESTful endpoints with detailed error handling
- **Input Validation**: Real-time field validation with helpful guidance

### 🧠 **Model Improvements**
- **Expanded Dataset**: 500 synthetic patients with realistic correlations
- **Proper Validation**: Train/test split with cross-validation
- **Hyperparameter Tuning**: GridSearchCV for optimal performance
- **Model Comparison**: Automatic selection of best-performing algorithm
- **Feature Scaling**: StandardScaler for improved model performance
- **Performance Metrics**: AUC, accuracy, and comprehensive evaluation

### 🎨 **UI/UX Enhancements**
- **Progressive Form**: Organized into logical sections with visual feedback
- **Real-time Validation**: Instant field validation with error messages
- **Enhanced Results**: Risk categorization with detailed explanations
- **Interactive Charts**: Plotly-powered visualizations
- **Mobile Responsive**: Optimized for all device sizes
- **Accessibility**: ARIA labels and keyboard navigation support

## 📊 **Clinical Parameters**

| Parameter | Description | Range | Impact |
|-----------|-------------|-------|--------|
| **Age** | Patient age in years | 1-120 | High |
| **Sex** | Biological sex (0: Female, 1: Male) | 0-1 | Moderate |
| **Chest Pain Type** | 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic | 0-3 | High |
| **Resting BP** | Resting blood pressure (mmHg) | 80-250 | Moderate |
| **Cholesterol** | Serum cholesterol (mg/dl) | 100-600 | Moderate |
| **Fasting Blood Sugar** | >120 mg/dl (0: No, 1: Yes) | 0-1 | Low |
| **Resting ECG** | 0: Normal, 1: ST-T abnormality, 2: LV hypertrophy | 0-2 | Moderate |
| **Max Heart Rate** | Maximum heart rate achieved (bpm) | 50-220 | High |
| **Exercise Angina** | Exercise induced angina (0: No, 1: Yes) | 0-1 | High |
| **ST Depression** | ST depression induced by exercise | 0-10 | High |
| **ST Slope** | 0: Upsloping, 1: Flat, 2: Downsloping | 0-2 | Moderate |
| **Major Vessels** | Number of major vessels colored by fluoroscopy | 0-3 | High |
| **Thalassemia** | 0: Normal, 1: Fixed defect, 2: Reversible defect | 0-2 | High |

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
```powershell
# Clone or download the repository
cd heart_Disease_Prediction-main

# Create virtual environment
python -m venv .venv
./.venv/Scripts/Activate.ps1

# Install dependencies
pip install -r enhanced_requirements.txt

# Train the enhanced model
cd "heart disease predictor"
python enhanced_retrain_model.py

# Start the application
python enhanced_app.py
```

### Access the Application
- **Web Interface**: http://127.0.0.1:5000
- **API Documentation**: http://127.0.0.1:5000/model-info
- **Health Check**: http://127.0.0.1:5000/health

## 🔧 **API Endpoints**

### Core Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Enhanced input form |
| `/predict` | POST | Web form prediction |
| `/result` | GET | Detailed results page |
| `/api/predict` | POST | JSON API prediction |
| `/health` | GET | Service health check |
| `/model-info` | GET | Model information |

### API Usage Example
```python
import requests

# Prepare patient data
patient_data = {
    "age": 54,
    "sex": 1,
    "cp": 1,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 1,
    "thal": 1
}

# Make prediction
response = requests.post("http://127.0.0.1:5000/api/predict", json=patient_data)
result = response.json()

print(f"Risk Level: {result['risk_info']['level']}")
print(f"Probability: {result['probability']:.1%}")
```

## 📈 **Model Performance**

### Current Performance Metrics
- **Algorithm**: Logistic Regression (selected via cross-validation)
- **Test Accuracy**: 85.0%
- **AUC Score**: 0.904
- **Features**: 13 clinical parameters
- **Dataset Size**: 500 synthetic patients with realistic correlations

### Model Validation
- Train/test split (80/20)
- 5-fold cross-validation
- Hyperparameter tuning with GridSearchCV
- Feature scaling with StandardScaler

## 🎨 **Screenshots & Features**

### Enhanced Input Form
- **Organized Sections**: Demographics, symptoms, vital signs, lab results, cardiac tests
- **Real-time Validation**: Instant feedback with progress tracking
- **Helpful Guidance**: Tooltips, ranges, and explanations for each field

### Detailed Results
- **Risk Assessment**: Clear categorization (Low/Moderate/High)
- **Interactive Visualizations**: Risk gauge and feature importance charts
- **Actionable Recommendations**: Specific next steps based on risk level
- **Feature Analysis**: Shows which factors most influenced the assessment

## 🔒 **Security & Privacy**

### Data Protection
- Input validation and sanitization
- Session-based data handling
- No permanent data storage
- Secure error handling

### Medical Compliance
- Clear medical disclaimers
- Educational purpose statements
- Professional consultation recommendations

## 🧪 **Testing**

### Run Tests
```powershell
# Test API endpoints
python test_enhanced_api.py

# Manual testing
# Use the web interface at http://127.0.0.1:5000
```

### Test Cases Included
- Low risk patient scenarios
- High risk patient scenarios
- Edge case validations
- API error handling

## 🔧 **Development**

### Project Structure
```
heart disease predictor/
├── enhanced_app.py              # Main Flask application
├── enhanced_retrain_model.py    # Model training script
├── test_enhanced_api.py         # API testing script
├── templates/
│   ├── enhanced_index.html      # Enhanced input form
│   └── enhanced_result.html     # Detailed results page
└── enhanced_heart_disease_model.pkl  # Trained model
```

### Adding New Features
1. **New Clinical Parameters**: Modify `enhanced_retrain_model.py`
2. **UI Enhancements**: Update template files
3. **API Extensions**: Extend `enhanced_app.py`
4. **Model Improvements**: Implement in training script

## 📝 **Configuration**

### Environment Variables
```bash
FLASK_SECRET_KEY=your-secret-key-here
FLASK_DEBUG=1                 # Development only
PORT=5000                     # Server port
```

### Model Configuration
- Modify `VALIDATION_RANGES` in `enhanced_app.py` for input validation
- Adjust hyperparameter grids in `enhanced_retrain_model.py`
- Update feature importance calculations for new models

## 🚀 **Deployment**

### Cloud Deployment
The application is ready for cloud deployment on:
- **Heroku**: Include `Procfile`
- **Railway**: Auto-deployment ready
- **Render**: Built-in support
- **AWS/Azure**: Container deployment

### Production Considerations
- Use `gunicorn` for production WSGI server
- Set up proper environment variables
- Configure logging and monitoring
- Implement rate limiting
- Add SSL/TLS certificates

## 📊 **Future Enhancements**

### Planned Features
- [ ] User authentication and profiles
- [ ] Prediction history tracking
- [ ] Database integration
- [ ] Advanced model interpretability (SHAP/LIME)
- [ ] Multi-language support
- [ ] Integration with medical systems (HL7 FHIR)
- [ ] Mobile app companion
- [ ] Continuous model retraining

### Model Improvements
- [ ] Ensemble methods
- [ ] Deep learning models
- [ ] External dataset integration
- [ ] Real-time model monitoring
- [ ] A/B testing framework

## ⚠️ **Important Disclaimers**

- **Educational Purpose Only**: This tool is designed for educational and research purposes
- **Not Medical Advice**: Should never replace professional medical consultation
- **Research Tool**: Suitable for learning ML concepts and heart disease risk factors
- **Data Synthetic**: Training data is synthetically generated for demonstration

## 🤝 **Contributing**

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new features
5. Submit a pull request

### Code Standards
- Python PEP 8 compliance
- Comprehensive error handling
- Clear documentation
- Unit test coverage
- Security best practices

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- UCI Heart Disease Dataset (inspiration)
- Flask and scikit-learn communities
- Medical professionals for domain knowledge
- Open source contributors

---

**Remember**: This enhanced system demonstrates significant improvements in accuracy, usability, and features while maintaining the educational focus of the original project.

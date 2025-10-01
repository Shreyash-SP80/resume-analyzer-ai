# Resume Analyzer AI 

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org) 
[![UV](https://img.shields.io/badge/UV-Package%20Manager-orange)](https://github.com/astral-sh/uv)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)](https://streamlit.io)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%20AI-lightgrey)](https://ai.google.dev/)

A sophisticated AI-powered resume analysis tool that provides comprehensive feedback on your resume's content, structure, and effectiveness. Built with Python UV and Google Gemini AI.
 
![Resume Analyzer Demo](https://via.placeholder.com/800x400.png?text=Resume+Analyzer+AI+Screenshot)   

## Features

### - AI-Powered Analysis  
- **Smart Content Review**: In-depth analysis using Google Gemini AI
- **Job Role Tailoring**: Specific recommendations based on target positions
- **Actionable Insights**: Practical suggestions for improvement

### - Structural Analysis
- **Comprehensive Metrics**: Word count, readability score, section coverage
- **Quality Assessment**: Professional evaluation of resume structure
- **Industry Benchmarks**: Comparison against optimal resume standards

### - Professional Feedback
- **Content Quality**: Evaluation of impact and effectiveness
- **Skills Presentation**: Assessment of technical and soft skills display
- **Experience Optimization**: Suggestions for better experience descriptions
- **ATS Optimization**: Tips for applicant tracking system compatibility

### - User-Friendly Interface
- **Multi-Format Support**: PDF and TXT file uploads
- **Analysis History**: Track previous analyses
- **Export Results**: Download comprehensive reports
- **Real-time Processing**: Instant feedback and suggestions

## Quick Start

### Prerequisites

- Python 3.8 or higher
- UV package manager
- Google Gemini API key

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/resume-analyzer-ai.git
cd resume-analyzer-ai
```
2. Set up virtual environment with UV
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install dependencies
```bash
uv sync
```
4. Configure environment variables
```bash
cp .env.example .env
```
Edit .env file and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
5. Run the application
```bash
uv run streamlit run src/main.py
```
The application will open in your browser at http://localhost:8501

## 📖 Usage Guide
### 1. Upload Your Resume
- Supported formats: PDF, TXT
- Maximum file size: 10MB
- Ensure text is selectable in PDF files

### 2. Specify Target Role (Optional)
- Enter the job title you're targeting
- Get role-specific recommendations
- Improve keyword optimization

### 3. Configure Analysis Options
- Content Quality & Impact
- Skills Presentation
- Experience Descriptions
- Achievements & Metrics
- Structure & Readability
- ATS Optimization
- Industry Alignment

### 4. Review Analysis Results
- Structural Analysis: Quantitative metrics and benchmarks
- AI Analysis: Qualitative feedback and suggestions
- Quality Assessment: Professional recommendations
- Export Options: Download comprehensive report

## 🔑 API Key Setup
- Getting Google Gemini API Key
- Visit Google AI Studio
- Sign in with your Google account
- Create a new API key
- Copy the key to your .env file

## Environment Configuration
Create a .env file in the project root:
  ```bash
     GOOGLE_API_KEY=your_actual_gemini_api_key
  ```

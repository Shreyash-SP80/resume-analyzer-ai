from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import PyPDF2
import os
import io
import re
from datetime import datetime
from typing import Dict, List 

class ResumeAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=api_key
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF with enhanced error handling."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def extract_text_from_file(self, uploaded_file) -> str:
        """Extract text from various file types."""
        file_type = uploaded_file.type
        
        if file_type == "application/pdf":
            return self.extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
        elif file_type == "text/plain":
            return uploaded_file.getvalue().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def analyze_resume_structure(self, text: str) -> Dict:
        """Perform basic structural analysis of the resume."""
        lines = text.split('\n')
        word_count = len(text.split())
        char_count = len(text)
        line_count = len(lines)
        
        sections = self._detect_sections(text)
        
        action_verbs = self._count_action_verbs(text)
        
        contact_info = self._detect_contact_info(text)
        
        readability_score = self._calculate_readability(text)
    
        quantifiable_achievements = self._count_quantifiable_achievements(text)
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'line_count': line_count,
            'sections_found': sections,
            'action_verbs_count': action_verbs,
            'contact_info_present': contact_info,
            'estimated_pages': max(1, word_count // 500),
            'readability_score': readability_score,
            'quantifiable_achievements': quantifiable_achievements,
            'section_coverage_score': self._calculate_section_coverage(sections)
        }
    
    def _detect_sections(self, text: str) -> List[str]:
        """Detect common resume sections."""
        section_keywords = {
            'experience': ['experience', 'work history', 'employment', 'career', 'professional experience'],
            'education': ['education', 'academic', 'degree', 'university', 'college', 'qualifications'],
            'skills': ['skills', 'technical skills', 'competencies', 'abilities', 'proficiencies'],
            'summary': ['summary', 'objective', 'profile', 'professional summary'],
            'projects': ['projects', 'portfolio', 'achievements', 'key projects'],
            'certifications': ['certifications', 'certificates', 'training', 'licenses'],
            'awards': ['awards', 'honors', 'recognitions', 'achievements']
        }
        
        found_sections = []
        text_lower = text.lower()
        
        for section, keywords in section_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_sections.append(section)
        
        return found_sections
    
    def _count_action_verbs(self, text: str) -> int:
        """Count action verbs commonly used in resumes."""
        action_verbs = [
            'managed', 'led', 'developed', 'implemented', 'created', 'designed',
            'analyzed', 'improved', 'increased', 'reduced', 'optimized', 'built',
            'coordinated', 'organized', 'planned', 'executed', 'achieved', 'won',
            'delivered', 'transformed', 'initiated', 'spearheaded', 'directed',
            'supervised', 'mentored', 'trained', 'resolved', 'streamlined'
        ]
        
        words = text.lower().split()
        return sum(1 for word in words if word in action_verbs)
    
    def _detect_contact_info(self, text: str) -> bool:
        """Check if basic contact information is present."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        has_email = bool(re.search(email_pattern, text))
        has_phone = bool(re.search(phone_pattern, text))
        
        return has_email and has_phone
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate a simple readability score."""
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        score = 100 - (avg_sentence_length * 2 + avg_word_length * 3)
        return max(0, min(100, score))
    
    def _count_quantifiable_achievements(self, text: str) -> int:
        """Count quantifiable achievements (numbers, percentages, etc.)."""
        patterns = [
            r'\d+%',  # percentages
            r'\$\d+',  # dollar amounts
            r'\d+\+',  # numbers with plus
            r'increased by \d+',  # increase phrases
            r'reduced by \d+',  # reduction phrases
            r'saved \$\d+',  # savings
            r'improved by \d+',  # improvements
        ]
        
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text.lower()))
        
        return count
    
    def _calculate_section_coverage(self, sections: List[str]) -> int:
        """Calculate section coverage score (0-100)."""
        essential_sections = ['experience', 'education', 'skills', 'contact']
        found_essential = sum(1 for section in essential_sections if section in sections)
        return int((found_essential / len(essential_sections)) * 100)
    
    def get_ai_analysis(self, text: str, job_role: str, focus_areas: List[str]) -> str:
        """Get AI-powered analysis of the resume."""
        focus_text = "\n".join([f"- {area.replace('_', ' ').title()}" for area in focus_areas])
        
        prompt = f"""As an expert resume consultant, provide a comprehensive analysis of this resume.

            Target Role: {job_role if job_role else "General position"}
            Focus Areas:
            {focus_text}
            
            Resume Content:
            {text[:12000]}
            
            Please provide your analysis in this structured format:
            
            EXECUTIVE SUMMARY
                Brief overall assessment
                
            STRENGTHS HIGHLIGHT
            - Key positive aspects
            - Notable achievements
            
            AREAS FOR IMPROVEMENT
            - Specific, actionable suggestions
            - Content enhancements
            - Formatting recommendations
            
            TAILORED RECOMMENDATIONS FOR {job_role if job_role else "TARGET ROLE"}
            - Role-specific optimizations
            - Keyword suggestions
            - Industry insights
            
            QUICK ACTION ITEMS
            - Top 3 immediate improvements
            
            Be constructive, professional, and provide specific examples where possible."""

        try:
            message = HumanMessage(content=prompt)
            response = self.model.invoke([message])
            return response.content
        except Exception as e:
            return f"Analysis temporarily unavailable. Please try again later. Error: {str(e)}"

def init_session_state():
    """Initialize session state variables."""
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'current_analysis' not in st.session_state:
        st.session_state.current_analysis = None

def save_analysis_history(analysis_data: Dict):
    """Save analysis to session history."""
    st.session_state.analysis_history.append(analysis_data)
    if len(st.session_state.analysis_history) > 5:
        st.session_state.analysis_history = st.session_state.analysis_history[-5:]

def display_structural_analysis(structural_data: Dict):
    """Display the structural analysis results in a professional format."""
    st.subheader("Structural Analysis")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", structural_data['word_count'])
        st.metric("Estimated Pages", structural_data['estimated_pages'])
    
    with col2:
        st.metric("Lines of Text", structural_data['line_count'])
        st.metric("Action Verbs", structural_data['action_verbs_count'])
    
    with col3:
        contact_status = "Complete" if structural_data['contact_info_present'] else "Missing"
        st.metric("Contact Information", contact_status)
        st.metric("Quantifiable Achievements", structural_data['quantifiable_achievements'])
    
    with col4:
        st.metric("Readability Score", f"{structural_data['readability_score']:.1f}/100")
        st.metric("Section Coverage", f"{structural_data['section_coverage_score']}%")
    
    # Quality assessment section
    st.markdown("---")
    st.subheader("Quality Assessment")
    
    # Content length assessment
    word_count = structural_data['word_count']
    if word_count < 300:
        length_status = "Insufficient"
        length_color = "red"
        length_recommendation = "Consider adding more detailed experiences and accomplishments"
    elif word_count < 600:
        length_status = "Optimal"
        length_color = "green"
        length_recommendation = "Good length for most professional resumes"
    elif word_count < 1000:
        length_status = "Comprehensive"
        length_color = "orange"
        length_recommendation = "Consider streamlining for conciseness"
    else:
        length_status = "Too Long"
        length_color = "red"
        length_recommendation = "Strongly recommend condensing to 1-2 pages"
    
    # Section coverage assessment
    section_count = len(structural_data['sections_found'])
    if section_count >= 5:
        section_status = "Comprehensive"
        section_color = "green"
    elif section_count >= 3:
        section_status = "Adequate"
        section_color = "orange"
    else:
        section_status = "Limited"
        section_color = "red"
    
    # Action verbs assessment
    action_verbs = structural_data['action_verbs_count']
    if action_verbs >= 15:
        verb_status = "Strong"
        verb_color = "green"
    elif action_verbs >= 8:
        verb_status = "Moderate"
        verb_color = "orange"
    else:
        verb_status = "Needs Improvement"
        verb_color = "red"
    
    # Display quality assessments in columns
    qual_col1, qual_col2, qual_col3 = st.columns(3)
    
    with qual_col1:
        st.markdown(f"**Content Length:** <span style='color:{length_color}'>{length_status}</span>", 
                   unsafe_allow_html=True)
        st.caption(f"{word_count} words - {length_recommendation}")
    
    with qual_col2:
        st.markdown(f"**Section Coverage:** <span style='color:{section_color}'>{section_status}</span>", 
                   unsafe_allow_html=True)
        st.caption(f"{section_count} sections detected")
    
    with qual_col3:
        st.markdown(f"**Action Language:** <span style='color:{verb_color}'>{verb_status}</span>", 
                   unsafe_allow_html=True)
        st.caption(f"{action_verbs} action verbs used")
    
    # Detected sections with status indicators
    st.markdown("---")
    st.subheader("Section Analysis")
    
    essential_sections = {
        'experience': 'Professional Experience',
        'education': 'Education', 
        'skills': 'Skills',
        'summary': 'Professional Summary'
    }
    
    optional_sections = {
        'projects': 'Projects',
        'certifications': 'Certifications', 
        'awards': 'Awards'
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Essential Sections**")
        for section_key, section_name in essential_sections.items():
            if section_key in structural_data['sections_found']:
                st.success(f"✓ {section_name}")
            else:
                st.error(f"✗ {section_name} (Recommended)")
    
    with col2:
        st.write("**Optional Sections**")
        for section_key, section_name in optional_sections.items():
            if section_key in structural_data['sections_found']:
                st.info(f"✓ {section_name}")
            else:
                st.write(f"- {section_name}")
    
    # Detailed metrics expander
    with st.expander("View Detailed Metrics"):
        st.write(f"**Character Count:** {structural_data['char_count']}")
        st.write(f"**Readability Score:** {structural_data['readability_score']:.1f}/100")
        st.write(f"**Quantifiable Achievements:** {structural_data['quantifiable_achievements']}")
        st.write(f"**Section Coverage Score:** {structural_data['section_coverage_score']}%")
        
        # Industry benchmarks
        st.write("**Industry Benchmarks:**")
        benchmarks = {
            "Ideal Word Count": "400-800 words",
            "Recommended Pages": "1-2 pages", 
            "Action Verbs": "15-30 verbs",
            "Essential Sections": "3-4 sections minimum",
            "Quantifiable Results": "3-5 achievements"
        }
        
        for benchmark, ideal in benchmarks.items():
            st.write(f"- {benchmark}: {ideal}")

def display_comparison_tips():
    """Display helpful comparison tips."""
    st.subheader("Industry Benchmarks")
    
    benchmarks = {
        "Content Length": "400-800 words provides optimal detail without overwhelming recruiters",
        "Page Count": "1-2 pages maximum for most professional levels",
        "Action Verbs": "15-30 strong action verbs demonstrate proactive accomplishments",
        "Section Coverage": "Include Experience, Education, Skills, and Summary as essential sections",
        "Quantifiable Achievements": "3-5 measurable results show concrete impact",
        "Contact Information": "Email and phone number are essential for recruiter contact"
    }
    
    for metric, description in benchmarks.items():
        with st.container():
            st.write(f"**{metric}**")
            st.caption(description)
            st.write("")

def main():
    # Page configuration
    st.set_page_config(
        page_title="Resume Analyzer - Professional Resume Analysis",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #333333;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.5rem;
        color: #444444;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 500;
        border-bottom: 2px solid #666666;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #888888;
        margin: 0.5rem 0;
    }
    .analysis-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #dddddd;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Professional Resume Analysis Tool")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Quick Start")
        st.info("Upload your resume to begin analysis")
        
        st.markdown("### Analysis History")
        if st.session_state.analysis_history:
            for i, analysis in enumerate(reversed(st.session_state.analysis_history)):
                with st.expander(f"Analysis {len(st.session_state.analysis_history)-i}"):
                    st.write(f"Role: {analysis.get('job_role', 'General')}")
                    st.write(f"Date: {analysis.get('timestamp', 'Unknown')}")
                    st.write(f"File: {analysis.get('file_name', 'Unknown')}")
        else:
            st.write("No analysis history available")
        
        st.markdown("### Professional Tips")
        tips = [
            "Quantify achievements with specific metrics and numbers",
            "Use industry-specific keywords from target job descriptions",
            "Maintain clear section headers for easy scanning",
            "Tailor content specifically for each application",
            "Proofread thoroughly for spelling and grammar errors",
            "Use consistent formatting and professional fonts"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload Your Resume",
        type=["pdf", "txt"],
        help="Supported formats: PDF, Text files"
    )
    
    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Job role input
        job_role = st.text_input(
            "Target Job Role (Optional)",
            placeholder="Example: Senior Software Engineer, Data Analyst, Marketing Manager"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        
        analysis_options = {
            "content_quality": "Content Quality and Impact",
            "skills_presentation": "Skills Presentation",
            "experience_descriptions": "Experience Descriptions", 
            "achievements_quantification": "Achievements and Metrics",
            "structure_readability": "Structure and Readability",
            "ats_optimization": "ATS Optimization",
            "industry_alignment": "Industry Alignment"
        }
        
        selected_options = []
        cols = st.columns(2)
        for i, (key, value) in enumerate(analysis_options.items()):
            with cols[i % 2]:
                if st.checkbox(value, value=True, key=key):
                    selected_options.append(key)
        
        # Analysis button
        if st.button("Start Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing your resume. This may take a moment..."):
                try:
                    # Load API key
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if not api_key:
                        st.error("API key not configured. Please check your .env file.")
                        return
                    
                    analyzer = ResumeAnalyzer(api_key)
                    
                    # Extract text
                    file_content = analyzer.extract_text_from_file(uploaded_file)
                    
                    if not file_content.strip():
                        st.error("The uploaded file appears to be empty or unreadable.")
                        return
                    
                    st.info(f"Resume content extracted: {len(file_content)} characters, {len(file_content.split())} words")
                    
                    structural_data = analyzer.analyze_resume_structure(file_content)
                    
                    # Get AI analysis
                    ai_analysis = analyzer.get_ai_analysis(
                        file_content, 
                        job_role, 
                        selected_options
                    )
                    
                    # Store analysis results
                    analysis_data = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'job_role': job_role,
                        'file_name': uploaded_file.name,
                        'structural_data': structural_data,
                        'ai_analysis': ai_analysis,
                        'focus_areas': selected_options
                    }
                    
                    st.session_state.current_analysis = analysis_data
                    save_analysis_history(analysis_data)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.info("Please try again with a different file or check the file format.")
    
    # Display analysis results
    if st.session_state.current_analysis:
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        analysis_data = st.session_state.current_analysis
        
        # Structural analysis
        display_structural_analysis(analysis_data['structural_data'])
        
        # AI analysis
        st.subheader("AI Analysis")
        st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
        st.markdown(analysis_data['ai_analysis'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Industry benchmarks
        display_comparison_tips()
        
        # Export options
        st.subheader("Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # Create downloadable report
            analysis_text = f"""
                RESUME ANALYSIS REPORT
                Generated: {analysis_data['timestamp']}
                Target Role: {analysis_data['job_role']}
                File: {analysis_data['file_name']}
                
                STRUCTURAL ANALYSIS RESULTS:
                - Word Count: {analysis_data['structural_data']['word_count']}
                - Estimated Pages: {analysis_data['structural_data']['estimated_pages']}
                - Lines of Text: {analysis_data['structural_data']['line_count']}
                - Sections Detected: {', '.join(analysis_data['structural_data']['sections_found'])}
                - Action Verbs: {analysis_data['structural_data']['action_verbs_count']}
                - Contact Information: {'Complete' if analysis_data['structural_data']['contact_info_present'] else 'Missing'}
                - Readability Score: {analysis_data['structural_data']['readability_score']:.1f}/100
                - Quantifiable Achievements: {analysis_data['structural_data']['quantifiable_achievements']}
                - Section Coverage: {analysis_data['structural_data']['section_coverage_score']}%
                
                AI ANALYSIS:
                {analysis_data['ai_analysis']}
                
                Analysis Focus Areas: {', '.join(analysis_data['focus_areas'])}
            """

            st.download_button(
                label="Download Analysis Report",
                data=analysis_text,
                file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if st.button("Analyze Another Resume", use_container_width=True):
                st.session_state.current_analysis = None
                st.rerun()

if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
        st.stop()
    
    main()


# from langchain_core.messages import HumanMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langgraph.prebuilt import create_react_agent
# from dotenv import load_dotenv
# import streamlit as st
# import PyPDF2
# import os
# import sys
# import io

# def load_api_key():
#     """Load Gemini API key from .env file."""
#     load_dotenv()
#     api_key = os.getenv("GOOGLE_API_KEY")

#     if not api_key:
#         st.error("❌ Error: GOOGLE_API_KEY not found in .env file")
#         st.stop()

#     return api_key

# def build_agent(api_key: str):
#     """Initialize the Gemini chat model and agent executor."""
#     model = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash",  
#         temperature=0,
#         google_api_key=api_key
#     )

#     tools = []  
#     return create_react_agent(model, tools)

# def extract_text_from_pdf(pdf_file):
#     """Extract text from PDF file."""
#     try:
#         pdf_reader = PyPDF2.PdfReader(pdf_file)
#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text() + "\n"
#         return text
#     except Exception as e:
#         st.error(f"Error reading PDF: {str(e)}")
#         return ""

# def extract_text_from_file(uploaded_file):
#     """Extract text from uploaded file (PDF or TXT)."""
#     try:
#         if uploaded_file.type == "application/pdf":
#             return extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
#         elif uploaded_file.type == "text/plain":
#             return uploaded_file.getvalue().decode("utf-8")
#         else:
#             st.error("Unsupported file type")
#             return ""
#     except Exception as e:
#         st.error(f"Error reading file: {str(e)}")
#         return ""

# def analyze_resume_with_gemini(file_content: str, job_role: str = "", api_key: str = ""):
#     """Analyze resume using Gemini API."""
#     try:
#         # Initialize Gemini model
#         model = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             temperature=0.7,
#             google_api_key=api_key
#         )
        
#         prompt = f"""Please analyze this resume and provide constructive feedback. 
#         Focus on the following aspects:
#         1. Content clarity and impact
#         2. Skills presentation
#         3. Experience descriptions
#         4. Achievements and quantifiable results
#         5. Overall structure and readability
#         {f"6. Specific improvements for {job_role} role" if job_role else "6. General improvements for job applications"}
        
#         Resume content:
#         {file_content[:15000]}  # Limit content to avoid token limits
        
#         Please provide your analysis in a clear, structured format with specific recommendations.
#         Be constructive and provide actionable advice."""
        
#         # Create message and get response
#         message = HumanMessage(content=prompt)
#         response = model.invoke([message])
        
#         return response.content
        
#     except Exception as e:
#         return f"Error analyzing resume: {str(e)}"

# # Streamlit UI
# def main():
#     st.set_page_config(
#         page_title="AI Resume Critiquer", 
#         page_icon="📃", 
#         layout="centered"
#     )
    
#     st.title("📃 AI Resume Critiquer")
#     st.markdown("Upload your resume and get AI-powered feedback tailored to your needs!")
    
#     # Load API key
#     api_key = load_api_key()
    
#     # File upload
#     uploaded_file = st.file_uploader(
#         "Upload your resume (PDF or TXT)", 
#         type=["pdf", "txt"],
#         help="Supported formats: PDF and Text files"
#     )
    
#     # Job role input
#     job_role = st.text_input(
#         "Enter the job role you're targeting (optional)",
#         placeholder="e.g., Software Engineer, Data Analyst, Marketing Manager"
#     )
    
#     # Analysis options
#     st.subheader("Analysis Focus")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         analyze_content = st.checkbox("Content & Impact", value=True)
#         analyze_skills = st.checkbox("Skills Presentation", value=True)
        
#     with col2:
#         analyze_experience = st.checkbox("Experience Descriptions", value=True)
#         analyze_structure = st.checkbox("Structure & Readability", value=True)
    
#     analyze_button = st.button("Analyze Resume", type="primary")
    
#     if analyze_button and uploaded_file:
#         with st.spinner("Analyzing your resume..."):
#             try:
#                 # Extract text from file
#                 file_content = extract_text_from_file(uploaded_file)
                
#                 if not file_content.strip():
#                     st.error("The uploaded file appears to be empty or couldn't be read.")
#                     st.stop()
                
#                 # Show file info
#                 st.info(f"📊 Resume length: {len(file_content)} characters")
                
#                 # Analyze with Gemini
#                 analysis_result = analyze_resume_with_gemini(file_content, job_role, api_key)
                
#                 # Display results
#                 st.markdown("## 📋 Analysis Results")
#                 st.markdown("---")
                
#                 st.success("Here's your personalized resume analysis:")
#                 st.markdown(analysis_result)
                
#                 # Tips section
#                 st.markdown("## 💡 Quick Tips")
#                 tips = """
#                 - **Quantify achievements** with numbers and metrics
#                 - **Use action verbs** to start bullet points
#                 - **Tailor your resume** to each specific job application
#                 - **Keep it concise** - aim for 1-2 pages maximum
#                 - **Proofread carefully** for spelling and grammar errors
#                 """
#                 st.markdown(tips)
                
#             except Exception as e:
#                 st.error(f"An error occurred during analysis: {str(e)}")
#                 st.info("Please try again with a different file or check your API key.")

#     elif analyze_button and not uploaded_file:
#         st.warning("Please upload a resume file first.")

#     # Sidebar with information
#     with st.sidebar:
#         st.header("ℹ️ How It Works")
#         st.markdown("""
#         1. **Upload** your resume (PDF or TXT)
#         2. **Specify** target job role (optional)
#         3. **Select** analysis focus areas
#         4. **Get** AI-powered feedback
        
#         **Features:**
#         - Content clarity assessment
#         - Skills presentation evaluation
#         - Experience description optimization
#         - Structure and formatting advice
#         """)
        
#         st.header("📝 Supported Formats")
#         st.markdown("""
#         - **PDF files** (.pdf)
#         - **Text files** (.txt)
#         - Maximum file size: 10MB
#         """)

# if __name__ == "__main__":
#     main()

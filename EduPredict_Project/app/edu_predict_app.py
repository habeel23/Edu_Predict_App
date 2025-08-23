import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="EduPredict - Academic Success Predictor", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)
# Custom CSS for modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { 
        background: linear-gradient(135deg, #e0f7fa 0%, #80deea 100%);
        font-family: 'Inter', sans-serif;
        height: 100vh; /* Full height */
        overflow: hidden; /* Prevent scrolling */
        
    }
    
    .main-header {
        background: linear-gradient(135deg, #00796b 0%, #004d40 100%);
        padding: 2rem 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00796b 0%, #004d40 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 1rem 2rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(0, 121, 107, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(0, 121, 107, 0.4);
        background: linear-gradient(135deg, #004d40 0%, #00796b 100%);
    }
    
    .prediction-card {
        background: rgba(255, 255, 255, 0.95); /* Slightly transparent */
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        margin: 1.5rem 0;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #333; /* Dark text for better visibility */
    }
    
    .success-card {
        background: rgba(102, 187, 106, 0.9); /* Slightly transparent */
        color: white;
    }
    
    .warning-card {
        background: rgba(255, 183, 77, 0.9); /* Slightly transparent */
        color: white;
    }
    
    .error-card {
        background: rgba(239, 83, 80, 0.9); /* Slightly transparent */
        color: white;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95); /* Slightly transparent */
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin: 1.2rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #333; /* Dark text for better visibility */
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .sidebar-content {
        background: linear-gradient(180deg, #004d40 0%, #00796b 100%);
        padding: 2rem 1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .input-card {
        background: rgba(255, 255, 255, 0.95); /* Slightly transparent */
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #333; /* Dark text for better visibility */
    }
    
    .tab-content {
        background: rgba(255, 255, 255, 0.8); /* Slightly transparent */
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #333; /* Dark text for better visibility */
    }
    
    .footer {
        background: linear-gradient(135deg, #00796b 0%, #004d40 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-top: 3rem;
    }
    
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }
    
    .login-form {
        background: rgba(255, 255, 255, 0.95); /* Slightly transparent */
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
        width: 100%;
        max-width: 450px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        color: #333; /* Dark text for better visibility */
    }
    
    .sidebar-user-card {
        background: rgba(102, 187, 106, 0.9); /* Slightly transparent */
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(67, 233, 123, 0.3);
    }
    
    .nav-item {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        color: #fff; /* White text for better visibility */
    }
    
    .nav-item:hover {
        background: rgba(0, 121, 107, 0.1);
    }
    
    .nav-item.active {
        background: linear-gradient(135deg, #00796b 0%, #004d40 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class='main-header'>
    <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🎓 EduPredict</h1>
    <p style='font-size: 1.3rem; margin: 0; opacity: 0.9;'>AI-Powered Academic Success Predictor</p>
    <p style='font-size: 1rem; margin: 1rem 0 0 0; opacity: 0.8;'>Predict • Analyze • Succeed</p>
</div>
""", unsafe_allow_html=True)

# Warning banner
st.markdown("""
<div style='background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);padding:1.2rem;border-radius:12px;
border:1px solid #ffd166;margin-bottom:2rem;box-shadow: 0 5px 15px rgba(255, 209, 102, 0.2);'>
    <h4 style='color:#856404;margin:0;display:flex;align-items:center;gap:10px;'>
        <span style='font-size:1.5rem;'>📢</span> 
        <span>Important: Predictions are advisory only. For academic support, consult your mentors.</span>
    </h4>
</div>
""", unsafe_allow_html=True)

# Authentication
auth_users = {
    "student": "student123",
    "teacher": "teach123",
    "counselor": "counsel123"
}

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Prediction"
if 'username' not in st.session_state:
    st.session_state.username = ""

# Sidebar - Only show when logged in
if st.session_state.logged_in:
    role = st.session_state.username.capitalize()
    
    # User profile card
    st.sidebar.markdown(f"""
    <div class='sidebar-user-card'>
        <div style='font-size: 3rem; margin-bottom: 0.5rem;'>👤</div>
        <h3 style='margin:0 0 0.5rem 0;'>{role}</h3>
        <p style='margin:0; opacity:0.9;'>Welcome back!</p>
        <small style='opacity:0.8;'>{datetime.now().strftime('%Y-%m-%d | %H:%M')}</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📊 Navigation")
    
    # Tab selection
    tabs = {
        "🎯 Prediction": "Prediction",
        "📈 Analytics": "Analytics",
        "📋 Reports": "Reports",
        "⚙️ Settings": "Settings"
    }
    
    for icon_label, tab_value in tabs.items():
        icon, label = icon_label.split(" ", 1)
        is_active = st.session_state.current_tab == tab_value
        if st.sidebar.button(f"{icon} {label}", use_container_width=True, key=f"nav_{tab_value}"):
            st.session_state.current_tab = tab_value
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Quick stats in sidebar
    st.sidebar.markdown("<div class='sidebar-nav'>", unsafe_allow_html=True)
    st.sidebar.markdown("### 📈 Quick Stats")
    
    try:
        df = pd.read_csv("data/academic_cleaned.csv")
        df["Grade"] = df.apply(lambda row: "Graduate" if row["Target_Graduate"] == 1 else 
                              "Enrolled" if row["Target_Enrolled"] == 1 else "Dropout", axis=1)
        
        total_students = df.shape[0]
        graduates = df[df['Grade'] == "Graduate"].shape[0]
        dropouts = df[df['Grade'] == "Dropout"].shape[0]
        avg_grade = round(df["Admission grade"].mean(), 2)
        
        st.sidebar.metric("Total Students", f"{total_students:,}")
        st.sidebar.metric("Graduation Rate", f"{(graduates/total_students*100):.1f}%")
        st.sidebar.metric("Dropout Rate", f"{(dropouts/total_students*100):.1f}%")
        st.sidebar.metric("Avg Admission Grade", avg_grade)
        
    except:
        st.sidebar.info("Data not available for statistics")
    
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True, key="sidebar_logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.current_tab = "Prediction"
        st.rerun()

# Centralized login form
if not st.session_state.logged_in:
    # Hide sidebar when not logged in
    st.markdown("""
    <style>
    .sidebar-content {
        display: none;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Fix login form sizing */
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 40vh;
        width: 100%;
        padding: 1rem 0;
    }
    
    .login-form {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.9) 100%);
        padding: 1rem;
        border-radius: 14px;
        box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 350px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        margin: 0 auto;
    }
    
    /* Ensure proper column spacing */
    .stColumn {
        padding: 0 0.5rem;
    }
    
    /* Reduce spacing in form elements */
    .stTextInput, .stButton {
        margin-bottom: 0.8rem;
    }
    
    /* Reduce expander padding */
    .streamlit-expanderHeader {
        padding: 0.75rem 1rem;
    }
    
    .streamlit-expanderContent {
        padding: 0.5rem 1rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Centered login form
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("""
        <div class='login-container'>
            <div class='login-form'>
                <div style='text-align: center; margin-bottom: 1rem;'>
                    <h2 style='color: #667eea; margin-bottom: 0.4rem; font-size: 1.6rem;'>Welcome to EduPredict</h2>
                    <p style='color: #666; font-size: 0.9rem; margin: 0;'>Please sign in to access your academic dashboard</p>
                </div>
        """, unsafe_allow_html=True)
        
        # Login form
        username = st.text_input("👤 Username", placeholder="Enter your username", key="login_user")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_pass")
        
        if st.button("🚀 Login to Dashboard", use_container_width=True, key="login_btn"):
            if username in auth_users and password == auth_users[username]:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful! Redirecting...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")
        
        # Demo accounts
        with st.expander("🔍 Demo Accounts", expanded=True):
            st.info("""
            **Student Account:** 
            - Username: student
            - Password: student123
            
            **Teacher Account:** 
            - Username: teacher  
            - Password: teach123
            
            **Counselor Account:**
            - Username: counselor
            - Password: counsel123
            """)
        
        st.markdown("</div></div>", unsafe_allow_html=True)# Main content when logged in
if st.session_state.logged_in:
    role = st.session_state.username.capitalize()
    
    # Lottie animation
    def load_lottieurl(url):
        try:
            r = requests.get(url, timeout=5)
            return r.json()
        except:
            return None

    lottie_animation = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_ydo1amjm.json")
    if lottie_animation:
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st_lottie(lottie_animation, height=180, key="logo")

    # Load data and models
    try:
        df = pd.read_csv("data/academic_cleaned.csv")
        model = joblib.load("models/rf_model.pkl")
        anomaly_model = joblib.load("models/anomaly_model.pkl")
        trend_model = joblib.load("models/trend_model.pkl")
        models_loaded = True
    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        st.info("Please ensure 'data/academic_cleaned.csv' and model files exist in the correct directories.")
        models_loaded = False

    if models_loaded:
        def rebuild_target(row):
            if row["Target_Graduate"] == 1:
                return "Graduate"
            elif row["Target_Enrolled"] == 1:
                return "Enrolled"
            else:
                return "Dropout"

        df["Grade"] = df.apply(rebuild_target, axis=1)

        # Tab content based on selection
        if st.session_state.current_tab == "Prediction":
            st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;'>
                        <span>🧾</span> Student Profile Input
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Input fields with icons
                input_col1, input_col2 = st.columns(2)
                
                with input_col1:
                    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                    age = st.slider("📅 Age at Enrollment", 17, 60, 22)
                    admission_grade = st.slider("📝 Admission Grade", 0.0, 200.0, 120.0)
                    gender = st.selectbox("👥 Gender", ["male", "female"])
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with input_col2:
                    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                    scholarship = st.selectbox("🎓 Scholarship", ["yes", "no"])
                    tuition_paid = st.selectbox("💰 Tuition Status", ["yes", "no"])
                    st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                sem1_grade = st.slider("📚 1st Semester Grade", 0.0, 20.0, 12.0)
                sem2_grade = st.slider("📖 2nd Semester Grade", 0.0, 20.0, 12.0)
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown("<div class='input-card'>", unsafe_allow_html=True)
                unemployment = st.slider("📉 Unemployment Rate", 0.0, 20.0, 7.5)
                inflation = st.slider("💹 Inflation Rate", 0.0, 10.0, 3.0)
                gdp = st.slider("🏦 GDP", 0.0, 200000.0, 100000.0)
                st.markdown("</div>", unsafe_allow_html=True)
                
                if st.button("🔮 Predict Academic Outcome", use_container_width=True, key="predict_btn"):
                    with st.spinner("Analyzing data and generating prediction..."):
                        time.sleep(1)
                        base_input = df.drop(columns=[col for col in df.columns if "Target" in col or col == "Grade"])
                        model_columns = base_input.columns.tolist()
                        input_template = pd.DataFrame(columns=model_columns)
                        input_template.loc[0] = 0

                        input_template["Age at enrollment"] = age
                        input_template["Admission grade"] = admission_grade
                        input_template["Gender"] = 1 if gender == "male" else 0
                        input_template["Scholarship holder"] = 1 if scholarship == "yes" else 0
                        input_template["Tuition fees up to date"] = 1 if tuition_paid == "yes" else 0
                        input_template["Curricular units 1st sem (grade)"] = sem1_grade
                        input_template["Curricular units 2nd sem (grade)"] = sem2_grade
                        input_template["Unemployment rate"] = unemployment
                        input_template["Inflation rate"] = inflation
                        input_template["GDP"] = gdp

                        input_template = input_template[model_columns]

                        prediction = model.predict(input_template)[0]
                        probabilities = model.predict_proba(input_template)[0]
                        confidence = round(probabilities[prediction] * 100, 2)
                        label_map = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
                        result = label_map[prediction]

                        if anomaly_model.predict(input_template)[0] == -1:
                            st.error("🚨 Unusual academic profile detected (Anomaly)!")
                        
                        sem2_pred = trend_model.predict([[sem1_grade]])[0]

                        # Display prediction result
                        if result == "Dropout":
                            st.markdown(f"""
                            <div class='prediction-card error-card'>
                                <h2 style='font-size: 2rem; margin-bottom: 1rem;'>❌ Predicted: {result}</h2>
                                <h3 style='margin: 0;'>Confidence: {confidence}%</h3>
                                <p style='margin: 1rem 0 0 0; opacity: 0.9;'>Early intervention recommended</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif sem1_grade < 8 or sem2_grade < 8:
                            st.markdown(f"""
                            <div class='prediction-card warning-card'>
                                <h2 style='font-size: 2rem; margin-bottom: 1rem;'>⚠️ Predicted: {result}</h2>
                                <h3 style='margin: 0;'>Confidence: {confidence}%</h3>
                                <p style='margin: 1rem 0 0 0; opacity: 0.9;'>Low academic performance detected</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class='prediction-card success-card'>
                                <h2 style='font-size: 2rem; margin-bottom: 1rem;'>🎓 Predicted: {result}</h2>
                                <h3 style='margin: 0;'>Confidence: {confidence}%</h3>
                                <p style='margin: 1rem 0 0 0; opacity: 0.9;'>Keep up the good work!</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Additional insights
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.info(f"**Confidence Score:** {confidence}%")
                        with col_b:
                            st.info(f"**Predicted Semester 2 Grade:** {round(sem2_pred, 2)}")
                        
                        # Download report
                        report = f"""
                        ═══════════════════════════════════════
                        🎓 EDUPREDICT ACADEMIC ANALYSIS REPORT
                        ═══════════════════════════════════════
                        
                        Role: {role}
                        Prediction: {result}
                        Confidence: {confidence}%
                        Anomaly: {"Yes" if anomaly_model.predict(input_template)[0] == -1 else "No"}
                        Predicted Sem-2 Grade: {round(sem2_pred, 2)}
                        
                        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        EduPredict - AI-Powered Academic Success Predictor
                        By Muhammad Anas (2209E01)
                        ═══════════════════════════════════════
                        """
                        st.download_button("📄 Download Detailed Report", report, 
                                        file_name="edu_predict_report.txt", use_container_width=True, key="download_report")

            with col2:
                st.markdown("""
                <div class='metric-card'>
                    <h3 style='color: #667eea; margin-bottom: 1rem; display: flex; align-items: center; gap: 10px;'>
                        <span>📊</span> Performance Insights & Analytics
                    </h3>
                </div>
                """, unsafe_allow_html=True)

                chart_type = st.selectbox("Select Visualization", 
                                        ["Class Distribution", "Gender Performance", "Semester Progress", "Scholarship Impact"],
                                        key="chart_select")

                if chart_type == "Class Distribution":
                    fig = px.pie(df, names="Grade", title="Student Outcome Distribution",
                               color_discrete_sequence=['#ff6b6b', '#4ecdc4', '#45b7d1'])
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Gender Performance":
                    fig = px.box(df, x="Gender", y="Admission grade", 
                               title="Admission Grade Distribution by Gender",
                               color="Gender", color_discrete_map={'male': '#667eea', 'female': '#f093fb'})
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Semester Progress":
                    fig = px.line(df[["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]].reset_index(),
                                title="Semester-wise Academic Progress",
                                labels={"value": "Grade", "variable": "Semester", "index": "Student Index"})
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scholarship Impact":
                    df_scholar = df.copy()
                    df_scholar["Scholarship Status"] = df_scholar["Scholarship holder"].map({1: "Yes", 0: "No"})
                    fig = px.violin(df_scholar, x="Scholarship Status", y="Curricular units 1st sem (grade)",
                                  title="Academic Performance by Scholarship Status", box=True, points="all")
                    st.plotly_chart(fig, use_container_width=True)

                # Summary metrics
                with st.expander("📊 Dataset Overview", expanded=True):
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Total Students", f"{df.shape[0]:,}")
                    with metric_col2:
                        st.metric("Graduation Rate", f"{round(df[df['Grade'] == 'Graduate'].shape[0]/df.shape[0]*100, 1)}%")
                    with metric_col3:
                        st.metric("Avg Admission Grade", round(df["Admission grade"].mean(), 1))
            
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.current_tab == "Analytics":
            st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
            st.markdown("""
            <h3 style='color: #667eea; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px;'>
                <span>📈</span> Advanced Academic Analytics
            </h3>
            """, unsafe_allow_html=True)
            
            # Advanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔹 Enrollment by Course")
                fig_course = px.histogram(df, x="Course", color="Grade", barmode="group",
                                       color_discrete_map={'Dropout': '#ff6b6b', 'Enrolled': '#4ecdc4', 'Graduate': '#45b7d1'})
                st.plotly_chart(fig_course, use_container_width=True)
                
                st.markdown("#### 🔹 Economic Factors Impact")
                fig_gdp = px.scatter(df, x="GDP", y="Admission grade", color="Grade",
                                 title="GDP vs Admission Grade", trendline="ols")
                st.plotly_chart(fig_gdp, use_container_width=True)
            
            with col2:
                st.markdown("#### 🔹 Age Group Analysis")
                df["Age Group"] = pd.cut(df["Age at enrollment"], bins=[16, 20, 25, 30, 40, 60],
                                 labels=["17-20", "21-25", "26-30", "31-40", "41+"])
                dropout_by_age = df[df["Grade"] == "Dropout"]["Age Group"].value_counts(normalize=True).sort_index()
                fig_age = px.bar(x=dropout_by_age.index, y=dropout_by_age.values * 100,
                         labels={"x": "Age Group", "y": "Dropout %"}, 
                         title="Dropout Percentage by Age Group",
                         color=dropout_by_age.index, color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig_age, use_container_width=True)
                
                st.markdown("#### 🔹 Scholarship Impact Analysis")
                df["Scholarship Label"] = df["Scholarship holder"].map({1: "Yes", 0: "No"})
                grade_scholar = df.groupby("Scholarship Label")[["Curricular units 1st sem (grade)", 
                                                         "Curricular units 2nd sem (grade)"]].mean().reset_index()
                fig_scholar = px.bar(grade_scholar, x="Scholarship Label", 
                                 y=["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"],
                                 barmode="group", title="Average Grades: Scholarship vs Non-Scholarship",
                                 labels={"value": "Average Grade", "variable": "Semester"})
                st.plotly_chart(fig_scholar, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.current_tab == "Reports":
            st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
            st.markdown("""
            <h3 style='color: #667eea; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px;'>
                <span>📋</span> Academic Reports
            </h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Generate Custom Report")
                report_type = st.selectbox("Select Report Type", 
                                         ["Student Performance", "Course Analytics", "Institutional Overview"])
                time_range = st.selectbox("Time Period", 
                                        ["Last Semester", "Last Year", "All Time"])
                
                if st.button("Generate Report", use_container_width=True):
                    with st.spinner("Generating report..."):
                        time.sleep(2)
                        st.success("Report generated successfully!")
                        
                        # Sample report data
                        report_data = {
                            "Total Students": df.shape[0],
                            "Average Admission Grade": round(df["Admission grade"].mean(), 2),
                            "Graduation Rate": f"{round(df[df['Grade'] == 'Graduate'].shape[0]/df.shape[0]*100, 1)}%",
                            "Dropout Rate": f"{round(df[df['Grade'] == 'Dropout'].shape[0]/df.shape[0]*100, 1)}%"
                        }
                        
                        st.dataframe(pd.DataFrame.from_dict(report_data, orient='index', columns=['Value']))
            
            with col2:
                st.markdown("#### 📈 Recent Reports")
                st.info("No recent reports available. Generate a new report to see it here.")
                
            st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.current_tab == "Settings":
            st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
            st.markdown("""
            <h3 style='color: #667eea; margin-bottom: 1.5rem; display: flex; align-items: center; gap: 10px;'>
                <span>⚙️</span> User Settings
            </h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 👤 Profile Settings")
                st.text_input("Full Name", value=f"{role} User")
                st.text_input("Email", value=f"{st.session_state.username}@edupredict.com")
                st.selectbox("Notification Preferences", ["All notifications", "Important only", "None"])
                st.button("Update Profile", use_container_width=True)
            
            with col2:
                st.markdown("#### 🎨 Display Settings")
                st.selectbox("Theme", ["Light", "Dark", "Auto"])
                st.selectbox("Language", ["English", "Spanish", "French"])
                st.slider("Font Size", 12, 24, 16)
                st.button("Save Preferences", use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Role-specific guidance
        st.markdown("---")
        guidance_col1, guidance_col2, guidance_col3 = st.columns(3)
        
        with guidance_col1:
            if role == "Student":
                st.info("""
                **📚 Student Guidance:**
                - Monitor your academic standing regularly
                - Use predictions for self-assessment
                - Seek help early if risks are identified
                """)
        
        with guidance_col2:
            if role == "Teacher":
                st.info("""
                **👩‍🏫 Teacher Guidance:**
                - Identify at-risk students early
                - Track academic progress patterns
                - Provide targeted support
                """)
        
        with guidance_col3:
            if role == "Counselor":
                st.info("""
                **🧠 Counselor Guidance:**
                - Proactive intervention planning
                - Identify high-risk cases
                - Develop support strategies
                """)

        # Footer
        st.markdown("""
        <div class='footer'>
            <h3 style='margin: 0 0 1rem 0;'>📬 Feedback & Support</h3>
            <p style='margin: 0 0 1.5rem 0;'>We value your feedback to improve EduPredict</p>
            <div style='display: flex; justify-content: center; gap: 1rem; margin-bottom: 1.5rem;'>
                <a href='https://forms.gle/Am71U3oEjHG42sJ59' style='color: white; text-decoration: none; 
                   background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 25px;'>Submit Feedback</a>
                <a href='#' style='color: white; text-decoration: none; 
                   background: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 25px;'>Help Center</a>
            </div>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'>EduPredict | Built with ❤️ for Academic Excellence</p>
            <p style='margin: 0; font-size: 0.9rem; opacity: 0.8;'>By Muhammad Anas (2209E01)</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("🚫 Cannot proceed without loading the required models and data files.")
        st.info("Please check if these files exist:")
        st.code("""
        📁 Project Structure:
        ├── data/
        │   └── academic_cleaned.csv
        ├── models/
        │   ├── rf_model.pkl
        │   ├── anomaly_model.pkl
        │   └── trend_model.pkl
        └── app/
            └── edu_predict_app.py
        """)
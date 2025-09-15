#!/usr/bin/env python3
"""
Enhanced Excel Data Cleaner with Multi-LLM Pipeline
Features:
- Multi-stage LLM cleaning pipeline
- Comprehensive metadata generation
- Code generation with retry logic
- Data quality scoring
- Automatic re-cleaning if quality is low
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import re
import time
import traceback
import logging
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Dict, List, Any, Tuple, Optional
from io import BytesIO
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(override=True)

# Import custom modules
from ai_service import make_ai_call, get_ai_service
from config import get_anthropic_api_key
from header_detector import SimpleHeaderDetector

# Import data quality scorer with fallback
try:
    from data_quality_scorer import calculate_quality_score, get_quality_report
except ImportError as e:
    print(f"Warning: Could not import from data_quality_scorer: {e}")
    # Fallback implementation
    def calculate_quality_score(df):
        # Basic quality score calculation
        missing_pct = (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0
        score = max(0, 100 - missing_pct)
        return score
    
    def get_quality_report(df):
        return {
            "score": calculate_quality_score(df),
            "quality_level": "UNKNOWN",
            "issues": [],
            "suggestions": []
        }

from cleaning_llm import CleaningLLM
from code_generator_llm import CodeGeneratorLLM
from format_validator import UniversalFormatValidator, highlight_format_violations

# Page configuration
st.set_page_config(
    page_title="Advanced Excel Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
    .quality-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .quality-high {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .quality-medium {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .quality-low {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    .metadata-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border-radius: 5px;
        transition: transform 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'uploaded_files_dict' not in st.session_state:
        st.session_state.uploaded_files_dict = {}  # Store multiple files with sheets info
    if 'selected_file' not in st.session_state:
        st.session_state.selected_file = None
    if 'selected_sheet' not in st.session_state:
        st.session_state.selected_sheet = None
    if 'available_sheets' not in st.session_state:
        st.session_state.available_sheets = []
    if 'uploaded_file_content' not in st.session_state:
        st.session_state.uploaded_file_content = None
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'cleaning_suggestions' not in st.session_state:
        st.session_state.cleaning_suggestions = []
    if 'quality_score' not in st.session_state:
        st.session_state.quality_score = 0
    if 'quality_report' not in st.session_state:
        st.session_state.quality_report = {}
    if 'cleaning_iteration' not in st.session_state:
        st.session_state.cleaning_iteration = 0
    if 'cleaning_history' not in st.session_state:
        st.session_state.cleaning_history = []
    if 'llm_logs' not in st.session_state:
        st.session_state.llm_logs = []

init_session_state()

def detect_and_process_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Simple processing: Remove empty rows/columns, use first row as header
    """
    changes = []
    
    try:
        # Initialize simple header detector
        header_detector = SimpleHeaderDetector()
        
        # Simple processing: remove empty rows/columns and use first row as header
        header_row_index, processed_df, detection_log = header_detector.detect_header_row(df)
        
        # Add detection logs to changes
        changes.extend(detection_log)
        
        return processed_df, changes
    except Exception as e:
        # Fallback: just use the original dataframe
        changes.append(f"⚠️ Processing failed: {str(e)[:100]}")
        changes.append("Using original data as-is")
        
        return df, changes

def basic_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    DISABLED: All cleaning now done in header processing step
    """
    changes = []
    changes.append("✅ No additional cleaning required - already processed")
    return df, changes

def generate_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive metadata for the dataframe
    """
    # Helper function to convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    # Convert dataframe records to serializable format
    def df_to_serializable_records(df_subset):
        records = []
        for _, row in df_subset.iterrows():
            record = {}
            for col, val in row.items():
                record[str(col)] = convert_numpy(val)
            records.append(record)
        return records
    
    metadata = {
        "basic_info": {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "memory_usage": float(df.memory_usage(deep=True).sum() / 1024**2),  # MB
            "shape": [int(df.shape[0]), int(df.shape[1])]
        },
        "columns": {},
        "data_samples": {
            "first_10_rows": df_to_serializable_records(df.head(10)),
            "random_10_rows": df_to_serializable_records(df.sample(min(10, len(df)))) if len(df) > 0 else [],
            "last_10_rows": df_to_serializable_records(df.tail(10))
        },
        "missing_values": {},
        "data_types": {},
        "unique_counts": {},
        "statistics": {}
    }
    
    # Column-wise analysis
    for col in df.columns:
        col_data = df[col]
        col_str = str(col)
        
        # Convert sample values to serializable format
        sample_vals = []
        for val in col_data.dropna().head(10):
            sample_vals.append(convert_numpy(val))
        
        # Convert value counts to serializable format
        val_counts = {}
        if col_data.nunique() < 100:
            for val, count in col_data.value_counts().head(10).items():
                val_counts[str(val)] = int(count)
        
        # Basic info
        metadata["columns"][col_str] = {
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isnull().sum()),
            "missing_percentage": float((col_data.isnull().sum() / len(df)) * 100),
            "unique_count": int(col_data.nunique()),
            "unique_percentage": float((col_data.nunique() / len(df)) * 100) if len(df) > 0 else 0,
            "sample_values": sample_vals,
            "value_counts": val_counts
        }
        
        # Missing values summary
        metadata["missing_values"][col_str] = {
            "count": int(col_data.isnull().sum()),
            "percentage": float((col_data.isnull().sum() / len(df)) * 100)
        }
        
        # Data type
        metadata["data_types"][col_str] = str(col_data.dtype)
        
        # Unique counts
        metadata["unique_counts"][col_str] = int(col_data.nunique())
        
        # Statistics for numeric columns
        if pd.api.types.is_numeric_dtype(col_data):
            metadata["statistics"][col_str] = {
                "mean": float(col_data.mean()) if not col_data.empty and not pd.isna(col_data.mean()) else None,
                "median": float(col_data.median()) if not col_data.empty and not pd.isna(col_data.median()) else None,
                "std": float(col_data.std()) if not col_data.empty and not pd.isna(col_data.std()) else None,
                "min": float(col_data.min()) if not col_data.empty and not pd.isna(col_data.min()) else None,
                "max": float(col_data.max()) if not col_data.empty and not pd.isna(col_data.max()) else None,
                "q25": float(col_data.quantile(0.25)) if not col_data.empty and not pd.isna(col_data.quantile(0.25)) else None,
                "q75": float(col_data.quantile(0.75)) if not col_data.empty and not pd.isna(col_data.quantile(0.75)) else None
            }
    
    # Data quality indicators
    metadata["quality_indicators"] = {
        "has_duplicate_rows": bool(df.duplicated().any()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "has_missing_values": bool(df.isnull().any().any()),
        "total_missing_values": int(df.isnull().sum().sum()),
        "missing_value_percentage": float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100) if df.size > 0 else 0,
        "columns_with_single_value": [str(col) for col in df.columns if df[col].nunique() == 1],
        "potentially_categorical": [str(col) for col in df.columns if df[col].nunique() < 10 and df[col].dtype == 'object'],
        "potentially_datetime": detect_potential_datetime_columns(df),
        "potentially_numeric_text": detect_numeric_text_columns(df)
    }
    
    return metadata

def detect_potential_datetime_columns(df: pd.DataFrame) -> List[str]:
    """Detect columns that might contain datetime data"""
    datetime_cols = []
    date_patterns = [
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        r'\d{1,2}-\w{3}-\d{2,4}',
        r'\w{3}\s+\d{1,2},?\s+\d{4}'
    ]
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(10).astype(str)
            for val in sample:
                for pattern in date_patterns:
                    if re.search(pattern, val):
                        datetime_cols.append(col)
                        break
                if col in datetime_cols:
                    break
    
    return datetime_cols

def detect_numeric_text_columns(df: pd.DataFrame) -> List[str]:
    """Detect text columns that contain numeric values"""
    numeric_text_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            sample = df[col].dropna().head(20)
            numeric_count = 0
            for val in sample:
                try:
                    # Try to convert to float
                    float(str(val).replace(',', '').replace('$', '').replace('€', '').replace('£', ''))
                    numeric_count += 1
                except:
                    pass
            
            if numeric_count > len(sample) * 0.5:  # More than 50% are numeric
                numeric_text_cols.append(col)
    
    return numeric_text_cols

def display_metadata(metadata: Dict[str, Any]):
    """Display metadata in a user-friendly format"""
    with st.expander("📊 Dataset Metadata", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", metadata["basic_info"]["total_rows"])
        with col2:
            st.metric("Total Columns", metadata["basic_info"]["total_columns"])
        with col3:
            st.metric("Memory Usage", f"{metadata['basic_info']['memory_usage']:.2f} MB")
        with col4:
            st.metric("Missing Values", f"{metadata['quality_indicators']['missing_value_percentage']:.1f}%")
        
        # Column details
        st.subheader("Column Analysis")
        col_df = pd.DataFrame({
            'Column': list(metadata['columns'].keys()),
            'Type': [metadata['data_types'][col] for col in metadata['columns'].keys()],
            'Missing %': [f"{metadata['missing_values'][col]['percentage']:.1f}%" for col in metadata['columns'].keys()],
            'Unique': [metadata['unique_counts'][col] for col in metadata['columns'].keys()]
        })
        st.dataframe(col_df, use_container_width=True)
        
        # Quality indicators
        st.subheader("Data Quality Indicators")
        quality_cols = st.columns(2)
        with quality_cols[0]:
            st.write(f"**Duplicate Rows:** {metadata['quality_indicators']['duplicate_row_count']}")
            st.write(f"**Single Value Columns:** {len(metadata['quality_indicators']['columns_with_single_value'])}")
            st.write(f"**Potentially Categorical:** {len(metadata['quality_indicators']['potentially_categorical'])}")
        with quality_cols[1]:
            st.write(f"**Potentially DateTime:** {len(metadata['quality_indicators']['potentially_datetime'])}")
            st.write(f"**Numeric Text Columns:** {len(metadata['quality_indicators']['potentially_numeric_text'])}")

def display_quality_score(score: float, report: Dict[str, Any]):
    """Display data quality score with visual feedback"""
    st.markdown("<div class='section-header'>Data Quality Score</div>", unsafe_allow_html=True)
    
    # Determine quality level
    if score >= 80:
        quality_class = "quality-high"
        quality_text = "Excellent"
        quality_emoji = "✨"
    elif score >= 50:
        quality_class = "quality-medium"
        quality_text = "Good"
        quality_emoji = "👍"
    else:
        quality_class = "quality-low"
        quality_text = "Needs Improvement"
        quality_emoji = "⚠️"
    
    st.markdown(f"""
        <div class='{quality_class} quality-score'>
            {quality_emoji} {score:.1f}/100 - {quality_text}
        </div>
    """, unsafe_allow_html=True)
    
    # Display detailed report
    with st.expander("📋 Detailed Quality Report", expanded=False):
        for category, details in report.items():
            st.subheader(category)
            if isinstance(details, dict):
                for key, value in details.items():
                    if isinstance(value, bool):
                        st.write(f"**{key}:** {'✅' if value else '❌'}")
                    else:
                        st.write(f"**{key}:** {value}")
            else:
                st.write(details)

def main():
    st.markdown("<h1 class='main-header'>🧹 Advanced Excel Data Cleaner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d;'>Multi-LLM Pipeline for Comprehensive Data Cleaning</p>", unsafe_allow_html=True)
    
    # Show file count if multiple files uploaded
    if len(st.session_state.uploaded_files_dict) > 0:
        col1, col2, col3 = st.columns([2, 3, 2])
        with col2:
            file_count = len(st.session_state.uploaded_files_dict)
            if file_count == 1:
                st.info(f"📄 **1 file** uploaded")
            else:
                st.info(f"📁 **{file_count} files** uploaded - Select one to analyze")
    
    # Check AI service availability
    ai_service = get_ai_service()
    if not ai_service.is_available():
        st.error("⚠️ AI Service Not Configured")
        st.info("Please add ANTHROPIC_API_KEY or OPENAI_API_KEY to your environment")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Cleaning Pipeline")
        st.info("""
        **Pipeline Stages:**
        1. 📤 Upload & Basic Clean
        2. 📊 Metadata Generation
        3. 🤖 LLM Analysis
        4. 🔧 Code Generation
        5. ✅ Quality Check
        6. 🔄 Auto Re-clean (if needed)
        """)
        
        st.markdown("---")
        
        # Pipeline status
        if st.session_state.cleaning_iteration > 0:
            st.markdown("### Pipeline Status")
            st.write(f"**Iteration:** {st.session_state.cleaning_iteration}/2")
            st.write(f"**Current Score:** {st.session_state.quality_score:.1f}/100")
        
        # Batch processing section
        if len(st.session_state.uploaded_files_dict) > 1:
            st.markdown("---")
            st.markdown("### 🚀 Batch Processing")
            st.write(f"**{len(st.session_state.uploaded_files_dict)} files** available")
            
            if st.button("⚡ Clean All Files", use_container_width=True):
                st.info("Batch processing is available in Pro version")
        
        # Clear session button
        st.markdown("---")
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            init_session_state()
            st.rerun()
    
    # Main content area
    tabs = st.tabs(["📤 Upload", "🧹 Clean", "📊 Results", "📜 Logs"])
    
    with tabs[0]:
        st.markdown("<div class='section-header'>Step 1: Upload Excel Files</div>", unsafe_allow_html=True)
        
        # Multiple file upload
        uploaded_files = st.file_uploader(
            "Choose Excel files (you can select multiple)",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload one or more Excel files for cleaning"
        )
        
        if uploaded_files:
            # Store all uploaded files with sheet information
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files_dict:
                    content = uploaded_file.getvalue()
                    # Detect sheets in the file
                    try:
                        with pd.ExcelFile(BytesIO(content), engine='openpyxl') as xl:
                            sheet_names = xl.sheet_names
                    except:
                        sheet_names = ['Sheet1']  # Default if detection fails
                    
                    st.session_state.uploaded_files_dict[uploaded_file.name] = {
                        'file': uploaded_file,
                        'content': content,
                        'sheets': sheet_names
                    }
            
            # Show file selection dropdown
            st.success(f"✅ {len(st.session_state.uploaded_files_dict)} file(s) uploaded")
            
            # File selector
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_filename = st.selectbox(
                    "Select a file to analyze:",
                    options=list(st.session_state.uploaded_files_dict.keys()),
                    key="file_selector"
                )
            
            with col2:
                if st.button("🗑️ Remove Selected", type="secondary"):
                    if selected_filename in st.session_state.uploaded_files_dict:
                        del st.session_state.uploaded_files_dict[selected_filename]
                        st.rerun()
            
            if selected_filename:
                st.session_state.selected_file = selected_filename
                file_data = st.session_state.uploaded_files_dict[selected_filename]
                st.session_state.available_sheets = file_data['sheets']
                
                # Show sheet selection if multiple sheets exist
                if len(file_data['sheets']) > 1:
                    st.markdown("---")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        selected_sheet = st.selectbox(
                            "📋 Select a sheet to analyze:",
                            options=file_data['sheets'],
                            key="sheet_selector",
                            help=f"This file contains {len(file_data['sheets'])} sheets"
                        )
                    with col2:
                        st.metric("Sheets", len(file_data['sheets']))
                else:
                    selected_sheet = file_data['sheets'][0]
                    st.info(f"📋 This file has 1 sheet: **{selected_sheet}**")
                
                st.session_state.selected_sheet = selected_sheet
                
                # Show all uploaded files with sheet counts
                with st.expander("📁 All Uploaded Files", expanded=False):
                    for idx, (filename, file_info) in enumerate(st.session_state.uploaded_files_dict.items(), 1):
                        sheet_count = len(file_info['sheets'])
                        status = "✅ Selected" if filename == selected_filename else "📄 Ready"
                        sheets_text = f"({sheet_count} sheet{'s' if sheet_count > 1 else ''})"
                        st.write(f"{idx}. {filename} {sheets_text} - {status}")
                        if filename == selected_filename:
                            st.write(f"   Sheets: {', '.join(file_info['sheets'])}")
                
                # Load and process selected file and sheet
                if st.button("📊 Load Selected Sheet", type="primary", use_container_width=True):
                    try:
                        # Get the selected file content
                        file_data = st.session_state.uploaded_files_dict[selected_filename]
                        
                        # Read the specific sheet WITHOUT header assumption
                        df_raw = pd.read_excel(BytesIO(file_data['content']), sheet_name=selected_sheet, header=None, engine='openpyxl')
                        
                        # Simple approach: Keep all data exactly as uploaded
                        df = df_raw.copy()
                        st.success(f"📋 Raw Excel data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
                        
                        st.session_state.uploaded_file_content = file_data['content']
                        st.session_state.original_df = df.copy()  # Store the trimmed version as original
                        
                        # Show simple processing info
                        st.info("📝 Simple processing: Will remove only completely empty rows/columns, then use first row as headers")
                        
                        # Initial Quality Assessment
                        with st.expander("📊 Initial Quality Assessment", expanded=True):
                            st.info("🔍 Analyzing raw data quality before any cleaning...")
                            
                            try:
                                # Calculate initial quality score
                                initial_quality_score = calculate_quality_score(df)
                                initial_quality_report = get_quality_report(df)
                                
                                # Display quality metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if initial_quality_score >= 80:
                                        quality_color = "🟢"
                                    elif initial_quality_score >= 60:
                                        quality_color = "🟡"
                                    else:
                                        quality_color = "🔴"
                                    
                                    st.metric(
                                        "Initial Quality Score",
                                        f"{quality_color} {initial_quality_score:.1f}%",
                                        help="Quality assessment of raw uploaded data"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Data Completeness",
                                        f"{100 - (df.isnull().sum().sum() / df.size * 100):.1f}%",
                                        help="Percentage of non-empty cells"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Structure",
                                        f"{df.shape[0]} × {df.shape[1]}",
                                        help="Rows × Columns after smart trimming"
                                    )
                                
                                # Show quality issues if any
                                if initial_quality_report.get('issues'):
                                    st.warning("⚠️ **Issues Detected:**")
                                    for issue in initial_quality_report['issues']:
                                        st.write(f"• {issue}")
                                
                                # Show detailed quality breakdown
                                if 'detailed_scores' in initial_quality_report:
                                    with st.expander("🔍 Detailed Quality Breakdown", expanded=False):
                                        scores = initial_quality_report['detailed_scores']
                                        for metric, score in scores.items():
                                            st.progress(score/100, text=f"{metric.replace('_', ' ').title()}: {score:.1f}%")
                                
                                # Store initial quality for comparison later
                                st.session_state.initial_quality_score = initial_quality_score
                                st.session_state.initial_quality_report = initial_quality_report
                                
                            except Exception as e:
                                st.error(f"❌ Initial quality assessment failed: {str(e)}")
                                st.session_state.initial_quality_score = 0
                        
                        # Simple processing: Remove empty rows/columns and set first row as headers
                        df_processed, processing_changes = detect_and_process_headers(df)
                        
                        # No additional cleaning - just use the processed data
                        changes = processing_changes
                        st.session_state.current_df = df_processed
                        
                        st.success(f"✅ Sheet loaded successfully: {selected_sheet} from {selected_filename}")
                        
                        # Display basic info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("File", selected_filename[:20] + "..." if len(selected_filename) > 20 else selected_filename)
                        with col2:
                            st.metric("Sheet", selected_sheet)
                        with col3:
                            st.metric("Original", f"{df.shape[0]}×{df.shape[1]}")
                        with col4:
                            st.metric("Cleaned", f"{df_processed.shape[0]}×{df_processed.shape[1]}")
                        
                        if changes:
                            st.write("**Basic Cleaning Applied:**")
                            for change in changes:
                                st.write(f"• {change}")
                        
                        # Show preview with format validation
                        st.subheader("Data Preview (After Basic Cleaning)")
                        
                        # Validate formats
                        validator = UniversalFormatValidator()
                        _, violations = validator.validate_dataframe(df_processed)
                        
                        if violations:
                            st.warning(f"⚠️ Found format violations in {len(violations)} columns")
                            with st.expander("📋 Format Violations Details"):
                                for col, info in violations.items():
                                    st.write(f"**{col}**: Expected format: {info['format_description']}")
                                    st.write(f"  - {len(info['violations'])} violations found")
                            
                            # Show styled dataframe with red highlights
                            st.write("❌ Red cells indicate format violations:")
                            styled_df = highlight_format_violations(df_processed.head(20), violations)
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.dataframe(df_processed.head(20), use_container_width=True)
                        
                        # Generate and display metadata
                        with st.spinner("Generating metadata..."):
                            metadata = generate_metadata(df_processed)
                            st.session_state.metadata = metadata
                        
                        display_metadata(metadata)
                        
                        st.success("✅ Ready for advanced cleaning! Go to the Clean tab.")
                        
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
    
    with tabs[1]:
        st.markdown("<div class='section-header'>Step 2: Advanced Cleaning Pipeline</div>", unsafe_allow_html=True)
        
        if st.session_state.current_df is None:
            st.warning("⚠️ Please upload and load a file first")
        else:
            # Show which file and sheet is being cleaned
            if st.session_state.selected_file:
                if st.session_state.selected_sheet:
                    st.info(f"📄 **Currently cleaning:** {st.session_state.selected_file} → Sheet: {st.session_state.selected_sheet}")
                else:
                    st.info(f"📄 **Currently cleaning:** {st.session_state.selected_file}")
            # Display current data quality
            if st.session_state.quality_score > 0:
                display_quality_score(st.session_state.quality_score, st.session_state.quality_report)
            
            # Cleaning controls - HARDCODED VALUES
            auto_clean = True  # Always auto re-clean
            max_iterations = 2  # Hardcoded to 2 iterations
            quality_threshold = 50  # Hardcoded threshold of 50
            
            st.info(f"📋 Settings: Max {max_iterations} iterations | Quality threshold: {quality_threshold}% | Auto re-clean: Enabled")
            
            if st.button("🚀 Start Advanced Cleaning", type="primary", use_container_width=True):
                cleaning_container = st.container()
                
                with cleaning_container:
                    # Initialize cleaning pipeline
                    cleaner = CleaningLLM()
                    code_gen = CodeGeneratorLLM()
                    
                    iteration = 0
                    quality_score = 0
                    
                    while iteration < max_iterations:
                        iteration += 1
                        st.session_state.cleaning_iteration = iteration
                        
                        st.markdown(f"### 🔄 Cleaning Iteration {iteration}")
                        
                        progress = st.progress(0)
                        status = st.empty()
                        
                        # Step 1: LLM Analysis
                        status.text("🤖 Analyzing data with LLM...")
                        progress.progress(20)
                        
                        suggestions = cleaner.analyze_and_suggest(
                            st.session_state.current_df,
                            st.session_state.metadata
                        )
                        st.session_state.cleaning_suggestions = suggestions
                        
                        if suggestions:
                            st.write(f"**Found {len(suggestions)} cleaning suggestions:**")
                            for i, suggestion in enumerate(suggestions[:5]):  # Show first 5
                                st.write(f"{i+1}. {suggestion['description']}")
                        
                        # Step 2: Generate and apply cleaning code
                        status.text("🔧 Generating cleaning code...")
                        progress.progress(40)
                        
                        cleaned_df = st.session_state.current_df.copy()
                        applied_count = 0
                        
                        for suggestion in suggestions:
                            if suggestion.get('requires_code', False):
                                # Generate code with retry logic
                                success = False
                                for attempt in range(3):
                                    try:
                                        # Ensure context is serializable
                                        context = suggestion.get('context', {})
                                        if isinstance(context, dict):
                                            # Clean up context to avoid serialization issues
                                            clean_context = {}
                                            for k, v in context.items():
                                                if k == 'dtypes' and hasattr(v, 'to_dict'):
                                                    clean_context[k] = {str(col): str(dtype) for col, dtype in v.to_dict().items()}
                                                elif k == 'columns':
                                                    clean_context[k] = [str(col) for col in v] if isinstance(v, list) else str(v)
                                                else:
                                                    clean_context[k] = v
                                            context = clean_context
                                        
                                        code = code_gen.generate_code(
                                            cleaned_df,
                                            suggestion['description'],
                                            context
                                        )
                                        
                                        # Execute code with better error handling
                                        local_vars = {
                                            'df': cleaned_df.copy(),
                                            'pd': pd,
                                            'np': np,
                                            're': re,
                                            'datetime': datetime
                                        }
                                        exec(code, {'__builtins__': __builtins__}, local_vars)
                                        cleaned_df = local_vars['df']
                                        success = True
                                        applied_count += 1
                                        break
                                    except Exception as e:
                                        error_msg = str(e)
                                        if 'ObjectDType' in error_msg:
                                            error_msg = "Data type conversion issue"
                                        st.warning(f"Attempt {attempt + 1} failed for: {suggestion['description'][:50]}...")
                                        if attempt == 2:
                                            st.error(f"Failed after 3 attempts: {error_msg}")
                        
                        # Step 2.5: Apply universal format standardization
                        status.text("🔧 Applying universal format standardization...")
                        progress.progress(50)
                        
                        validator = UniversalFormatValidator()
                        cleaned_df, format_changes = validator.auto_standardize_dataframe(cleaned_df)
                        
                        if format_changes:
                            st.write("**📐 Format Standardization Applied:**")
                            for change in format_changes:
                                st.write(f"  • {change}")
                        
                        st.session_state.current_df = cleaned_df
                        st.write(f"✅ Applied {applied_count}/{len(suggestions)} suggestions + format standardization")
                        
                        # Step 3: Calculate quality score
                        status.text("📊 Calculating data quality score...")
                        progress.progress(60)
                        
                        quality_score = calculate_quality_score(cleaned_df)
                        quality_report = get_quality_report(cleaned_df)
                        
                        st.session_state.quality_score = quality_score
                        st.session_state.quality_report = quality_report
                        
                        display_quality_score(quality_score, quality_report)
                        
                        # Step 4: Check if re-cleaning is needed
                        progress.progress(80)
                        
                        if quality_score >= quality_threshold:
                            status.text("✨ Cleaning complete! Quality threshold met.")
                            progress.progress(100)
                            st.success(f"🎉 Data cleaning completed successfully! Final quality score: {quality_score:.1f}/100")
                            break
                        elif iteration < max_iterations and auto_clean:
                            st.warning(f"Quality score {quality_score:.1f} is below threshold {quality_threshold}. Starting iteration {iteration + 1}...")
                            time.sleep(1)
                        else:
                            status.text("Cleaning complete (max iterations reached or manual mode)")
                            progress.progress(100)
                            break
                    
                    # Check final quality score - if still below threshold after max iterations, revert to original
                    if quality_score < quality_threshold and iteration >= max_iterations:
                        st.error(f"⚠️ QUALITY THRESHOLD NOT MET: Final score {quality_score:.1f}% is below {quality_threshold}% after {max_iterations} attempts.")
                        st.warning("🔄 **REVERTING TO ORIGINAL EXCEL FILE** - The cleaned data quality is too low for safe use.")
                        
                        # Revert to original dataframe
                        st.session_state.current_df = st.session_state.original_df.copy()
                        
                        # Calculate quality score for original data for comparison
                        original_quality_score = calculate_quality_score(st.session_state.original_df)
                        st.session_state.quality_score = original_quality_score
                        
                        st.info(f"📊 Showing original Excel file with quality score: {original_quality_score:.1f}%")
                    
                    # Save cleaning history
                    st.session_state.cleaning_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'iterations': iteration,
                        'final_score': quality_score,
                        'suggestions_applied': applied_count,
                        'reverted_to_original': quality_score < quality_threshold and iteration >= max_iterations
                    })
    
    with tabs[2]:
        st.markdown("<div class='section-header'>Cleaning Results</div>", unsafe_allow_html=True)
        
        if st.session_state.current_df is not None:
            # Show which file and sheet's results
            if st.session_state.selected_file:
                if st.session_state.selected_sheet:
                    st.info(f"📄 **Results for:** {st.session_state.selected_file} → Sheet: {st.session_state.selected_sheet}")
                else:
                    st.info(f"📄 **Results for:** {st.session_state.selected_file}")
            # Comparison metrics
            col1, col2, col3 = st.columns(3)
            
            if st.session_state.original_df is not None:
                with col1:
                    st.metric(
                        "Rows",
                        st.session_state.current_df.shape[0],
                        st.session_state.current_df.shape[0] - st.session_state.original_df.shape[0]
                    )
                with col2:
                    st.metric(
                        "Columns",
                        st.session_state.current_df.shape[1],
                        st.session_state.current_df.shape[1] - st.session_state.original_df.shape[1]
                    )
                with col3:
                    # Calculate improvement from initial score
                    initial_score = getattr(st.session_state, 'initial_quality_score', 0)
                    current_score = st.session_state.quality_score
                    improvement = current_score - initial_score
                    
                    # Format the delta with proper sign and context
                    if improvement > 0:
                        delta_text = f"+{improvement:.1f}% (improved)"
                        delta_color = "normal"
                    elif improvement < 0:
                        delta_text = f"{improvement:.1f}% (degraded)"
                        delta_color = "inverse"
                    else:
                        delta_text = "No change"
                        delta_color = "off"
                    
                    st.metric(
                        "Quality Score",
                        f"{current_score:.1f}%",
                        delta_text,
                        delta_color=delta_color,
                        help=f"Initial: {initial_score:.1f}% → Final: {current_score:.1f}%"
                    )
            
            # Check if data was reverted to original
            reverted_to_original = False
            if st.session_state.cleaning_history:
                latest_cleaning = st.session_state.cleaning_history[-1]
                reverted_to_original = latest_cleaning.get('reverted_to_original', False)
            
            # Display appropriate title based on whether data was reverted
            if reverted_to_original:
                st.subheader("📝 Original Excel Data (Quality Too Low After Cleaning)")
                st.warning("🚫 The cleaning process did not achieve the required quality threshold. Showing original data for safety.")
            else:
                st.subheader("✨ Cleaned Data")
            
            # Validate final formats
            validator = UniversalFormatValidator()
            _, final_violations = validator.validate_dataframe(st.session_state.current_df)
            
            if final_violations:
                st.warning(f"⚠️ {len(final_violations)} columns still have format violations")
                
                # Show format violation summary
                with st.expander("📋 Remaining Format Issues"):
                    for col, info in final_violations.items():
                        st.write(f"**{col}**: {len(info['violations'])} violations - Expected: {info['format_description']}")
                
                # Show styled dataframe with highlights
                st.write("Data with format highlights (red = violation):")
                styled_final = highlight_format_violations(st.session_state.current_df, final_violations)
                st.dataframe(styled_final, use_container_width=True)
            else:
                st.success("✅ All data follows universal format standards!")
                st.dataframe(st.session_state.current_df, use_container_width=True)
            
            # Download button
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.current_df.to_excel(writer, index=False, sheet_name='Data')
            
            # Determine if data was reverted to original
            reverted_to_original = False
            if st.session_state.cleaning_history:
                latest_cleaning = st.session_state.cleaning_history[-1]
                reverted_to_original = latest_cleaning.get('reverted_to_original', False)
            
            # Create filename with original name and sheet
            if st.session_state.selected_file:
                base_name = st.session_state.selected_file.rsplit('.', 1)[0]
                if st.session_state.selected_sheet and len(st.session_state.available_sheets) > 1:
                    # Include sheet name only if multiple sheets exist
                    sheet_name = st.session_state.selected_sheet.replace(' ', '_')
                    if reverted_to_original:
                        download_filename = f"original_{base_name}_{sheet_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    else:
                        download_filename = f"cleaned_{base_name}_{sheet_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                else:
                    if reverted_to_original:
                        download_filename = f"original_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    else:
                        download_filename = f"cleaned_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            else:
                if reverted_to_original:
                    download_filename = f"original_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                else:
                    download_filename = f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # Set button label and type based on reversion status
            if reverted_to_original:
                button_label = "📄 Download Original Excel (Quality Check Failed)"
                button_type = "secondary"
            else:
                button_label = "📥 Download Cleaned Excel"
                button_type = "primary"
            
            st.download_button(
                label=button_label,
                data=output.getvalue(),
                file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type=button_type
            )
            
            # Cleaning summary
            if st.session_state.cleaning_history:
                st.subheader("Cleaning Summary")
                for i, history in enumerate(st.session_state.cleaning_history):
                    reverted = history.get('reverted_to_original', False)
                    session_title = f"Session {i+1} - {history['timestamp'][:19]}"
                    if reverted:
                        session_title += " ⚠️ REVERTED"
                    
                    with st.expander(session_title):
                        st.write(f"**Iterations:** {history['iterations']}")
                        st.write(f"**Final Score:** {history['final_score']:.1f}/100")
                        st.write(f"**Suggestions Applied:** {history['suggestions_applied']}")
                        if reverted:
                            st.error("🔄 **Data was reverted to original due to low quality score after maximum cleaning attempts.**")
        else:
            st.info("No cleaned data available yet. Please upload and clean a file first.")
    
    with tabs[3]:
        st.markdown("<div class='section-header'>LLM Processing Logs</div>", unsafe_allow_html=True)
        
        if st.session_state.llm_logs:
            for log in st.session_state.llm_logs:
                with st.expander(f"{log['timestamp']} - {log['type']}"):
                    st.write(f"**LLM:** {log['llm']}")
                    st.write(f"**Status:** {log['status']}")
                    if 'prompt' in log:
                        st.text_area("Prompt", log['prompt'], height=100)
                    if 'response' in log:
                        st.text_area("Response", log['response'], height=100)
                    if 'error' in log:
                        st.error(f"Error: {log['error']}")
        else:
            st.info("No LLM logs available yet. Logs will appear here after cleaning operations.")

if __name__ == "__main__":
    main()
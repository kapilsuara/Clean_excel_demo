"""
Simple Header Detection Module
Simple logic: Remove empty rows/columns, then use first row as header
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class SimpleHeaderDetector:
    """
    Simple header detection: 
    1. Remove completely empty rows and columns
    2. Use first row as header
    3. Never remove any other data
    """
    
    def __init__(self):
        pass
    
    def detect_header_row(self, df: pd.DataFrame) -> Tuple[int, pd.DataFrame, List[str]]:
        """
        Simple logic: Remove empty rows/columns, then use first row as header
        
        Returns:
            Tuple of (header_row_index=0, processed_df, detection_log)
        """
        detection_log = []
        original_shape = df.shape
        
        # Step 1: Remove completely empty rows and columns
        df_clean = self._remove_empty_rows_and_columns(df)
        
        if df_clean.shape != original_shape:
            removed_rows = original_shape[0] - df_clean.shape[0]
            removed_cols = original_shape[1] - df_clean.shape[1]
            detection_log.append(f"🧹 Removed {removed_rows} empty rows and {removed_cols} empty columns")
        
        # Step 2: Use first row as header (if data exists)
        if len(df_clean) > 0:
            processed_df = self._process_with_first_row_header(df_clean)
            detection_log.append(f"📋 Using first row as headers, {len(processed_df)} data rows remaining")
            return 0, processed_df, detection_log
        else:
            detection_log.append("⚠️ No data rows available after removing empty rows")
            return -1, df_clean, detection_log
    
    def _remove_empty_rows_and_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ONLY completely empty rows and columns (100% null/empty)"""
        df_result = df.copy()
        
        # Remove completely empty rows
        rows_to_remove = []
        for idx in df_result.index:
            row = df_result.loc[idx]
            # Check if all values are NaN, None, empty string, or just whitespace
            is_empty = True
            for val in row:
                if pd.notna(val) and str(val).strip() != '':
                    is_empty = False
                    break
            
            if is_empty:
                rows_to_remove.append(idx)
        
        if rows_to_remove:
            df_result = df_result.drop(index=rows_to_remove)
            logger.info(f"Removed {len(rows_to_remove)} completely empty rows")
        
        # Remove completely empty columns  
        cols_to_remove = []
        for col in df_result.columns:
            col_data = df_result[col]
            # Check if all values are NaN, None, empty string, or just whitespace
            is_empty = True
            for val in col_data:
                if pd.notna(val) and str(val).strip() != '':
                    is_empty = False
                    break
            
            if is_empty:
                cols_to_remove.append(col)
        
        if cols_to_remove:
            df_result = df_result.drop(columns=cols_to_remove)
            logger.info(f"Removed {len(cols_to_remove)} completely empty columns")
        
        return df_result
    
    def _process_with_first_row_header(self, df: pd.DataFrame) -> pd.DataFrame:
        """Use first row as header and remove it from data"""
        if len(df) == 0:
            return df
        
        # Get first row values for column names
        header_values = df.iloc[0].fillna('').astype(str)
        new_columns = []
        seen = {}
        
        for i, col_val in enumerate(header_values):
            # Create column name from header value
            if col_val.strip() == '' or pd.isna(col_val):
                col_name = f"Column_{i+1}"
            else:
                col_name = str(col_val).strip()
            
            # Handle duplicates
            if col_name in seen:
                seen[col_name] += 1
                col_name = f"{col_name}_{seen[col_name]}"
            else:
                seen[col_name] = 0
            
            new_columns.append(col_name)
        
        # Create new dataframe with first row as headers and remaining rows as data
        if len(df) > 1:
            df_processed = df.iloc[1:].copy()  # Remove first row (now used as headers)
            df_processed.columns = new_columns
            df_processed = df_processed.reset_index(drop=True)
        else:
            # Only one row (header row), create empty dataframe with proper columns
            df_processed = pd.DataFrame(columns=new_columns)
        
        return df_processed

# Backward compatibility
AIHeaderDetector = SimpleHeaderDetector
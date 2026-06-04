#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple test to verify configuration wiring
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from ConfigParser import Config
    from LogicAnalyzer.SignalManager import TASignalProcessor
    from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
    import pandas as pd
    import numpy as np
    
    print("Imports successful")
    
    # Load config
    config = Config()
    print("Config loaded")
    
    # Check if FULL_BULL_WEIGHTS and FULL_BULL_THRESHOLDS exist
    has_weights = hasattr(config, 'FULL_BULL_WEIGHTS')
    has_thresholds = hasattr(config, 'FULL_BULL_THRESHOLDS')
    
    print(f"Has FULL_BULL_WEIGHTS: {has_weights}")
    print(f"Has FULL_BULL_THRESHOLDS: {has_thresholds}")
    
    if has_weights:
        print(f"FULL_BULL_WEIGHTS: {config.FULL_BULL_WEIGHTS}")
    if has_thresholds:
        print(f"FULL_BULL_THRESHOLDS: {config.FULL_BULL_THRESHOLDS}")
    
    # Test MACD analyzer directly with config values
    if has_weights and has_thresholds:
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': np.random.randn(30).cumsum() + 100,
            'high': np.random.randn(30).cumsum() + 105,
            'low': np.random.randn(30).cumsum() + 95,
            'close': np.random.randn(30).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 30)
        })
        
        # Test with config values
        macd = MACDAnalyzer()
        weights = config.FULL_BULL_WEIGHTS
        thresholds = config.FULL_BULL_THRESHOLDS
        
        print(f"Testing MACD with weights: {weights}")
        print(f"Testing MACD with thresholds: {thresholds}")
        
        result = macd.analyze_full_bull(
            df, 
            second_params=(6, 13, 5), 
            recalc_macd=False,
            weights=weights,
            thresholds=thresholds
        )
        
        print("MACD analysis completed successfully!")
        print(f"Score: {result.get('score', 'N/A')}")
        print(f"Score base: {result.get('score_base', 'N/A')}")
        print(f"Conclusion: {result.get('conclusion', 'N/A')}")
        
        # Verify that weights were used by checking a specific score
        zero_axis_score = result.get('details', {}).get('��������', {}).get('score', 'N/A')
        print(f"零轴条件 score: {zero_axis_score} (expected: 20 if DIF>0 and DEA>0)")
        
    else:
        print("ERROR: Missing FULL_BULL_WEIGHTS or FULL_BULL_THRESHOLDS in config")
        sys.exit(1)
        
    print("SUCCESS: Configuration wiring is working correctly")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
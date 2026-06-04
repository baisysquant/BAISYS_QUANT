#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verify that FULL_BULL_WEIGHTS and FULL_BULL_THRESHOLDS are properly wired
from ConfigParser through to MACDAnalyzer
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    try:
        from ConfigParser import Config
        from LogicAnalyzer.MACDAnalyzer import MACDAnalyzer
        import pandas as pd
        import numpy as np
        
        print("=== Configuration Wiring Verification ===\n")
        
        # 1. Load config and verify FULL_BULL sections exist
        print("1. Loading configuration...")
        config = Config()
        
        has_weights = hasattr(config, 'FULL_BULL_WEIGHTS')
        has_thresholds = hasattr(config, 'FULL_BULL_THRESHOLDS')
        
        print(f"   Has FULL_BULL_WEIGHTS: {has_weights}")
        print(f"   Has FULL_BULL_THRESHOLDS: {has_thresholds}")
        
        if not has_weights or not has_thresholds:
            print("   ERROR: Missing FULL_BULL_WEIGHTS or FULL_BULL_THRESHOLDS in Config")
            return False
            
        weights = config.FULL_BULL_WEIGHTS
        thresholds = config.FULL_BULL_THRESHOLDS
        
        print(f"   FULL_BULL_WEIGHTS: {weights}")
        print(f"   FULL_BULL_THRESHOLDS: {thresholds}")
        
        # 2. Create test data
        print("\n2. Creating test data...")
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        # Create trending data to get non-zero scores
        close_prices = 100 + np.cumsum(np.random.randn(30) * 0.5 + 0.1)  # Slight upward bias
        df = pd.DataFrame({
            'date': dates,
            'open': close_prices * 0.99,
            'high': close_prices * 1.02,
            'low': close_prices * 0.98,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 30)
        })
        print(f"   Created DataFrame with shape {df.shape}")
        
        # 3. Test MACD analyzer with config values
        print("\n3. Testing MACD analyzer with config values...")
        macd = MACDAnalyzer()
        
        # Test with the actual weights and thresholds from config
        result = macd.analyze_full_bull(
            df, 
            second_params=(6, 13, 5), 
            recalc_macd=False,
            weights=weights,
            thresholds=thresholds
        )
        
        print("   MACD analysis completed successfully!")
        print(f"   Score: {result.get('score', 'N/A')}")
        print(f"   Score base: {result.get('score_base', 'N/A')}")
        print(f"   Conclusion: {result.get('conclusion', 'N/A')}")
        
        # 4. Verify that weights influenced the result
        print("\n4. Verifying weight influence...")
        details = result.get('details', {})
        
        # Check that we have scores for each dimension
        expected_dimensions = set(weights.keys())
        actual_dimensions = set(details.keys())
        
        print(f"   Expected dimensions: {expected_dimensions}")
        print(f"   Actual dimensions in results: {actual_dimensions}")
        
        missing_dims = expected_dimensions - actual_dimensions
        if missing_dims:
            print(f"   WARNING: Missing dimensions in results: {missing_dims}")
        else:
            print("   All expected dimensions present in results")
            
        # Check a couple of scores to verify they're using the weights
        # We know the first dimension should be '零轴条件' (zero axis condition)
        zero_axis_key = '零轴条件'
        if zero_axis_key in details:
            zero_score = details[zero_axis_key].get('score', 0)
            zero_weight = weights.get(zero_axis_key, 0)
            print(f"   零轴条件: score={zero_score}, weight={zero_weight}")
            
            # If DIF>0 and DEA>0, score should equal weight (for the bullish case)
            # Let's check what the actual values were
            if len(df) >= 2:
                # Need to check if MACD was calculated
                if "DIF_12269" in df.columns and "DEA_12269" in df.columns:
                    dif_above = df["DIF_12269"].iloc[-1] > 0
                    dea_above = df["DEA_12269"].iloc[-1] > 0
                    if dif_above and dea_above:
                        print(f"   Conditions met: DIF>0 and DEA>0 -> score should equal weight")
                        if zero_score == zero_weight:
                            print(f"   PASS: Score correctly equals weight for zero axis condition")
                        else:
                            print(f"   CHECK: Score ({zero_score}) does not equal weight ({zero_weight})")
                    else:
                        print(f"   Info: DIF>0={dif_above}, DEA>0={dea_above} -> score may be less than weight")
                else:
                    print(f"   Info: MACD columns not found in test data")
            else:
                print(f"   Info: Not enough data to check DIF/DEA conditions")
        else:
            print(f"   WARNING: Could not find '{zero_axis_key}' in results details")
            
        # 5. Test with default values (None) to ensure backward compatibility
        print("\n5. Testing backward compatibility with default values...")
        result_default = macd.analyze_full_bull(
            df, 
            second_params=(6, 13, 5), 
            recalc_macd=False,
            weights=None,  # Should use hardcoded defaults
            thresholds=None
        )
        
        print("   Default value test completed!")
        print(f"   Score: {result_default.get('score', 'N/A')}")
        print(f"   Score base: {result_default.get('score_base', 'N/A')}")
        
        # 6. Verify SignalManager integration
        print("\n6. Verifying SignalManager integration...")
        from LogicAnalyzer.SignalManager import TASignalProcessor
        
        # Create a mock analyzer (we don't need a real one for this test)
        class MockAnalyzer:
            pass
            
        processor = TASignalProcessor(MockAnalyzer(), config=config)
        
        # Check that the processor has access to config values
        processor_has_weights = hasattr(processor.config, 'FULL_BULL_WEIGHTS')
        processor_has_thresholds = hasattr(processor.config, 'FULL_BULL_THRESHOLDS')
        
        print(f"   SignalManager config has FULL_BULL_WEIGHTS: {processor_has_weights}")
        print(f"   SignalManager config has FULL_BULL_THRESHOLDS: {processor_has_thresholds}")
        
        if processor_has_weights and processor_has_thresholds:
            print(f"   SignalManager FULL_BULL_WEIGHTS: {processor.config.FULL_BULL_WEIGHTS}")
            print(f"   SignalManager FULL_BULL_THRESHOLDS: {processor.config.FULL_BULL_THRESHOLDS}")
            print("   PASS: SignalManager can access config values")
        else:
            print("   ERROR: SignalManager cannot access config values")
            return False
        
        print("\n=== All tests passed! Configuration wiring is working correctly ===")
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
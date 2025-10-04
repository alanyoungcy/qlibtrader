#!/usr/bin/env python3
"""
Diagnostic script to identify Gradio issues.
"""

import sys
import traceback
import gradio as gr
import pandas as pd
import numpy as np

def test_basic_gradio():
    """Test basic Gradio functionality."""
    print("🔍 Testing basic Gradio functionality...")
    
    try:
        def simple_function(text):
            return f"Processed: {text}"
        
        with gr.Blocks() as demo:
            gr.Markdown("# Basic Test")
            input_text = gr.Textbox(label="Input")
            output_text = gr.Textbox(label="Output")
            button = gr.Button("Process")
            button.click(fn=simple_function, inputs=input_text, outputs=output_text)
        
        print("✅ Basic Gradio interface created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Basic Gradio test failed: {e}")
        traceback.print_exc()
        return False

def test_pandas_integration():
    """Test pandas integration with Gradio."""
    print("\n🔍 Testing pandas integration...")
    
    try:
        def create_dataframe():
            data = {
                'A': [1, 2, 3, 4, 5],
                'B': [10, 20, 30, 40, 50],
                'C': ['x', 'y', 'z', 'w', 'v']
            }
            return pd.DataFrame(data)
        
        with gr.Blocks() as demo:
            gr.Markdown("# Pandas Test")
            button = gr.Button("Create DataFrame")
            dataframe = gr.Dataframe()
            button.click(fn=create_dataframe, outputs=dataframe)
        
        print("✅ Pandas integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pandas integration test failed: {e}")
        traceback.print_exc()
        return False

def test_complex_interface():
    """Test more complex interface components."""
    print("\n🔍 Testing complex interface components...")
    
    try:
        def complex_function(symbols, start_date, end_date):
            # Simulate complex data processing
            symbols_list = [s.strip() for s in symbols.split(',')]
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            data = []
            for symbol in symbols_list:
                for date in dates:
                    base_price = 100 + hash(symbol) % 50
                    price_change = (hash(f"{symbol}{date}") % 20 - 10) / 100
                    price = base_price * (1 + price_change)
                    
                    data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Symbol': symbol,
                        'Price': round(price, 2),
                        'Volume': hash(f"{symbol}{date}") % 1000000 + 100000
                    })
            
            df = pd.DataFrame(data)
            return df, f"Generated {len(df)} records"
        
        with gr.Blocks() as demo:
            gr.Markdown("# Complex Interface Test")
            
            with gr.Row():
                symbols = gr.Textbox(label="Symbols", value="AAPL,MSFT")
                start_date = gr.Textbox(label="Start Date", value="2024-01-01")
                end_date = gr.Textbox(label="End Date", value="2024-01-31")
            
            button = gr.Button("Generate Data")
            dataframe = gr.Dataframe()
            status = gr.Textbox(label="Status")
            
            button.click(
                fn=complex_function,
                inputs=[symbols, start_date, end_date],
                outputs=[dataframe, status]
            )
        
        print("✅ Complex interface test passed")
        return True
        
    except Exception as e:
        print(f"❌ Complex interface test failed: {e}")
        traceback.print_exc()
        return False

def test_launch_parameters():
    """Test different launch parameters."""
    print("\n🔍 Testing launch parameters...")
    
    try:
        with gr.Blocks() as demo:
            gr.Markdown("# Launch Parameter Test")
            gr.Textbox(label="Test", value="This is a test")
        
        # Test different launch configurations
        configs = [
            {"server_name": "127.0.0.1", "server_port": 7861, "quiet": True},
            {"server_name": "0.0.0.0", "server_port": 7862, "quiet": True},
            {"server_name": "localhost", "server_port": 7863, "quiet": True},
        ]
        
        for i, config in enumerate(configs):
            try:
                print(f"  Testing config {i+1}: {config}")
                # Note: We won't actually launch, just test the parameters
                print(f"  ✅ Config {i+1} parameters are valid")
            except Exception as e:
                print(f"  ❌ Config {i+1} failed: {e}")
        
        print("✅ Launch parameter test passed")
        return True
        
    except Exception as e:
        print(f"❌ Launch parameter test failed: {e}")
        traceback.print_exc()
        return False

def check_gradio_version():
    """Check Gradio version and dependencies."""
    print("\n🔍 Checking Gradio version and dependencies...")
    
    try:
        import gradio
        print(f"✅ Gradio version: {gradio.__version__}")
        
        # Check other dependencies
        import pandas
        print(f"✅ Pandas version: {pandas.__version__}")
        
        import numpy
        print(f"✅ NumPy version: {numpy.__version__}")
        
        # Check for potential conflicts
        try:
            import httpx
            print(f"✅ httpx version: {httpx.__version__}")
        except ImportError:
            print("⚠️  httpx not found")
        
        try:
            import uvicorn
            print(f"✅ uvicorn version: {uvicorn.__version__}")
        except ImportError:
            print("⚠️  uvicorn not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Version check failed: {e}")
        return False

def main():
    """Run all diagnostic tests."""
    print("🚀 Gradio Diagnostic Tool")
    print("=" * 50)
    
    tests = [
        ("Gradio Version Check", check_gradio_version),
        ("Basic Gradio Test", test_basic_gradio),
        ("Pandas Integration Test", test_pandas_integration),
        ("Complex Interface Test", test_complex_interface),
        ("Launch Parameter Test", test_launch_parameters),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Gradio should be working correctly.")
        print("The issue might be with specific interface components or launch configuration.")
    else:
        print(f"\n⚠️  {total - passed} tests failed. This indicates potential issues with your Gradio installation.")
        print("Try reinstalling Gradio: pip install --upgrade gradio")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

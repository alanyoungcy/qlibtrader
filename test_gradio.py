#!/usr/bin/env python3
"""
Simple test script for Gradio interface.
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_simple_gradio():
    """Test with a simple Gradio interface."""
    try:
        import gradio as gr
        
        def greet(name):
            return f"Hello {name}!"
        
        # Create a simple interface
        with gr.Blocks() as demo:
            gr.Markdown("# Simple Test Interface")
            name_input = gr.Textbox(label="Name")
            output = gr.Textbox(label="Output")
            greet_btn = gr.Button("Greet")
            greet_btn.click(fn=greet, inputs=name_input, outputs=output)
        
        print("✅ Simple Gradio interface created successfully")
        return demo
        
    except Exception as e:
        print(f"❌ Error creating simple Gradio interface: {e}")
        return None

def test_trading_ui():
    """Test the trading system UI."""
    try:
        from ui.gradio_app import TradingSystemUI
        
        print("Creating TradingSystemUI...")
        ui = TradingSystemUI()
        print("✅ TradingSystemUI created successfully")
        
        print("Creating interface...")
        interface = ui.create_interface()
        print("✅ Interface created successfully")
        
        return interface
        
    except Exception as e:
        print(f"❌ Error creating trading UI: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function."""
    print("Testing Gradio interfaces...")
    
    # Test simple interface
    print("\n1. Testing simple Gradio interface...")
    simple_demo = test_simple_gradio()
    
    # Test trading UI
    print("\n2. Testing trading system UI...")
    trading_demo = test_trading_ui()
    
    if simple_demo and trading_demo:
        print("\n✅ All tests passed!")
        print("\nTo launch the trading system, use:")
        print("python start_trading_system.py --mode gradio")
    else:
        print("\n❌ Some tests failed!")

if __name__ == "__main__":
    main()

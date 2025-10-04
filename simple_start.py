#!/usr/bin/env python3
"""
Simple startup script for the Trading System with minimal Gradio interface.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/simple_start.log')
        ]
    )
    return logging.getLogger(__name__)

def create_simple_interface():
    """Create a simple Gradio interface for testing."""
    try:
        import gradio as gr
        
        def greet(name):
            return f"Hello {name}! Welcome to the Trading System."
        
        def get_status():
            return "Trading System is running successfully!"
        
        with gr.Blocks(title="Trading System - Simple Interface") as demo:
            gr.Markdown("# 🚀 AI-Powered Trading System")
            gr.Markdown("Simple interface for testing the system.")
            
            with gr.Row():
                with gr.Column():
                    name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
                    greet_btn = gr.Button("Greet", variant="primary")
                    output = gr.Textbox(label="Output", interactive=False)
                    
                with gr.Column():
                    status_btn = gr.Button("Check Status", variant="secondary")
                    status_output = gr.Textbox(label="System Status", interactive=False)
            
            greet_btn.click(fn=greet, inputs=name_input, outputs=output)
            status_btn.click(fn=get_status, outputs=status_output)
            
            gr.Markdown("""
            ## Next Steps
            1. This is a simplified interface for testing
            2. The full interface will be available once all components are working
            3. Check the logs for any issues
            """)
        
        return demo
        
    except Exception as e:
        print(f"Error creating interface: {e}")
        return None

def main():
    """Main function."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting simple trading system interface")
    
    try:
        # Check dependencies
        import gradio as gr
        import pandas as pd
        import numpy as np
        logger.info("All required packages are available")
        
        # Create interface
        interface = create_simple_interface()
        if interface is None:
            logger.error("Failed to create interface")
            return 1
        
        logger.info("Interface created successfully")
        
        # Launch interface
        print("\n🌐 Starting Trading System Interface")
        print("   Host: 127.0.0.1")
        print("   Port: 7860")
        print("\n📱 Open your browser and navigate to:")
        print("   http://127.0.0.1:7860")
        print("\n💡 Press Ctrl+C to stop the server")
        
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=False
        )
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Trading System stopped by user")
        return 0
        
    except Exception as e:
        logger.error(f"Error running Trading System: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Minimal working Gradio app for testing.
This is the most basic possible version to isolate the issue.
"""

import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def greet(name):
    """Simple greeting function."""
    return f"Hello {name}! Welcome to the Trading System."

def get_system_status():
    """Get basic system status."""
    return "✅ System is running normally"

def generate_sample_data(symbols, start_date, end_date):
    """Generate sample trading data."""
    try:
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
                    'Open': round(price * 0.99, 2),
                    'High': round(price * 1.02, 2),
                    'Low': round(price * 0.98, 2),
                    'Close': round(price, 2),
                    'Volume': hash(f"{symbol}{date}") % 1000000 + 100000
                })
        
        df = pd.DataFrame(data)
        return df, f"Generated {len(df)} records for {len(symbols_list)} symbols"
        
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def run_simple_backtest(initial_capital, commission_rate):
    """Run a simple backtest simulation."""
    try:
        # Simple simulation
        final_capital = initial_capital * (1 + np.random.normal(0.05, 0.1))  # Random return
        total_return = (final_capital - initial_capital) / initial_capital
        
        results = [
            ["Initial Capital", f"${initial_capital:,.2f}"],
            ["Final Capital", f"${final_capital:,.2f}"],
            ["Total Return", f"{total_return:.2%}"],
            ["Commission Rate", f"{commission_rate:.2%}"],
            ["Net Return", f"{(total_return - commission_rate):.2%}"]
        ]
        
        return pd.DataFrame(results, columns=["Metric", "Value"]), f"Backtest completed with {total_return:.2%} return"
        
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}"

def create_interface():
    """Create the minimal Gradio interface."""
    with gr.Blocks(title="Minimal Trading System") as demo:
        gr.Markdown("# 🚀 Minimal Trading System")
        gr.Markdown("A basic trading system interface for testing.")
        
        with gr.Tabs():
            # Tab 1: Welcome
            with gr.Tab("Welcome"):
                gr.Markdown("## Welcome to the Trading System")
                
                name_input = gr.Textbox(label="Your Name", placeholder="Enter your name")
                greet_btn = gr.Button("Greet", variant="primary")
                greet_output = gr.Textbox(label="Greeting", interactive=False)
                
                status_btn = gr.Button("Check Status", variant="secondary")
                status_output = gr.Textbox(label="System Status", interactive=False)
                
                greet_btn.click(fn=greet, inputs=name_input, outputs=greet_output)
                status_btn.click(fn=get_system_status, outputs=status_output)
            
            # Tab 2: Data
            with gr.Tab("Data"):
                gr.Markdown("## Sample Data Generation")
                
                with gr.Row():
                    symbols_input = gr.Textbox(
                        label="Symbols (comma-separated)",
                        value="AAPL,MSFT,GOOGL",
                        placeholder="AAPL,MSFT,GOOGL"
                    )
                
                with gr.Row():
                    start_date = gr.Textbox(
                        label="Start Date",
                        value=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                    )
                    end_date = gr.Textbox(
                        label="End Date",
                        value=datetime.now().strftime("%Y-%m-%d")
                    )
                
                generate_btn = gr.Button("Generate Data", variant="primary")
                data_display = gr.Dataframe(label="Sample Data")
                data_status = gr.Textbox(label="Status", interactive=False)
                
                generate_btn.click(
                    fn=generate_sample_data,
                    inputs=[symbols_input, start_date, end_date],
                    outputs=[data_display, data_status]
                )
            
            # Tab 3: Backtest
            with gr.Tab("Backtest"):
                gr.Markdown("## Simple Backtest")
                
                with gr.Row():
                    capital_input = gr.Number(
                        label="Initial Capital ($)",
                        value=100000,
                        minimum=1000,
                        maximum=1000000
                    )
                    commission_input = gr.Slider(
                        label="Commission Rate (%)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.1,
                        step=0.01
                    )
                
                backtest_btn = gr.Button("Run Backtest", variant="primary")
                results_display = gr.Dataframe(label="Results")
                backtest_status = gr.Textbox(label="Status", interactive=False)
                
                backtest_btn.click(
                    fn=run_simple_backtest,
                    inputs=[capital_input, commission_input],
                    outputs=[results_display, backtest_status]
                )
        
        gr.Markdown("---")
        gr.Markdown("**Minimal Trading System** - Basic functionality test")
    
    return demo

def main():
    """Main function."""
    print("🚀 Starting Minimal Trading System...")
    print("This is a basic version to test Gradio functionality.")
    
    try:
        # Create interface
        interface = create_interface()
        
        print("\n🌐 Interface created successfully")
        print("📱 The interface will open in your browser")
        print("💡 Press Ctrl+C to stop the server")
        
        # Launch with minimal configuration
        interface.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=False
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

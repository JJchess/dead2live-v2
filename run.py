"""Launch the dead2live web UI.

    python run.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
from dead2live.app import build_ui

if __name__ == "__main__":
    build_ui().launch(server_name="127.0.0.1", server_port=7860,
                      inbrowser=True, theme=gr.themes.Soft())

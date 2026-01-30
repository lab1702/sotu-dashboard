"""
Loading indicator utilities for Presidential Speech Dashboard.
Provides prominent busy indicators for long-running operations.
"""

import streamlit as st
from contextlib import contextmanager


def show_loading_overlay(message: str = "Processing..."):
    """
    Display a prominent full-page loading overlay.

    Call this at the start of a long operation, then call hide_loading_overlay() when done.
    """
    overlay_css = """
    <style>
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background-color: rgba(0, 0, 0, 0.7);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 9999;
    }
    .loading-spinner {
        width: 80px;
        height: 80px;
        border: 8px solid #f3f3f3;
        border-top: 8px solid #0072B2;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    .loading-text {
        color: white;
        font-size: 24px;
        margin-top: 20px;
        font-weight: 500;
    }
    .loading-subtext {
        color: #cccccc;
        font-size: 14px;
        margin-top: 8px;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """

    overlay_html = f"""
    {overlay_css}
    <div class="loading-overlay" id="loading-overlay">
        <div class="loading-spinner"></div>
        <div class="loading-text">{message}</div>
        <div class="loading-subtext">This may take a moment for large datasets</div>
    </div>
    """

    return st.markdown(overlay_html, unsafe_allow_html=True)


@contextmanager
def loading_indicator(message: str = "Processing...", container=None):
    """
    Context manager for showing a prominent loading indicator.

    Usage:
        with loading_indicator("Analyzing emotions..."):
            # long running operation
            result = analyze_data()

    Args:
        message: Message to display during loading
        container: Optional st container to use (defaults to creating new one)
    """
    if container is None:
        container = st.empty()

    # Show the loading overlay
    with container:
        overlay_css = """
        <style>
        .loading-box {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 16px;
            padding: 40px 60px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            margin: 40px auto;
            max-width: 500px;
        }
        .loading-spinner-large {
            width: 60px;
            height: 60px;
            border: 5px solid rgba(255, 255, 255, 0.1);
            border-top: 5px solid #0072B2;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 20px auto;
        }
        .loading-message {
            color: white;
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 8px;
        }
        .loading-hint {
            color: #888;
            font-size: 14px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        """

        st.markdown(f"""
        {overlay_css}
        <div class="loading-box">
            <div class="loading-spinner-large"></div>
            <div class="loading-message">{message}</div>
            <div class="loading-hint">Processing full speech texts...</div>
        </div>
        """, unsafe_allow_html=True)

    try:
        yield
    finally:
        # Clear the loading indicator
        container.empty()


def progress_indicator(message: str, current: int, total: int):
    """
    Show a progress bar with message.

    Args:
        message: Status message
        current: Current item number
        total: Total items
    """
    progress = current / total if total > 0 else 0

    st.markdown(f"""
    <div style="
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    ">
        <div style="color: white; font-size: 16px; margin-bottom: 10px;">
            {message}
        </div>
        <div style="
            background: #333;
            border-radius: 5px;
            height: 20px;
            overflow: hidden;
        ">
            <div style="
                background: linear-gradient(90deg, #0072B2, #00a8e8);
                height: 100%;
                width: {progress * 100}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="color: #888; font-size: 12px; margin-top: 8px;">
            {current} of {total} ({progress * 100:.0f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)


class StatusIndicator:
    """
    A prominent status indicator that can be updated during long operations.

    Usage:
        status = StatusIndicator("Analyzing speeches")
        status.update("Processing speech 1/100...")
        status.update("Processing speech 2/100...")
        status.complete("Analysis complete!")
    """

    def __init__(self, title: str):
        self.title = title
        self.container = st.empty()
        self._show_initial()

    def _show_initial(self):
        with self.container:
            st.markdown(f"""
            <style>
            .status-box {{
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                padding: 24px 32px;
                margin: 20px 0;
                border-left: 4px solid #0072B2;
            }}
            .status-title {{
                color: white;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .status-spinner {{
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255, 255, 255, 0.2);
                border-top: 3px solid #0072B2;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }}
            .status-message {{
                color: #aaa;
                font-size: 14px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            <div class="status-box">
                <div class="status-title">
                    <div class="status-spinner"></div>
                    {self.title}
                </div>
                <div class="status-message">Starting...</div>
            </div>
            """, unsafe_allow_html=True)

    def update(self, message: str):
        """Update the status message."""
        with self.container:
            st.markdown(f"""
            <style>
            .status-box {{
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                padding: 24px 32px;
                margin: 20px 0;
                border-left: 4px solid #0072B2;
            }}
            .status-title {{
                color: white;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .status-spinner {{
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255, 255, 255, 0.2);
                border-top: 3px solid #0072B2;
                border-radius: 50%;
                animation: spin 0.8s linear infinite;
            }}
            .status-message {{
                color: #aaa;
                font-size: 14px;
            }}
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
            </style>
            <div class="status-box">
                <div class="status-title">
                    <div class="status-spinner"></div>
                    {self.title}
                </div>
                <div class="status-message">{message}</div>
            </div>
            """, unsafe_allow_html=True)

    def complete(self, message: str = "Complete!"):
        """Mark the operation as complete."""
        with self.container:
            st.markdown(f"""
            <style>
            .status-box-complete {{
                background: linear-gradient(135deg, #1a2e1a 0%, #162e16 100%);
                border-radius: 12px;
                padding: 24px 32px;
                margin: 20px 0;
                border-left: 4px solid #009E73;
            }}
            .status-title-complete {{
                color: white;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .status-check {{
                color: #009E73;
                font-size: 20px;
            }}
            .status-message-complete {{
                color: #009E73;
                font-size: 14px;
            }}
            </style>
            <div class="status-box-complete">
                <div class="status-title-complete">
                    <span class="status-check">✓</span>
                    {self.title}
                </div>
                <div class="status-message-complete">{message}</div>
            </div>
            """, unsafe_allow_html=True)

    def clear(self):
        """Clear the status indicator."""
        self.container.empty()

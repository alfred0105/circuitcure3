"""Opaque, app-wide loading screen for synchronous PCB processing."""

from __future__ import annotations

from html import escape
from typing import Any


def show_fullscreen_loading(st: Any, message: str) -> Any:
    """Cover the current Streamlit page until the returned placeholder is cleared."""
    placeholder = st.empty()
    placeholder.markdown(
        """
        <style>
          .cc-fullscreen-loading {
            position: fixed; inset: 0; z-index: 2147483000;
            display: grid; place-items: center; background: #0b1119;
            color: #edf3f8; font-family: sans-serif; text-align: center;
          }
          .cc-fullscreen-loading-card { width: min(92vw, 440px); padding: 2rem; }
          .cc-fullscreen-loading-spinner {
            width: 52px; height: 52px; margin: 0 auto 1.5rem;
            border: 5px solid #2a3b4d; border-top-color: #52a8ff;
            border-radius: 50%; animation: cc-loader-spin .8s linear infinite;
          }
          .cc-fullscreen-loading h2 { margin: 0 0 .6rem; font-size: 1.35rem; }
          .cc-fullscreen-loading p { margin: 0; color: #a8bacb; }
          @keyframes cc-loader-spin { to { transform: rotate(360deg); } }
        </style>
        """ + f'<div class="cc-fullscreen-loading"><div class="cc-fullscreen-loading-card">'
        f'<div class="cc-fullscreen-loading-spinner"></div><h2>잠시만 기다려 주세요</h2>'
        f'<p>{escape(message)}</p></div></div>',
        unsafe_allow_html=True,
    )
    return placeholder

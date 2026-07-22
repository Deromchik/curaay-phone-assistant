"""Git metadata helpers for the Streamlit prompt tester."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import streamlit as st


def get_git_head(repo_dir: Path) -> Optional[dict[str, str]]:
    """Return short commit hash and last commit timestamp for HEAD."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        committed_at = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci", "HEAD"],
            cwd=repo_dir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not commit:
            return None
        return {"commit": commit, "committed_at": committed_at}
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def init_git_session_state(repo_dir: Path) -> None:
    """Capture git HEAD and app start time once per Streamlit server session."""
    if "app_started_at" not in st.session_state:
        st.session_state.app_started_at = datetime.now(timezone.utc)
    if "loaded_git_head" not in st.session_state:
        st.session_state.loaded_git_head = get_git_head(repo_dir)


def _format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def render_git_sync_indicator(repo_dir: Path) -> None:
    """Show whether the running app matches the git commit on disk."""
    init_git_session_state(repo_dir)

    disk_head: Optional[dict[str, str]] = get_git_head(repo_dir)
    loaded_head: Optional[dict[str, str]] = st.session_state.loaded_git_head
    app_started: datetime = st.session_state.app_started_at

    st.markdown("### Git status")

    if not disk_head or not loaded_head:
        st.caption("Git metadata unavailable (not a git repo or git not installed).")
        st.caption(f"App started: {_format_utc(app_started)}")
        return

    synced = disk_head["commit"] == loaded_head["commit"]
    if synced:
        st.success("Git synced")
    else:
        st.warning("New git commit on disk — restart Streamlit")
        st.caption(
            f"Running: `{loaded_head['commit']}` · "
            f"On disk: `{disk_head['commit']}`"
        )

    st.caption(f"Commit: `{disk_head['commit']}`")
    st.caption(f"Committed: {disk_head['committed_at']}")
    st.caption(f"App started: {_format_utc(app_started)}")

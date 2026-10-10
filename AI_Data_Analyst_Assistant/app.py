"""Streamlit interface for the local AI Data Analyst Assistant."""

from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from data_analyzer import (
    build_cleaning_preview,
    describe_dataset,
    detect_suspicious_columns,
    find_date_columns,
    load_uploaded_file,
)
from llm_assistant import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaError,
    analyze_question,
    validate_ollama_url,
)


st.set_page_config(
    page_title="AI Data Analyst Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7fb;
        color: #1f2937;
        --text-color: #1f2937;
        --secondary-text-color: #475569;
    }
    [data-testid="stMain"],
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label,
    [data-testid="stMain"] li,
    [data-testid="stMain"] [data-testid="stMarkdownContainer"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stCaptionContainer"] {
        color: #1f2937 !important;
    }
    [data-testid="stSidebar"] { background: #101b32; }
    [data-testid="stSidebar"] * { color: #f2f5fb; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid #e7ebf2;
        padding: 18px 20px; border-radius: 14px;
    }
    [data-testid="stMetricLabel"] { color: #475569; }
    .hero { padding: 1.25rem 0 .5rem; }
    .muted { color: #64748b; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_upload(upload: Any) -> None:
    """Load a new upload and clear results tied to the previous dataset."""
    if upload is None:
        return
    fingerprint = hashlib.sha256(upload.name.encode("utf-8") + upload.getvalue()).hexdigest()
    if fingerprint == st.session_state.get("upload_fingerprint"):
        upload_error = st.session_state.get("upload_error")
        if upload_error:
            st.sidebar.error(upload_error)
        return
    try:
        frame = load_uploaded_file(upload)
    except ValueError as exc:
        st.session_state.pop("dataset", None)
        st.session_state.pop("dataset_name", None)
        st.session_state.analysis_result = None
        st.session_state.chat_history = []
        st.session_state.upload_fingerprint = fingerprint
        st.session_state.upload_error = str(exc)
        st.sidebar.error(st.session_state.upload_error)
        return
    st.session_state.dataset = frame
    st.session_state.dataset_name = upload.name
    st.session_state.upload_fingerprint = fingerprint
    st.session_state.pop("upload_error", None)
    st.session_state.analysis_result = None
    st.session_state.chat_history = []


def _numeric_columns(frame: pd.DataFrame) -> List[str]:
    return frame.select_dtypes(include="number").columns.tolist()


def _display_overview(frame: pd.DataFrame, dataset_name: str) -> None:
    profile = describe_dataset(frame)
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("Your data, understood.")
    st.caption("A private workspace for exploring data and finding useful signals.")
    st.markdown("</div>", unsafe_allow_html=True)

    kpis = st.columns(4)
    kpis[0].metric("Rows", f"{profile['row_count']:,}")
    kpis[1].metric("Columns", f"{profile['column_count']:,}")
    kpis[2].metric("Missing cells", f"{profile['missing_cells']:,}")
    kpis[3].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

    st.subheader(f"Dataset preview · {dataset_name}")
    st.dataframe(frame.head(1000), use_container_width=True, height=380)
    st.caption("Showing up to the first 1,000 rows.")

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Column profile")
        st.dataframe(profile["columns"], use_container_width=True, hide_index=True)
    with right:
        st.subheader("Numeric summary")
        stats = profile["numeric_summary"]
        if stats.empty:
            st.info("No numeric columns were detected.")
        else:
            st.dataframe(stats, use_container_width=True)


def _display_quality(frame: pd.DataFrame) -> None:
    st.header("Data quality & cleaning")
    profile = describe_dataset(frame)
    st.subheader("Quality checks")
    st.dataframe(profile["columns"], use_container_width=True, hide_index=True)
    suspicious = detect_suspicious_columns(frame)
    if suspicious:
        st.warning(
            "Possible type mismatches: "
            + ", ".join(f"{column} → {kind}" for column, kind in suspicious.items())
        )
    else:
        st.success("No likely numeric or date columns stored as text were detected.")

    st.subheader("Propose cleaning changes")
    st.caption("Preview the proposed result first. Nothing changes until you approve it.")
    remove_duplicates = st.checkbox("Remove duplicate rows")
    fill_missing = st.checkbox("Fill missing values (numeric median, text mode)")
    convert_options = [
        f"{column} → {kind}" for column, kind in suspicious.items()
    ]
    selected_conversions = st.multiselect(
        "Convert suspicious columns", convert_options
    )
    convert_columns = [item.split(" → ", 1)[0] for item in selected_conversions]
    proposal, changes = build_cleaning_preview(
        frame,
        remove_duplicates=remove_duplicates,
        fill_missing=fill_missing,
        convert_columns=convert_columns,
    )
    if changes:
        st.write("Proposed changes")
        st.dataframe(
            pd.DataFrame(changes, columns=["Change", "Details"]),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(proposal.head(1000), use_container_width=True)
        if st.button("Approve and apply changes", type="primary"):
            st.session_state.dataset = proposal
            st.session_state.dataset_name = (
                st.session_state.dataset_name.rsplit(".", 1)[0] + "_cleaned.csv"
            )
            st.session_state.analysis_result = None
            st.session_state.chat_history = []
            st.success("Approved changes have been applied to the working dataset.")
            st.rerun()
    else:
        st.info("Choose one or more cleaning actions to preview them.")

    st.download_button(
        "Download current dataset as CSV",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )


def _display_dashboard(frame: pd.DataFrame) -> None:
    st.header("Interactive dashboard")
    numeric = _numeric_columns(frame)
    if not numeric or len(frame.columns) < 2:
        st.info("At least two columns, including one numeric column, are needed to chart.")
        return

    filtered = frame.copy()
    date_columns = find_date_columns(frame)
    with st.expander("Filter data", expanded=True):
        category_columns = [
            col for col in frame.columns if col not in numeric and col not in date_columns
        ]
        filter_col = st.selectbox(
            "Category filter", ["(none)"] + category_columns, key="dashboard_filter"
        )
        if filter_col != "(none)":
            values = frame[filter_col].dropna().unique().tolist()
            selected_values = st.multiselect(
                "Keep values",
                values,
                default=values,
                key="dashboard_values",
            )
            filtered = filtered[filtered[filter_col].isin(selected_values)]

        if date_columns:
            date_col = st.selectbox("Date range column", ["(none)"] + date_columns)
            if date_col != "(none)":
                parsed = pd.to_datetime(filtered[date_col], errors="coerce")
                valid = parsed.dropna()
                if not valid.empty:
                    bounds = (valid.min().date(), valid.max().date())
                    if bounds[0] < bounds[1]:
                        selected_range = st.date_input(
                            "Date range", value=bounds, key="dashboard_dates"
                        )
                        if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
                            start_date, end_date = selected_range
                            filtered = filtered[
                                parsed.dt.date.between(start_date, end_date)
                            ]
                        elif isinstance(selected_range, (tuple, list)) and len(selected_range) == 1:
                            filtered = filtered[parsed.dt.date >= selected_range[0]]
                        elif isinstance(selected_range, date):
                            filtered = filtered[parsed.dt.date == selected_range]
                    else:
                        filtered = filtered[parsed.dt.date == bounds[0]]

    chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter"])
    columns = frame.columns.tolist()
    default_x = next((c for c in columns if c not in numeric), columns[0])
    x_col = st.selectbox("X axis", columns, index=columns.index(default_x))
    y_col = st.selectbox("Y axis", numeric, index=0)

    if filtered.empty:
        st.warning("No rows match the selected filters.")
        return
    if chart_type == "Scatter" and x_col not in numeric:
        st.warning("Scatter charts require a numeric X axis.")
        return
    chart_data = filtered.copy()
    if x_col in date_columns:
        chart_data[x_col] = pd.to_datetime(chart_data[x_col], errors="coerce")
        chart_data = chart_data.dropna(subset=[x_col]).sort_values(x_col)
    chart_func = {"Bar": px.bar, "Line": px.line, "Scatter": px.scatter}[chart_type]
    fig = chart_func(chart_data, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    fig.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)


def _display_analyst(frame: pd.DataFrame) -> None:
    st.header("Ask your data")
    st.caption(
        "The local model selects a supported analysis. Pandas calculates every result; "
        "the model only explains those verified results."
    )
    with st.form("analyst_question"):
        question = st.text_input(
            "What would you like to know?",
            placeholder="Which region generated the highest revenue?",
            max_chars=1000,
        )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question before starting an analysis.")
        else:
            with st.spinner("Interpreting the question and calculating results locally..."):
                try:
                    result = analyze_question(
                        question,
                        frame,
                        model=os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                        base_url=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL),
                    )
                    st.session_state.analysis_result = result
                    st.session_state.chat_history.append(
                        {"question": question, "result": result}
                    )
                except (OllamaError, ValueError) as exc:
                    st.error(str(exc))

    for item in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**You:** {item['question']}")
            st.markdown(f"**Model interpretation:** {item['result']['explanation']}")
            table = item["result"]["table"]
            if not table.empty:
                st.markdown("**Pandas-verified results:**")
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download analysis table",
                    data=table.to_csv(index=False).encode("utf-8"),
                    file_name="analysis_results.csv",
                    mime="text/csv",
                    key=f"analysis_csv_{id(item)}",
                )
            st.caption(f"Verified operation: {item['result']['operation']}")


def _business_summary(frame: pd.DataFrame, result: Optional[Dict[str, Any]]) -> str:
    profile = describe_dataset(frame)
    lines = [
        "AI Data Analyst Assistant — Business Summary",
        f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "Dataset overview",
        f"- Rows: {profile['row_count']:,}",
        f"- Columns: {profile['column_count']:,}",
        f"- Missing cells: {profile['missing_cells']:,}",
        f"- Duplicate rows: {profile['duplicate_rows']:,}",
        "",
        "Observed data quality facts",
    ]
    if profile["missing_cells"]:
        lines.append(f"- {profile['missing_cells']:,} cells are missing.")
    else:
        lines.append("- No missing cells were found.")
    if profile["duplicate_rows"]:
        lines.append(f"- {profile['duplicate_rows']:,} duplicate rows were found.")
    else:
        lines.append("- No duplicate rows were found.")
    if result:
        lines.extend(
            [
                "",
                "Latest verified analysis",
                f"- Question: {result['question']}",
                f"- Model interpretation (not a proven cause): {result['explanation']}",
                "- Numerical results below were calculated with Pandas.",
                result["table"].to_string(index=False) if not result["table"].empty else "",
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation note",
            "Observed facts describe this dataset only. Any possible business explanation "
            "is a hypothesis and should be checked against additional context.",
        ]
    )
    return "\n".join(lines)


def _display_insights(frame: pd.DataFrame) -> None:
    st.header("Business insights & export")
    numeric = _numeric_columns(frame)
    if numeric:
        columns = st.columns(min(len(numeric), 4))
        for index, column in enumerate(numeric[:4]):
            values = frame[column].dropna()
            if not values.empty:
                columns[index].metric(
                    f"Average {column}", f"{values.mean():,.2f}"
                )
    else:
        st.info("No numeric columns are available for summary metrics.")

    st.markdown(
        "**Observed facts** are calculated from the uploaded rows. "
        "**Possible explanations** are hypotheses and are not proven by the data alone."
    )
    latest = st.session_state.analysis_result
    if latest:
        st.write("**Model interpretation (not a proven cause):**", latest["explanation"])
        if not latest["table"].empty:
            st.write("**Observed, Pandas-verified result:**")
            st.dataframe(latest["table"], use_container_width=True, hide_index=True)
    else:
        st.info("Run an analysis in the AI Analyst section to include its result here.")

    st.download_button(
        "Download business summary report",
        data=_business_summary(frame, latest).encode("utf-8"),
        file_name="business_summary.txt",
        mime="text/plain",
    )


def main() -> None:
    st.sidebar.title("📊 Data Analyst")
    st.sidebar.caption("Your data stays on this machine.")
    uploaded = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        help="Supported formats: .csv and .xlsx",
    )
    _load_upload(uploaded)

    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Data Quality & Cleaning",
            "Interactive Dashboard",
            "AI Analyst",
            "Insights & Export",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption(
        f"Ollama model: {os.environ.get('OLLAMA_MODEL', DEFAULT_OLLAMA_MODEL)}"
    )
    try:
        validate_ollama_url(os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL))
        st.sidebar.caption("Ollama endpoint: local only")
    except ValueError as exc:
        st.sidebar.error(str(exc))

    frame = st.session_state.get("dataset")
    if frame is None:
        st.title("AI Data Analyst Assistant")
        st.write("Upload a CSV or Excel workbook in the sidebar to get started.")
        st.info(
            "For question analysis, start Ollama locally and configure a model. "
            "The dashboard and data profiling work without Ollama."
        )
        return

    dataset_name = st.session_state.get("dataset_name", "Uploaded dataset")
    if page == "Overview":
        _display_overview(frame, dataset_name)
    elif page == "Data Quality & Cleaning":
        _display_quality(frame)
    elif page == "Interactive Dashboard":
        _display_dashboard(frame)
    elif page == "AI Analyst":
        _display_analyst(frame)
    else:
        _display_insights(frame)


if __name__ == "__main__":
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    main()

"""LLM Data Analyst: a Streamlit app that lets users analyse a CSV by asking
questions in plain English.

Flow: upload CSV -> profile the data -> user asks a question -> LLM writes an
analysis plan and Python code -> code is checked and executed -> results are
shown as tables/charts -> LLM explains the results -> sources are listed.

Run with:  streamlit run app.py
"""

import ast
import contextlib
import io
import os
import re
import traceback
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()

# Model names change over time. Override with GEMINI_MODEL in the .env file
# if you see a "model not found" error.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
MAX_RETRIES = 2  # how many times the LLM may try to fix failing code

SYSTEM_PROMPT = """You are a data analysis assistant. You will be given a profile of a
pandas DataFrame called `df` and a question from the user.

Respond in exactly this format:
1. A short numbered analysis plan (2-5 steps) in plain text.
2. One Python code block that answers the question. Rules for the code:
   - The DataFrame is already loaded as `df`. Do not read any files.
   - Use only pandas, numpy, scipy, matplotlib.pyplot (as plt) and plotly.express (as px).
   - Do not import statsmodels or sklearn. For trendlines, use numpy.polyfit instead.
   - Store any table to display in a variable named `result_table` (a DataFrame).
   - Store any chart in a variable named `fig`.
   - Do not use input(), network calls, or write to disk.
   - Only use column names that appear in the profile."""

# ---------------------------------------------------------------------------
# Safety checks for LLM-generated code
# (This is a basic filter, not a true sandbox. See README for limitations.)
# ---------------------------------------------------------------------------
ALLOWED_MODULES = {"pandas", "numpy", "scipy", "matplotlib", "plotly", "math", "statistics"}
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
}
BLOCKED_ATTRS = {
    "to_csv", "to_excel", "to_pickle", "to_json", "to_parquet", "to_sql",
    "read_csv", "read_excel", "read_pickle", "read_json", "read_parquet",
    "read_sql", "savefig", "write_html", "write_image", "system", "popen",
}


def validate_code(code: str) -> None:
    """Raise ValueError if the code uses anything we don't allow."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES:
                    raise ValueError(f"Import not allowed: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_MODULES:
                raise ValueError(f"Import not allowed: {node.module}")
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr in BLOCKED_ATTRS or node.attr.startswith("__"):
                raise ValueError(f"Not allowed: .{node.attr}")


def run_code(code: str, df: pd.DataFrame):
    """Validate and run generated code. Returns (result_table, fig, printed_text)."""
    validate_code(code)
    plt.close("all")
    namespace = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "px": px}
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(code, namespace)

    table = namespace.get("result_table")
    fig = namespace.get("fig")
    if fig is None and plt.get_fignums():
        fig = plt.gcf()
    return table, fig, buffer.getvalue()


# ---------------------------------------------------------------------------
# Data profiling and source information
# ---------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame) -> str:
    """Build a compact text summary of the dataset to give to the LLM."""
    lines = [f"Rows: {len(df)}, Columns: {df.shape[1]}", "", "Columns:"]
    for col in df.columns:
        s = df[col]
        info = f"- {col} (dtype: {s.dtype}, missing: {s.isna().sum()}, unique: {s.nunique()})"
        if pd.api.types.is_numeric_dtype(s):
            info += f", min={s.min()}, max={s.max()}, mean={s.mean():.2f}"
        lines.append(info)
    lines += ["", "Sample rows:", df.head(5).to_string(index=False)]
    return "\n".join(lines)


def columns_used(code: str, df: pd.DataFrame) -> list:
    """Which dataset columns are mentioned in the generated code."""
    return [c for c in df.columns if re.search(rf"\b{re.escape(str(c))}\b", code)]


def libraries_used(code: str) -> list:
    """Which libraries the generated code imports."""
    mods = set()
    try:
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
    except SyntaxError:
        pass
    return sorted(mods)


# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------
def generate_plan_and_code(client, question, profile, previous_code=None, error=None):
    """Ask the LLM for a plan and code. If previous_code and error are given,
    ask it to fix the failing code. Returns (plan_text, code_text)."""
    contents = f"Dataset profile:\n{profile}\n\nQuestion: {question}"
    if previous_code and error:
        contents += (
            "\n\nYour previous code failed.\n"
            f"Code:\n```python\n{previous_code}\n```\n"
            f"Error: {error}\n"
            "Return the same format again: a short plan, then one corrected code block."
        )
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    text = response.text or ""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    code = match.group(1).strip() if match else ""
    plan = text[: match.start()].strip() if match else text
    return plan, code


def explain_results(client, question: str, table, printed: str) -> str:
    """Ask the LLM to explain the results in plain language."""
    parts = []
    if table is not None:
        parts.append("Result table:\n" + table.to_string()[:3000])
    if printed.strip():
        parts.append("Printed output:\n" + printed[:2000])
    results_text = "\n\n".join(parts) or "No table or text output (only a chart)."

    prompt = (
        f"The user asked: {question}\n\n{results_text}\n\n"
        "Explain these results in plain language for a non-technical reader, "
        "in 3-5 sentences. State which columns were used. Mention one limitation "
        "or caution (for example: correlation is not causation, or a small sample). "
        "Do not invent numbers that are not shown above."
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text


def run_with_retries(client, question, profile, df, plan, code):
    """Run the generated code. If it fails, send the error back to the LLM and
    retry up to MAX_RETRIES times. Returns a dict describing the outcome."""
    errors = []
    for attempt in range(MAX_RETRIES + 1):
        if not code:
            error_msg = "The LLM did not return any code."
            errors.append(error_msg)
        else:
            try:
                table, fig, printed = run_code(code, df)
                return {
                    "success": True, "plan": plan, "code": code, "table": table,
                    "fig": fig, "printed": printed, "errors": errors,
                }
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                errors.append(error_msg)

        if attempt < MAX_RETRIES:
            with st.spinner(
                f"The code failed. Asking the LLM to fix it "
                f"(attempt {attempt + 2} of {MAX_RETRIES + 1})..."
            ):
                try:
                    plan, code = generate_plan_and_code(
                        client, question, profile, code, error_msg
                    )
                except Exception as e:
                    st.error(f"LLM request failed while fixing the code: {e}")
                    break

    return {
        "success": False, "plan": plan, "code": code, "table": None,
        "fig": None, "printed": "", "errors": errors,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def show_figure(fig) -> None:
    if hasattr(fig, "to_plotly_json"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig)
        plt.close("all")


def show_outputs(table, fig, printed: str) -> None:
    """Display whichever of table / chart / printed text the code produced."""
    if table is not None:
        if isinstance(table, pd.Series):
            table = table.to_frame()
        st.dataframe(table)
    if fig is not None:
        show_figure(fig)
    if printed.strip():
        st.text(printed)
    if table is None and fig is None and not printed.strip():
        st.info("The code ran but produced no table, chart, or output.")


def show_sources(uploaded, df: pd.DataFrame, code: str, errors: list) -> None:
    """Show where the analysis came from: data, columns, libraries, model."""
    st.subheader("Data and sources")
    used = columns_used(code, df)
    st.markdown(
        f"- **Dataset:** {uploaded.name} ({len(df)} rows, {df.shape[1]} columns)\n"
        f"- **Columns used:** {', '.join(used) if used else 'not detected'}\n"
        f"- **Libraries used:** {', '.join(libraries_used(code)) or 'none'}\n"
        f"- **Generated by:** {MODEL}\n"
        f"- **Attempts needed:** {len(errors) + 1}\n"
        f"- **Run at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    st.caption(
        "AI-generated analysis can contain mistakes. Check the code above "
        "before relying on the results."
    )


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("LLM Data Analyst")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "No API key found. Add GEMINI_API_KEY=your-key to the .env file "
            "and restart the app."
        )
        st.stop()
    client = genai.Client(api_key=api_key)

    # 1. Upload a dataset
    uploaded = st.file_uploader("Upload a CSV", type="csv")
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read that file as a CSV: {e}")
        st.stop()
    if df.empty:
        st.error("The file has no rows of data.")
        st.stop()

    st.subheader("Data preview")
    st.dataframe(df.head())

    profile = profile_dataframe(df)
    with st.expander("Dataset profile (this is what the LLM will see)"):
        st.text(profile)

    # 2. Ask a question in natural language
    st.subheader("Ask a question")
    question = st.text_input(
        "e.g. Does study time relate to GPA?",
        placeholder="Type your question about the data...",
    )

    if not st.button("Analyse"):
        return
    if not question.strip():
        st.warning("Please type a question first.")
        st.stop()

    # 3. LLM generates an analysis plan and code
    with st.spinner("Asking the LLM..."):
        try:
            plan, code = generate_plan_and_code(client, question, profile)
        except Exception as e:
            st.error(f"LLM request failed: {e}")
            st.stop()

    # 4. Execute the code (with automatic retries on failure)
    outcome = run_with_retries(client, question, profile, df, plan, code)
    plan, code, errors = outcome["plan"], outcome["code"], outcome["errors"]

    st.subheader("Analysis plan")
    st.write(plan)
    if code:
        with st.expander("Generated code"):
            st.code(code, language="python")

    # 5. Display results
    st.subheader("Results")
    if not outcome["success"]:
        st.error(
            "Sorry, I could not produce a working analysis after "
            f"{len(errors)} attempt(s). Try rephrasing your question "
            "or asking something simpler."
        )
        with st.expander("What went wrong"):
            for i, err in enumerate(errors, 1):
                st.text(f"Attempt {i}: {err}")
        return

    if errors:
        st.caption(
            f"The first attempt failed, so the LLM fixed its code "
            f"({len(errors)} retry/retries)."
        )
        with st.expander("Errors that were fixed automatically"):
            for i, err in enumerate(errors, 1):
                st.text(f"Attempt {i}: {err}")

    show_outputs(outcome["table"], outcome["fig"], outcome["printed"])

    # 6. Plain-language explanation
    with st.spinner("Writing explanation..."):
        try:
            explanation = explain_results(
                client, question, outcome["table"], outcome["printed"]
            )
            st.subheader("Explanation")
            st.write(explanation)
        except Exception as e:
            st.warning(f"Could not generate an explanation: {e}")

    # 7. Data and source references
    show_sources(uploaded, df, code, errors)


main()

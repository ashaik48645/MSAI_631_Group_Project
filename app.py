"""
MSAI 631 Course Support RAG Chatbot
Contributor area: Jyothirmayi Sunkara

Reference Gradio interface.
Jyothirmayi should review, modify, test, and commit her own version.
"""

from pathlib import Path
from typing import List

import gradio as gr

from rag_pipeline import CourseSupportRAG


rag = None


def build_rag():
    global rag
    if rag is None:
        rag = CourseSupportRAG(enable_llm=True)
    return rag


def index_uploaded_files(files):
    if not files:
        return "Please select one or more PDF, DOCX, or TXT files."

    file_paths: List[str] = []
    for item in files:
        if isinstance(item, str):
            file_paths.append(item)
        elif hasattr(item, "name"):
            file_paths.append(item.name)

    try:
        count = build_rag().index_files(file_paths)
        names = ", ".join(Path(p).name for p in file_paths)
        return f"Indexed {count} text chunks from: {names}"
    except Exception as exc:
        return f"Indexing failed: {type(exc).__name__}: {exc}"


def chat(message, history):
    try:
        return build_rag().ask(message).answer
    except Exception as exc:
        return (
            "The chatbot encountered an error while processing the question: "
            f"{type(exc).__name__}: {exc}"
        )


with gr.Blocks(title="MSAI 631 Course Support RAG Chatbot") as demo:
    gr.Markdown(
        """
        # MSAI 631 Course Support RAG Chatbot
        Upload approved course documents, index them, and ask questions.
        Answers are intended to be grounded in the uploaded material and show sources.
        """
    )

    with gr.Row():
        files = gr.File(
            label="Course documents",
            file_count="multiple",
            file_types=[".pdf", ".docx", ".txt"],
            type="filepath",
        )
        index_button = gr.Button("Index Documents", variant="primary")

    status = gr.Textbox(label="Index status", interactive=False)
    index_button.click(index_uploaded_files, inputs=files, outputs=status)

    gr.ChatInterface(
        fn=chat,
        title="Ask about the uploaded course documents",
        description=(
            "Try a question such as: "
            "\"How many pages should the design document have?\""
        ),
    )


if __name__ == "__main__":
    demo.launch()

import argparse
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


MAP_TEMPLATE = """
You are extracting climate-policy key phrases from a Chinese policy document.

Document: {doc_name}

Text chunk:
{context}

Return concise climate-related policy phrases only. Use a numbered Chinese list.
"""


REDUCE_TEMPLATE = """
You are consolidating phrase extraction results from one policy document.

Document: {doc_name}

Chunk-level extraction results:
{local_extractions}

Merge duplicates, standardize wording, and keep only climate-related key phrases.
Return a numbered Chinese list.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a policy PDF vector index and extract key phrases.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-index")
    build.add_argument("--pdf-dir", action="append", required=True, help="PDF directory. Can be repeated.")
    build.add_argument("--vectorstore", default="outputs/climate_policy_vectorstore")
    build.add_argument("--chunk-size", type=int, default=1000)
    build.add_argument("--chunk-overlap", type=int, default=200)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--vectorstore", default="outputs/climate_policy_vectorstore")
    extract.add_argument("--output-xlsx", default="outputs/policy_key_phrases.xlsx")
    extract.add_argument("--provider", choices=["qwen", "openai"], default="qwen")
    extract.add_argument("--model", default=None)
    extract.add_argument("--batch-size", type=int, default=10)

    normalize = subparsers.add_parser("normalize-excel")
    normalize.add_argument("--input-xlsx", required=True)
    normalize.add_argument("--output-xlsx", required=True)
    normalize.add_argument("--filename-column", default="文件名")
    normalize.add_argument("--content-column", default="关键信息提取")

    return parser.parse_args()


def make_llm(provider: str, model: str | None, temperature: float) -> ChatOpenAI:
    load_dotenv()
    if provider == "qwen":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        model_name = model or os.getenv("DASHSCOPE_MODEL", "qwen-plus")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = None
        model_name = model or "gpt-4o-mini"

    if not api_key:
        raise ValueError(f"Missing API key for provider '{provider}'. Check .env.example.")

    kwargs = {"model": model_name, "temperature": temperature, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def make_embeddings() -> OpenAIEmbeddings:
    load_dotenv()
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("EMBEDDING_BASE_URL")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    if not api_key:
        raise ValueError("Missing embedding API key. Set EMBEDDING_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")

    kwargs = {"api_key": api_key, "model": model}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


def build_index(args: argparse.Namespace) -> None:
    pdf_paths: list[Path] = []
    for folder in args.pdf_dir:
        pdf_paths.extend(sorted(Path(folder).glob("*.pdf")))
    if not pdf_paths:
        raise ValueError("No PDF files found in the provided directories.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？"],
    )

    documents = []
    for pdf_path in tqdm(pdf_paths, desc="Loading PDFs"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = pdf_path.name
        documents.extend(splitter.split_documents(pages))

    embeddings = make_embeddings()
    vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
    vector_store.save_local(args.vectorstore)
    print(f"Saved vector store: {args.vectorstore}")


def extract_phrases(args: argparse.Namespace) -> None:
    load_dotenv()
    embeddings = make_embeddings()
    vector_store = FAISS.load_local(
        folder_path=args.vectorstore,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    map_chain = LLMChain(llm=make_llm(args.provider, args.model, 0), prompt=ChatPromptTemplate.from_template(MAP_TEMPLATE))
    reduce_chain = LLMChain(
        llm=make_llm(args.provider, args.model, 0.1),
        prompt=ChatPromptTemplate.from_template(REDUCE_TEMPLATE),
    )

    all_docs = list(vector_store.docstore._dict.values())
    file_names = sorted({doc.metadata.get("source", "unknown") for doc in all_docs})

    results = []
    for doc_name in tqdm(file_names, desc="Extracting"):
        chunks = [doc for doc in all_docs if doc.metadata.get("source") == doc_name]
        local_extractions = []
        for start in range(0, len(chunks), args.batch_size):
            batch = chunks[start : start + args.batch_size]
            context = "\n".join(chunk.page_content for chunk in batch)
            local_extractions.append(map_chain.run({"doc_name": doc_name, "context": context}))

        final_text = reduce_chain.run(
            {
                "doc_name": doc_name,
                "local_extractions": "\n".join(local_extractions),
            }
        )
        results.append({"文件名": doc_name, "关键信息提取": final_text})

    output_path = Path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"Saved extraction results: {output_path}")


def normalize_excel(args: argparse.Namespace) -> None:
    df = pd.read_excel(args.input_xlsx, usecols=[args.filename_column, args.content_column])
    rows = []
    for _, row in df.iterrows():
        filename = row[args.filename_column]
        content = str(row[args.content_column]).strip()
        items = re.split(r"\s+\d+\.\s*", content)
        for item in items:
            phrase = item.strip()
            if phrase:
                rows.append({args.filename_column: filename, "标准化专业语料": phrase})

    output_path = Path(args.output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(output_path, index=False)
    print(f"Saved normalized phrase table: {output_path}")


def main() -> None:
    args = parse_args()
    if args.command == "build-index":
        build_index(args)
    elif args.command == "extract":
        extract_phrases(args)
    elif args.command == "normalize-excel":
        normalize_excel(args)


if __name__ == "__main__":
    main()

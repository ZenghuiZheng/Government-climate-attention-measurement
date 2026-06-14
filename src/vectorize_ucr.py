import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vectorize text and aggregate phrase similarities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_vector = argparse.ArgumentParser(add_help=False)
    common_vector.add_argument("--text-column", default="报告全文")
    common_vector.add_argument("--model-dir", required=True, help="BERT checkpoint or model directory.")
    common_vector.add_argument("--output-dir", required=True)
    common_vector.add_argument("--max-length", type=int, default=256)
    common_vector.add_argument("--batch-size", type=int, default=64)
    common_vector.add_argument("--num-workers", type=int, default=4)

    documents = subparsers.add_parser("documents", parents=[common_vector], help="Vectorize each document.")
    documents.add_argument("--input-csv", required=True)
    documents.add_argument("--group-columns", nargs="*", default=[])

    sentences = subparsers.add_parser("sentences", parents=[common_vector], help="Vectorize sentence-level text.")
    sentences.add_argument("--input-csv", required=True)
    sentences.add_argument("--group-columns", nargs="*", default=[])
    sentences.add_argument("--min-sentence-length", type=int, default=3)

    phrases = subparsers.add_parser("phrases", parents=[common_vector], help="Vectorize phrase list.")
    phrases.add_argument("--input-file", required=True, help="CSV or Excel file.")
    phrases.add_argument("--sheet-name", default=None, help="Excel sheet name.")

    similarity = subparsers.add_parser("similarity", help="Aggregate sentence-to-phrase similarity.")
    similarity.add_argument("--sentence-vectors", required=True)
    similarity.add_argument("--sentence-mapping", required=True)
    similarity.add_argument("--phrase-vectors", required=True)
    similarity.add_argument("--output-csv", required=True)
    similarity.add_argument("--group-columns", nargs="+", required=True)
    similarity.add_argument("--top-quantiles", nargs="*", type=float, default=[0.10, 0.25, 0.50])
    similarity.add_argument("--chunk-size", type=int, default=4096)

    merge = subparsers.add_parser("merge-panel", help="Merge similarity indicators into panel data.")
    merge.add_argument("--panel-csv", required=True)
    merge.add_argument("--similarity-csv", required=True)
    merge.add_argument("--output-csv", required=True)
    merge.add_argument("--key-columns", nargs="+", required=True)

    return parser.parse_args()


def make_vectorization_config(args: argparse.Namespace):
    from gca_project.vectorization import VectorizationConfig

    return VectorizationConfig(
        text_column=args.text_column,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


def main() -> None:
    args = parse_args()
    if args.command == "documents":
        from gca_project.vectorization import vectorize_documents

        vectorize_documents(args.input_csv, args.group_columns, make_vectorization_config(args))
    elif args.command == "sentences":
        from gca_project.vectorization import vectorize_sentences

        vectorize_sentences(
            args.input_csv,
            args.group_columns,
            args.min_sentence_length,
            make_vectorization_config(args),
        )
    elif args.command == "phrases":
        from gca_project.vectorization import vectorize_phrases

        vectorize_phrases(args.input_file, args.sheet_name, make_vectorization_config(args))
    elif args.command == "similarity":
        from gca_project.similarity import aggregate_similarity

        aggregate_similarity(
            sentence_vectors_path=args.sentence_vectors,
            sentence_mapping_path=args.sentence_mapping,
            phrase_vectors_path=args.phrase_vectors,
            output_csv=args.output_csv,
            group_columns=args.group_columns,
            top_quantiles=args.top_quantiles,
            chunk_size=args.chunk_size,
        )
    elif args.command == "merge-panel":
        from gca_project.panel import merge_panel_data

        merge_panel_data(args.panel_csv, args.similarity_csv, args.output_csv, args.key_columns)


if __name__ == "__main__":
    main()

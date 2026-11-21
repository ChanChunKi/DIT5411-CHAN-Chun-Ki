# Result provenance

`test_comparison.csv` and `test_comparison.json` are the original full-precision exports supplied from the completed Google Drive experiment. They replace the earlier reconstructed summaries. `README_test_results.md` is the saved Markdown summary from the same experiment.

The JSON includes original timing values, checkpoint hashes, and the split-manifest hash. The checkpoint paths refer to the original Colab/Google Drive environment; model binaries are not included in this repository. Accuracies use the 0–1 scale in CSV/JSON and percentages in the Markdown tables.

The manifest hash in all three JSON result records matches the included `data/split_manifest.csv`. All 684,677 source-image paths, 13,065 label mappings and 43 split exceptions were checked for consistency. The original `labels.json` was recovered from the earlier Kaggle backup and matches the manifest's label, folder and character mapping.

# Multimodal Information Synthesis for AI-Assisted Agricultural Field Assessment Using Model Context Protocol

## Author

**Stefan Stiller**  
Leibniz-Centre for Agricultural Landscape Research (ZALF) e.V.  
Email: stefan \[dot\] stiller \[at\] zalf \[dot\] de, stillsen \[at\] gmail \[dot\] com  
ORCID: [0009-0004-7468-1678](https://orcid.org/0009-0004-7468-1678)

## Description

This repository accompanies the study *"Multimodal Information Synthesis for AI-Assisted Agricultural Field Assessment Using Model Context Protocol"*.

It provides software for structured multimodal information access for large language models (LLMs) via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/):

1. **MCP resource servers** that expose agricultural information modalities to LLM clients:
   - yield prediction from RGB imagery
   - scientific PDF literature (topic-based access)
   - soil CSV data (lab results and moisture)
2. **Evaluation plotting code** that regenerates the manuscript figures comparing:
   - structured MCP access vs unstructured availability of the same sources
   - increasing modality dose under MCP (single resources → pairs → all resources)

Related work on the underlying yield model is available at:  
[Self-Supervised Learning for Crop Classification and Yield Prediction](https://github.com/stillsen/Self-Supervised-Learning-for-Crop-Classification-and-Yield-Prediction)

## Repository contents

```
.
├── MCP_yield_server.py                 # MCP server: image-based yield prediction
├── MCP_pdf_resource_server.py          # MCP server: topic-based PDF literature access
├── MCP_csv_resource_server.py          # MCP server: soil CSV resources
├── llm_judge-two-question-figures.py   # Manuscript Fig. 1 and Fig. 2
├── requirements.txt
├── CITATION.cff
├── LICENSE                             # GNU GPLv3
└── README.md
```

## Usage

### MCP servers

Start each server from an MCP-compatible client configuration (e.g. Claude Desktop / Cursor), pointing to the corresponding Python file:

- `MCP_yield_server.py`
- `MCP_pdf_resource_server.py`
- `MCP_csv_resource_server.py`

### Manuscript figures

```bash
python llm_judge-two-question-figures.py
```

This writes Figure 1 (structured vs unstructured access) and Figure 2 (MCP modality ladder), plus caption text files, to `Evaluation_Reports/`.

## Notes for reuse

- Judge model names in the figures are remapped for display; scores come from the evaluation CSV referenced in the script.
- Edit `OUTPUT_DIR` / `CSV_NAME` in the figure script if your layout differs.

## Citation

If you use this code, please cite it as indicated in `CITATION.cff` (and the associated paper when available).

## License

This project is licensed under the GNU GPLv3 License — see the [LICENSE](LICENSE) file for details.

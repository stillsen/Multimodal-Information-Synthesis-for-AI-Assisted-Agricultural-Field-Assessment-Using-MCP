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

## Data and model requirements

The MCP servers expect local assets next to the scripts (or adjust paths in the scripts):

| Component | Expected location |
|---|---|
| PDF literature / CSV inputs | `Inputs/` (beside the MCP servers) |
| Yield model checkpoint | `Models/SSL_VICRegConvNeXt-tiny_Lupine_f1.ckpt` |
| Trainer helper used by the yield server | `Templates/RGBYieldRegressor_Trainer.py` (from the SSL repository above) |
| LLM-judge score table | CSV named in `llm_judge-two-question-figures.py` (`CSV_NAME`) |

Place the judge CSV in the figure script output directory (default: `./Evaluation_Reports/`), then run the plotting script.

## Installation

```bash
git clone https://github.com/stillsen/<this-repository>.git
cd <this-repository>
python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### MCP servers

Start each server from an MCP-compatible client configuration (e.g. Claude Desktop / Cursor), pointing to the corresponding Python file:

- `MCP_yield_server.py`
- `MCP_pdf_resource_server.py`
- `MCP_csv_resource_server.py`

Ensure `Inputs/`, `Models/`, and `Templates/` are available as described above.

### Manuscript figures

```bash
python llm_judge-two-question-figures.py
```

This writes Figure 1 (structured vs unstructured access) and Figure 2 (MCP modality ladder), plus caption text files, to `Evaluation_Reports/`.

## Notes for reuse

- Judge model names in the figures are remapped for display; scores come from the provided CSV.
- Absolute machine paths were removed from the figure script for publication; edit `OUTPUT_DIR` / `CSV_NAME` if your layout differs.
- Large model checkpoints are not included here; provide them locally or via the associated data deposit.

## Citation

If you use this code, please cite it as indicated in `CITATION.cff` (and the associated paper when available).

## License

This project is licensed under the GNU GPLv3 License — see the [LICENSE](LICENSE) file for details.

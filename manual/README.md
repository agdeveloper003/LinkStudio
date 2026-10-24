# LinkStudio manual

LaTeX source of *LinkStudio - Theory and User Manual* (`linkstudio-manual.pdf`).

## Layout

| path | content |
|------|---------|
| `linkstudio-manual.tex` | main file (title page, part and chapter order) |
| `preamble.tex` | packages, colours, TikZ/pgfplots styles, macros (`\blk`, `\prm`, `\meta`, `\codeexcerpt`, boxes) |
| `chapters/` | one file per chapter; `A-`, `B-`, `C-` are appendices |
| `tools/generate.py` | runs the simulator to produce figure data, extracts code excerpts, builds reference tables |
| `generated/` | output of the generator (do not edit by hand) |

## Build

```bash
python manual/tools/generate.py          # from the repository root (about 1 minute)
cd manual
latexmk -pdf linkstudio-manual.tex          # or run pdflatex three times
```

The generator has three jobs:

- **Figure data.** It runs the real blocks and writes `generated/data/*.dat`. Every measured curve in the manual is simulator output.
- **Code excerpts.** It extracts them with `inspect` into `generated/snippets/`, recording file names and line numbers in `generated/snippets.tex`.
- **Reference tables.** It writes the block table (`blocks.tex`, from the block registry) and the scenario table (`scenarios.tex`, from `backend/scenarios`).

Re-run it after changing the code base, so that figures, excerpts and tables match the code.

Useful options:

```bash
python manual/tools/generate.py --list               # figure names
python manual/tools/generate.py --only coding ook    # regenerate some figures
python manual/tools/generate.py --no-figures         # excerpts and tables only
```

## Writing conventions

- Each theory chapter follows the same three steps: model, then implementation (an *In the code* box plus `\codeexcerpt`), then validation (simulated markers against closed-form lines).
- To add an excerpt, add an entry to `SNIPPETS` in `tools/generate.py` (name, dotted object, optional first/last line regexes), then use `\codeexcerpt{name}{caption}`.
- To add a figure with data, add a `@figure` function that writes a `.dat` file, then plot it with `\datafile{name}`.

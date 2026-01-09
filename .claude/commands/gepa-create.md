Read the skill definition at `skills/observable-gepa-optimizer/SKILL.md` and the create-from-scratch workflow at `skills/observable-gepa-optimizer/use-cases/create-from-scratch.md`.

Also read all reference files to understand the available patterns:
- `skills/observable-gepa-optimizer/references/task-types.md` - Task signatures
- `skills/observable-gepa-optimizer/references/metric-templates.md` - Metric functions
- `skills/observable-gepa-optimizer/references/multimodal-patterns.md` - PDF/image handling
- `skills/observable-gepa-optimizer/references/output-templates/single-file.md` - Single file template
- `skills/observable-gepa-optimizer/references/output-templates/modular-structure.md` - Modular template
- `skills/observable-gepa-optimizer/references/output-templates/notebook.md` - Notebook template

Follow the interactive workflow in create-from-scratch.md to help the user build an optimized DSPy program from their labelled dataset. Guide them through all 5 phases:

1. **Data Understanding** - Ask about their data, examine it, identify inputs/outputs
2. **Task Definition** - Determine task type, propose DSPy Signature, choose module
3. **Evaluation Strategy** - Define how to compare predictions to ground truth
4. **Output Preferences** - Single file, modular, or notebook; requirements.txt
5. **Code Generation** - Generate complete, runnable code

If the user has already provided a data path, start by examining that data. Otherwise, ask them for the path to their labelled dataset.

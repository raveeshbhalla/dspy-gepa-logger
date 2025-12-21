"""Generate HTML comparison reports for GEPA optimization runs."""

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def get_run_info(conn: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    """Get run metadata."""
    cur = conn.execute(
        "SELECT run_id, status, started_at, completed_at, total_iterations, "
        "accepted_count, seed_score, final_score FROM runs WHERE run_id = ?",
        (run_id,)
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Run {run_id} not found")

    return {
        'run_id': row[0],
        'status': row[1],
        'started_at': row[2],
        'completed_at': row[3],
        'total_iterations': row[4],
        'accepted_count': row[5],
        'seed_score': row[6],
        'final_score': row[7],
    }


def get_program(conn: sqlite3.Connection, program_id: int) -> dict[str, Any]:
    """Get program details."""
    cur = conn.execute(
        "SELECT program_id, signature, instructions_json FROM programs WHERE program_id = ?",
        (program_id,)
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Program {program_id} not found")

    return {
        'program_id': row[0],
        'signature': row[1],
        'instructions': json.loads(row[2]),
    }


def identify_baseline_program(conn: sqlite3.Connection, run_id: str, baseline_id: int | None) -> int:
    """Identify the baseline program ID."""
    if baseline_id:
        return baseline_id

    # Try original_program_id from runs
    cur = conn.execute("SELECT original_program_id FROM runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]

    # Use iteration 0's candidate program (which has the actual baseline rollouts)
    cur = conn.execute(
        "SELECT candidate_program_id FROM iterations WHERE run_id = ? AND iteration_number = 0",
        (run_id,)
    )
    row = cur.fetchone()
    if row and row[0]:
        return row[0]

    raise ValueError("Could not identify baseline program")


def identify_optimized_program(conn: sqlite3.Connection, run_id: str, optimized_id: int | None) -> int:
    """Identify the optimized program ID."""
    if optimized_id:
        return optimized_id

    # Try optimized_program_id from runs
    cur = conn.execute("SELECT optimized_program_id FROM runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    if row and row[0]:
        return row[0]

    # Use the last accepted iteration's candidate program
    cur = conn.execute(
        "SELECT candidate_program_id FROM iterations "
        "WHERE run_id = ? AND accepted = 1 AND candidate_program_id IS NOT NULL "
        "ORDER BY iteration_number DESC LIMIT 1",
        (run_id,)
    )
    row = cur.fetchone()
    if row and row[0]:
        return row[0]

    raise ValueError("Could not identify optimized program")


def get_validation_examples(conn: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    """Get all validation examples."""
    cur = conn.execute(
        "SELECT example_id, example_index, inputs_json, outputs_json "
        "FROM examples WHERE run_id = ? AND dataset_type = 'val' "
        "ORDER BY example_index",
        (run_id,)
    )

    examples = []
    for row in cur.fetchall():
        examples.append({
            'example_id': row[0],
            'example_index': row[1],
            'inputs': json.loads(row[2]),
            'outputs': json.loads(row[3]) if row[3] else None,
        })

    return examples


def get_rollout_output(conn: sqlite3.Connection, run_id: str, program_id: int, example_id: int) -> dict[str, Any] | None:
    """Get rollout output for a specific program and example."""
    cur = conn.execute(
        """
        SELECT r.output_json, r.score, r.feedback
        FROM rollouts r
        JOIN iterations i ON r.iteration_id = i.iteration_id
        WHERE i.run_id = ? AND r.program_id = ? AND r.example_id = ? AND r.rollout_type = 'validation'
        LIMIT 1
        """,
        (run_id, program_id, example_id)
    )

    row = cur.fetchone()
    if not row:
        return None

    return {
        'output': json.loads(row[0]) if row[0] else None,
        'score': row[1],
        'feedback': row[2],
    }


def categorize_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Categorize examples into improvements, regressions, and equivalent."""
    improvements = []
    regressions = []
    equivalent = []

    for ex in examples:
        baseline_score = ex.get('baseline_score', 0.0) or 0.0
        optimized_score = ex.get('optimized_score', 0.0) or 0.0
        delta = optimized_score - baseline_score

        if delta > 0.001:  # Improvement threshold
            improvements.append(ex)
        elif delta < -0.001:  # Regression threshold
            regressions.append(ex)
        else:
            equivalent.append(ex)

    return {
        'improvements': improvements,
        'regressions': regressions,
        'equivalent': equivalent,
    }


def generate_html(
    run_info: dict[str, Any],
    baseline_program: dict[str, Any],
    optimized_program: dict[str, Any],
    examples_by_category: dict[str, list[dict[str, Any]]],
    baseline_avg_score: float,
    optimized_avg_score: float,
) -> str:
    """Generate the HTML report."""

    # Calculate counts
    n_improvements = len(examples_by_category['improvements'])
    n_regressions = len(examples_by_category['regressions'])
    n_equivalent = len(examples_by_category['equivalent'])

    # Format prompts
    baseline_prompt = json.dumps(baseline_program['instructions'], indent=2)
    optimized_prompt = json.dumps(optimized_program['instructions'], indent=2)

    # Build examples HTML for each tab
    def build_example_rows(examples: list[dict[str, Any]], category: str) -> str:
        if not examples:
            return f'<div class="empty-state">No {category} to display</div>'

        rows = []
        for i, ex in enumerate(examples):
            baseline_out = ex.get('baseline_output', {}) or {}
            optimized_out = ex.get('optimized_output', {}) or {}
            baseline_score = ex.get('baseline_score', 0.0) or 0.0
            optimized_score = ex.get('optimized_score', 0.0) or 0.0
            delta = optimized_score - baseline_score

            # Truncate outputs for preview
            baseline_preview = str(baseline_out.get('answer', ''))[:100]
            optimized_preview = str(optimized_out.get('answer', ''))[:100]

            input_str = json.dumps(ex['inputs'])[:100]

            # Escape for JSON embedding
            baseline_full = json.dumps(baseline_out)
            optimized_full = json.dumps(optimized_out)
            inputs_full = json.dumps(ex['inputs'])

            rows.append(f'''
                <div class="example-row" onclick='openModal({i}, "{category}", {inputs_full}, {baseline_full}, {optimized_full}, {baseline_score}, {optimized_score})'>
                    <div class="example-header">
                        <span class="example-num">Example #{ex["example_index"]}</span>
                        <span class="score-delta {'positive' if delta > 0 else 'negative' if delta < 0 else 'neutral'}">{delta:+.3f}</span>
                    </div>
                    <div class="example-input">{input_str}{'...' if len(input_str) >= 100 else ''}</div>
                    <div class="example-outputs">
                        <div class="output-col">
                            <div class="output-label">Baseline ({baseline_score:.3f})</div>
                            <div class="output-preview">{baseline_preview}{'...' if len(baseline_preview) >= 100 else ''}</div>
                        </div>
                        <div class="output-col">
                            <div class="output-label">Optimized ({optimized_score:.3f})</div>
                            <div class="output-preview">{optimized_preview}{'...' if len(optimized_preview) >= 100 else ''}</div>
                        </div>
                    </div>
                </div>
            ''')

        return '\n'.join(rows)

    improvements_html = build_example_rows(examples_by_category['improvements'], 'improvements')
    regressions_html = build_example_rows(examples_by_category['regressions'], 'regressions')
    equivalent_html = build_example_rows(examples_by_category['equivalent'], 'equivalent')

    html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GEPA Optimization Report - {run_info['run_id'][:8]}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
            color: #333;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background: #2c3e50;
            color: white;
            padding: 20px 30px;
        }}

        .header h1 {{
            font-size: 24px;
            margin-bottom: 8px;
        }}

        .header .meta {{
            font-size: 14px;
            opacity: 0.9;
        }}

        .prompt-section {{
            padding: 30px;
            border-bottom: 1px solid #e0e0e0;
        }}

        .prompt-section h2 {{
            font-size: 18px;
            margin-bottom: 20px;
            color: #2c3e50;
        }}

        .prompt-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            max-height: 400px;
        }}

        .prompt-panel {{
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }}

        .prompt-header {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-bottom: 1px solid #ddd;
        }}

        .prompt-title {{
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }}

        .prompt-stats {{
            font-size: 12px;
            color: #666;
        }}

        .prompt-stats .stat {{
            margin-right: 12px;
        }}

        .prompt-stats .positive {{
            color: #22c55e;
        }}

        .prompt-stats .negative {{
            color: #ef4444;
        }}

        .prompt-stats .neutral {{
            color: #94a3b8;
        }}

        .prompt-body {{
            padding: 16px;
            overflow-y: auto;
            max-height: 340px;
            background: #fafafa;
        }}

        .prompt-body pre {{
            margin: 0;
            font-family: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Consolas, monospace;
            font-size: 12px;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .examples-section {{
            padding: 30px;
        }}

        .tabs {{
            display: flex;
            border-bottom: 2px solid #e0e0e0;
            margin-bottom: 20px;
        }}

        .tab {{
            padding: 12px 24px;
            cursor: pointer;
            background: none;
            border: none;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }}

        .tab:hover {{
            color: #2c3e50;
        }}

        .tab.active {{
            color: #2c3e50;
            border-bottom-color: #3b82f6;
        }}

        .tab-badge {{
            margin-left: 8px;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
        }}

        .tab-badge.improvements {{
            background: #dcfce7;
            color: #16a34a;
        }}

        .tab-badge.regressions {{
            background: #fee2e2;
            color: #dc2626;
        }}

        .tab-badge.equivalent {{
            background: #f1f5f9;
            color: #64748b;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        .empty-state {{
            text-align: center;
            padding: 60px 20px;
            color: #999;
            font-size: 14px;
        }}

        .example-row {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 16px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.2s;
            max-height: 180px;
            overflow: hidden;
        }}

        .example-row:hover {{
            border-color: #3b82f6;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.1);
        }}

        .example-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }}

        .example-num {{
            font-weight: 600;
            font-size: 13px;
            color: #2c3e50;
        }}

        .score-delta {{
            font-size: 12px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }}

        .score-delta.positive {{
            background: #dcfce7;
            color: #16a34a;
        }}

        .score-delta.negative {{
            background: #fee2e2;
            color: #dc2626;
        }}

        .score-delta.neutral {{
            background: #f1f5f9;
            color: #64748b;
        }}

        .example-input {{
            font-size: 12px;
            color: #666;
            margin-bottom: 12px;
            padding: 8px;
            background: #f8f9fa;
            border-radius: 4px;
        }}

        .example-outputs {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }}

        .output-col {{
            font-size: 12px;
        }}

        .output-label {{
            font-weight: 600;
            margin-bottom: 4px;
            color: #555;
        }}

        .output-preview {{
            color: #666;
            line-height: 1.4;
        }}

        /* Modal */
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            overflow-y: auto;
        }}

        .modal.active {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
        }}

        .modal-content {{
            background: white;
            border-radius: 8px;
            width: 100%;
            max-width: 1200px;
            max-height: 90vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}

        .modal-header {{
            padding: 20px 30px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #f8f9fa;
        }}

        .modal-header h3 {{
            font-size: 18px;
            color: #2c3e50;
        }}

        .modal-close {{
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #666;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
        }}

        .modal-close:hover {{
            background: #e0e0e0;
        }}

        .modal-body {{
            padding: 30px;
        }}

        .modal-input {{
            margin-bottom: 20px;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 6px;
        }}

        .modal-input h4 {{
            font-size: 14px;
            margin-bottom: 8px;
            color: #555;
        }}

        .modal-input pre {{
            margin: 0;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 12px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .modal-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .modal-output {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
        }}

        .modal-output-header {{
            padding: 12px 16px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
            font-weight: 600;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
        }}

        .modal-output-body {{
            padding: 16px;
            max-height: 400px;
            overflow-y: auto;
        }}

        .modal-output-body pre {{
            margin: 0;
            font-family: "SF Mono", Monaco, monospace;
            font-size: 12px;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GEPA Optimization Report</h1>
            <div class="meta">
                Run ID: {run_info['run_id']} |
                Status: {run_info['status']} |
                Iterations: {run_info['total_iterations']} |
                Accepted: {run_info['accepted_count']}
            </div>
        </div>

        <div class="prompt-section">
            <h2>Prompt Comparison</h2>
            <div class="prompt-comparison">
                <div class="prompt-panel">
                    <div class="prompt-header">
                        <div class="prompt-title">Baseline Program <span style="font-weight: normal; color: #666;">(avg: {baseline_avg_score:.3f})</span></div>
                        <div class="prompt-stats">
                            <span class="stat">ID: {baseline_program['program_id']}</span>
                        </div>
                    </div>
                    <div class="prompt-body">
                        <pre>{baseline_prompt}</pre>
                    </div>
                </div>

                <div class="prompt-panel">
                    <div class="prompt-header">
                        <div class="prompt-title">Optimized Program <span style="font-weight: normal; color: #666;">(avg: {optimized_avg_score:.3f})</span></div>
                        <div class="prompt-stats">
                            <span class="stat positive">↑ {n_improvements} improved</span>
                            <span class="stat negative">↓ {n_regressions} regressed</span>
                            <span class="stat neutral">= {n_equivalent} equivalent</span>
                        </div>
                    </div>
                    <div class="prompt-body">
                        <pre>{optimized_prompt}</pre>
                    </div>
                </div>
            </div>
        </div>

        <div class="examples-section">
            <div class="tabs">
                <button class="tab active" onclick="switchTab('improvements')">
                    Improvements
                    <span class="tab-badge improvements">{n_improvements}</span>
                </button>
                <button class="tab" onclick="switchTab('regressions')">
                    Regressions
                    <span class="tab-badge regressions">{n_regressions}</span>
                </button>
                <button class="tab" onclick="switchTab('equivalent')">
                    Equivalent
                    <span class="tab-badge equivalent">{n_equivalent}</span>
                </button>
            </div>

            <div id="improvements-content" class="tab-content active">
                {improvements_html}
            </div>

            <div id="regressions-content" class="tab-content">
                {regressions_html}
            </div>

            <div id="equivalent-content" class="tab-content">
                {equivalent_html}
            </div>
        </div>
    </div>

    <div id="modal" class="modal" onclick="closeModalOnBackdrop(event)">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modal-title">Example Comparison</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body">
                <div class="modal-input">
                    <h4>Input</h4>
                    <pre id="modal-input"></pre>
                </div>
                <div class="modal-comparison">
                    <div class="modal-output">
                        <div class="modal-output-header">
                            <span>Baseline</span>
                            <span id="modal-baseline-score"></span>
                        </div>
                        <div class="modal-output-body">
                            <pre id="modal-baseline-output"></pre>
                        </div>
                    </div>
                    <div class="modal-output">
                        <div class="modal-output-header">
                            <span>Optimized</span>
                            <span id="modal-optimized-score"></span>
                        </div>
                        <div class="modal-output-body">
                            <pre id="modal-optimized-output"></pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabName) {{
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            event.target.closest('.tab').classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.getElementById(tabName + '-content').classList.add('active');
        }}

        function openModal(index, category, inputs, baselineOutput, optimizedOutput, baselineScore, optimizedScore) {{
            const modal = document.getElementById('modal');
            document.getElementById('modal-title').textContent = 'Example Comparison';
            document.getElementById('modal-input').textContent = JSON.stringify(inputs, null, 2);
            document.getElementById('modal-baseline-output').textContent = JSON.stringify(baselineOutput, null, 2);
            document.getElementById('modal-optimized-output').textContent = JSON.stringify(optimizedOutput, null, 2);
            document.getElementById('modal-baseline-score').textContent = 'Score: ' + baselineScore.toFixed(3);
            document.getElementById('modal-optimized-score').textContent = 'Score: ' + optimizedScore.toFixed(3);
            modal.classList.add('active');
        }}

        function closeModal() {{
            document.getElementById('modal').classList.remove('active');
        }}

        function closeModalOnBackdrop(event) {{
            if (event.target.id === 'modal') {{
                closeModal();
            }}
        }}

        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>
    '''

    return html


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate HTML comparison report for GEPA optimization run')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--run-id', required=True, help='Run ID to generate report for')
    parser.add_argument('--baseline-program-id', type=int, help='Baseline program ID (default: auto-detect)')
    parser.add_argument('--optimized-program-id', type=int, help='Optimized program ID (default: auto-detect)')
    parser.add_argument('--output', required=True, help='Output HTML file path')

    args = parser.parse_args()

    # Connect to database
    conn = sqlite3.connect(args.db)

    try:
        # Get run info
        run_info = get_run_info(conn, args.run_id)

        # Identify programs
        baseline_id = identify_baseline_program(conn, args.run_id, args.baseline_program_id)
        optimized_id = identify_optimized_program(conn, args.run_id, args.optimized_program_id)

        baseline_program = get_program(conn, baseline_id)
        optimized_program = get_program(conn, optimized_id)

        # Get validation examples
        examples = get_validation_examples(conn, args.run_id)

        # Augment with outputs and scores
        for ex in examples:
            baseline_result = get_rollout_output(conn, args.run_id, baseline_id, ex['example_index'])
            optimized_result = get_rollout_output(conn, args.run_id, optimized_id, ex['example_index'])

            ex['baseline_output'] = baseline_result['output'] if baseline_result else None
            ex['baseline_score'] = baseline_result['score'] if baseline_result else 0.0
            ex['optimized_output'] = optimized_result['output'] if optimized_result else None
            ex['optimized_score'] = optimized_result['score'] if optimized_result else 0.0

        # Categorize examples
        examples_by_category = categorize_examples(examples)

        # Calculate average scores
        baseline_scores = [ex.get('baseline_score', 0.0) or 0.0 for ex in examples]
        optimized_scores = [ex.get('optimized_score', 0.0) or 0.0 for ex in examples]
        baseline_avg_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        optimized_avg_score = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0

        # Generate HTML
        html = generate_html(run_info, baseline_program, optimized_program, examples_by_category, baseline_avg_score, optimized_avg_score)

        # Write output
        output_path = Path(args.output)
        output_path.write_text(html, encoding='utf-8')

        print(f"Report generated: {output_path}")
        print(f"Baseline program: {baseline_id}")
        print(f"Optimized program: {optimized_id}")
        print(f"Improvements: {len(examples_by_category['improvements'])}")
        print(f"Regressions: {len(examples_by_category['regressions'])}")
        print(f"Equivalent: {len(examples_by_category['equivalent'])}")

    finally:
        conn.close()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
extract_review_prompts.py

Parse a long code-review file with blocks like:

### Comment 29
<location> `tests/test_baseline_to_scorer.py:19-22` </location>
<code_context>
    while time.monotonic() - start < 1.0:
        if sid in vcmain.scorer_tasks:
            break
        time.sleep(0.01)
</code_context>
<issue_to_address>
**suggestion (code-quality):** Move a guard clause...
</issue_to_address>

and produce prompt files in prompts/ named:
  prompts/<pr_name>_<comment_id>.prompt.md
or a single file:
  prompts/<pr_name>.prompt.md

Usage:
    python tools/extract_review_prompts.py reviews/all_comments.txt \
        --pr-name "fix/review-29-simplify-wait-loop" -o prompts/

    # single file mode
    python tools/extract_review_prompts.py reviews/all_comments.txt \
        --pr-name "batch/fixes-2025-11-23" --single-file -o prompts/
"""

import argparse
import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

PROMPT_TEMPLATE = """Task: Fix a code review comment and produce a minimal, well-tested patch.

Files and location:
- **file:** {file_path}
- **lines:** {lines}
- **comment id:** {comment_id}

Context:
{code_context}

Issue summary:
{issue_summary}

Suggested change:
{suggested_change}

Acceptance criteria:
- The code compiles and passes existing unit tests
- New or updated unit tests cover the changed behavior if applicable
- Linting and formatting applied
- Commit message follows format: fix(scope): short description; include review id
- PR description includes: what changed, why, how tested, and links to failing CI or review comment

Implementation steps:
1. Reproduce locally: run the specific test or scenario that triggered the comment.
2. Implement the minimal change in the file.
3. Run unit tests and targeted test(s).
4. Add or update tests if the change affects behavior.
5. Run linter and formatter.
6. Run full test suite if quick; otherwise run CI on branch.
7. Commit with message and open PR referencing the review comment.

Deliverables:
- Patch or diff
- Test run output showing passing tests
- Commit hash and PR link or branch name
"""

DEFAULTS = {
    "file_path": "<unknown file>",
    "lines": "<unknown lines>",
    "code_context": "<no code_context provided>",
    "issue_summary": "<no issue summary provided>",
    "suggested_change": "<no suggested change provided>",
}


def sanitize_filename(s: str, max_len: int = 120) -> str:
    """Sanitize a string to be safe as a filename."""
    s = s.strip()
    # replace spaces and slashes
    s = re.sub(r"[\/\\]+", "-", s)
    s = re.sub(r"\s+", "-", s)
    # remove unsafe chars
    s = re.sub(r"[^A-Za-z0-9._\-]", "", s)
    return s[:max_len]


def find_comment_blocks(text: str) -> List[Dict[str, str]]:
    header_re = re.compile(r'^(###\s*Comment\s*\d+)', re.MULTILINE)
    headers = list(header_re.finditer(text))
    blocks = []

    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        raw = text[start:end].strip()

        comment_id_match = re.search(r'###\s*Comment\s*(\d+)', raw)
        comment_id = comment_id_match.group(1) if comment_id_match else "<unknown>"

        def extract_tag(tag: str) -> Optional[str]:
            pat = re.compile(rf'<{tag}>\s*(.*?)\s*</{tag}>', re.DOTALL | re.IGNORECASE)
            mtag = pat.search(raw)
            if not mtag:
                pat2 = re.compile(rf'<{tag}>\s*(.*?)\s*(?=(<\w+>|$))', re.DOTALL | re.IGNORECASE)
                m2 = pat2.search(raw)
                return m2.group(1).strip() if m2 else None
            return mtag.group(1).strip()

        location = extract_tag("location") or ""
        code_context = extract_tag("code_context") or ""
        issue_to_address = extract_tag("issue_to_address") or ""

        suggested_change = ""
        i  # Extract suggestion fenced block if present: ```suggestion ... ```
    suggested_change = ""
    if issue_to_address:
        # first try explicit ```suggestion fenced block
        fence_sugg = re.search(r'```suggestion\s*\n(.*?)\n```', issue_to_address, re.DOTALL | re.IGNORECASE)
        if fence_sugg:
            suggested_change = fence_sugg.group(1).strip()
        else:
            # fallback: any fenced code block
            fence_any = re.search(r'```(?:\w+)?\s*\n(.*?)\n```', issue_to_address, re.DOTALL)
            if fence_any:
                suggested_change = fence_any.group(1).strip()
            else:
                # fallback: first non-empty paragraph
                paragraphs = [p.strip() for p in issue_to_address.splitlines() if p.strip()]
                suggested_change = paragraphs[0] if paragraphs else ""


        blocks.append(
            {
                "comment_id": comment_id,
                "raw_block": raw,
                "location": location,
                "code_context": code_context,
                "issue_to_address": issue_to_address,
                "suggested_change": suggested_change,
            }
        )

    return blocks


def parse_location(location_text: str) -> Dict[str, str]:
    if not location_text:
        return {"file_path": DEFAULTS["file_path"], "lines": DEFAULTS["lines"]}
    loc = location_text.strip().strip('`').strip()
    if ":" in loc:
        file_path, lines = loc.split(":", 1)
        return {"file_path": file_path.strip(), "lines": lines.strip()}
    else:
        return {"file_path": loc, "lines": DEFAULTS["lines"]}


def build_prompt(block: Dict[str, str]) -> str:
    loc_parsed = parse_location(block["location"])
    code_ctx = block["code_context"].strip()
    if code_ctx:
        code_ctx = "```\n" + code_ctx + "\n```"
    else:
        code_ctx = DEFAULTS["code_context"]

    issue_summary = block["issue_to_address"].strip().splitlines()[0] if block["issue_to_address"].strip() else DEFAULTS["issue_summary"]
    suggested_change = block["suggested_change"].strip() or DEFAULTS["suggested_change"]

    prompt = PROMPT_TEMPLATE.format(
        file_path=loc_parsed["file_path"],
        lines=loc_parsed["lines"],
        comment_id=block["comment_id"],
        code_context=code_ctx,
        issue_summary=issue_summary,
        suggested_change=suggested_change,
    )
    return prompt


def write_prompts(blocks: List[Dict[str, str]], out_dir: Path, pr_name: str, single_file: bool) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    safe_pr = sanitize_filename(pr_name or "pr-fix")
    if single_file:
        out_path = out_dir / f"{safe_pr}.prompt.md"
        with out_path.open("w", encoding="utf-8") as fh:
            for b in blocks:
                header = f"## Comment {b['comment_id']}\n\n"
                fh.write(header)
                fh.write(build_prompt(b))
                fh.write("\n\n---\n\n")
        written.append(out_path)
        return written

    for b in blocks:
        # filename: <pr>_<comment_id>.prompt.md
        fname = out_dir / f"{safe_pr}_{b['comment_id']}.prompt.md"
        prompt_text = build_prompt(b)
        with fname.open("w", encoding="utf-8") as fh:
            fh.write(prompt_text)
        written.append(fname)
    return written


def main():
    parser = argparse.ArgumentParser(description="Extract review prompts and save as prompts/<pr>.prompt.md")
    parser.add_argument("review_file", type=Path, help="Path to the review file")
    parser.add_argument("--pr-name", required=True, help="Name of the PR or fix (used to name prompt files)")
    parser.add_argument("-o", "--out", type=Path, default=Path("prompts"), help="Output directory for prompts")
    parser.add_argument("--single-file", action="store_true", help="Write all prompts into a single prompts/<pr-name>.prompt.md file")
    args = parser.parse_args()

    if not args.review_file.exists():
        print(f"Error: review file not found: {args.review_file}")
        raise SystemExit(1)

    text = args.review_file.read_text(encoding="utf-8")
    blocks = find_comment_blocks(text)
    if not blocks:
        print("No '### Comment' blocks found. Exiting.")
        return

    written = write_prompts(blocks, args.out, args.pr_name, args.single_file)
    print(f"Wrote {len(written)} prompt file(s) to {args.out.resolve()}")
    for p in written[:20]:
        print(" -", p.name)


if __name__ == "__main__":
    main()
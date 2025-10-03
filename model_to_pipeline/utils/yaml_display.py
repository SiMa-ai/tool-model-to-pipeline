from rich.console import Console
from rich.table import Table
from rich.text import Text
import shutil
import re


def parse_yaml_as_stream(filepath):
    """Parse YAML as a stream of (type, key, value, description)."""
    rows = []
    pending_desc = None
    stack = []

    with open(filepath, "r") as f:
        for line in f:
            raw = line.rstrip("\n")

            # Section headers (# ---- ... ----)
            if re.match(r"^\s*#\s*-{2,}.*-{2,}\s*$", raw):
                section = re.sub(r"^\s*#\s*-{2,}\s*(.*?)\s*-{2,}\s*$", r"\1", raw)
                rows.append(("__SECTION__", section, "", ""))
                pending_desc = None
                continue

            # Normal comments (used for description of next field)
            if raw.strip().startswith("#"):
                comment_text = raw.lstrip("#").strip()
                if comment_text:
                    pending_desc = comment_text
                continue

            if ":" not in raw:
                continue

            indent = len(raw) - len(raw.lstrip())
            key, value = raw.split(":", 1)
            key, value = key.strip(), value.strip()

            level = indent // 2
            if level < len(stack):
                stack = stack[:level]
            stack.append(key)

            full_key = ".".join(stack)

            # Handle inline comment
            inline_desc = None
            if "#" in value:
                value, inline_comment = value.split("#", 1)
                value = value.strip()
                inline_desc = inline_comment.strip()  # <-- remove leading '#' and spaces
            else:
                value = value.strip()

            display_value = value if value else ""

            # Replace booleans with icons
            if display_value.lower() in ("true", "false"):
                display_value = (
                    Text("✅", style="green")
                    if display_value.lower() == "true"
                    else Text("❌", style="red")
                )

            # Prefer inline desc > pending comment
            desc = inline_desc if inline_desc else pending_desc
            if desc:
                desc = desc.lstrip("#").strip()  # <-- sanitize description again

            rows.append(("__FIELD__", full_key, display_value, desc if desc else ""))

            pending_desc = None

    return rows

def build_tables_from_yaml(filepath):
    console = Console()
    rows = parse_yaml_as_stream(filepath)

    # Calculate 60% width of terminal
    term_width = shutil.get_terminal_size().columns
    table_width = int(term_width * 0.6)

    table = None

    for row in rows:
        row_type, key, val, desc = row

        if row_type == "__SECTION__":
            # Print current table if it exists
            if table:
                console.print(table)
                console.print()  # blank line between tables

            # Section heading
            console.print(Text(key, style="bold yellow", justify="center"))
            table = Table(show_lines=True, width=table_width)
            table.add_column("Field", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            table.add_column("Description", style="green")

        elif row_type == "__FIELD__" and val != "":
            # If no table yet, create a default one
            if table is None:
                console.print(Text("General", style="bold yellow", justify="center"))
                table = Table(show_lines=True, width=table_width)
                table.add_column("Field", style="cyan", no_wrap=True)
                table.add_column("Value", style="magenta")
                table.add_column("Description", style="green")

            table.add_row(key, val, desc)

    # Print last table
    if table:
        console.print(table)


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "samples/yolov9c.yaml"
    build_tables_from_yaml(filepath)

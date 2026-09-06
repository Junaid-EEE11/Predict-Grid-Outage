from pathlib import Path
from typing import Dict, Any, List
import pandas as pd

class PublicationTableGenerator:
    """Generates formatted publication Tables 1 to 8 in Markdown, CSV, and LaTeX."""

    @staticmethod
    def export_table(df: pd.DataFrame, table_id: str, title: str, output_dir: str = "results/tables") -> str:
        """Export DataFrame to Markdown, CSV, and LaTeX."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = out_path / f"{table_id}.csv"
        md_file = out_path / f"{table_id}.md"
        tex_file = out_path / f"{table_id}.tex"

        df.to_csv(csv_file, index=False)
        
        md_content = f"### {title}\n\n" + df.to_markdown(index=False)
        md_file.write_text(md_content, encoding="utf-8")

        # Basic LaTeX tabular
        tex_content = df.to_latex(index=False)
        tex_file.write_text(tex_content, encoding="utf-8")

        return md_content

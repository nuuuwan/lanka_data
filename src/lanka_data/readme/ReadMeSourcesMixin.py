class ReadMeSourcesMixin:
    def get_lines_for_sources(self, output_idx):

        lines = [
            "## Data Sources",
            "",
        ]
        source_idx = {}
        for output in output_idx.values():
            if "result" in output:
                result = output["result"]
                source = result["source"]
                source_url = result["source_url"]
                source_idx[source] = source_url

        for source, source_url in sorted(source_idx.items()):
            lines.append(f"- [{source}]({source_url})")
        lines.append("")
        return lines

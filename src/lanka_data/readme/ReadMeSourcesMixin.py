class ReadMeSourcesMixin:
    def get_lines_for_sources(self, output_idx):

        lines = [
            "## 1. Data Sources",
            "",
        ]
        source_idx = {}
        for output in output_idx.values():
            for source in output["sources"]:
                source_name = source["name"]
                source_url = source["url"]
                source_idx[source_name] = source_url

        for source, source_url in sorted(source_idx.items()):
            lines.append(f"- [{source}]({source_url})")
        lines.append("")
        return lines

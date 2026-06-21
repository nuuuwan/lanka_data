from dataclasses import dataclass


@dataclass
class DataSource:
    name: str
    url: str

    @classmethod
    def merge_datasource_list_of_lists(cls, datasource_list_of_lists):
        combined_list = [
            item for sublist in datasource_list_of_lists for item in sublist
        ]
        seen = set()
        unique_combined_list = []
        for item in combined_list:
            identifier = (item.name, item.url)
            if identifier not in seen:
                seen.add(identifier)
                unique_combined_list.append(item)
        sorted_unique_combined_list = sorted(
            unique_combined_list, key=lambda x: x.name
        )
        return sorted_unique_combined_list

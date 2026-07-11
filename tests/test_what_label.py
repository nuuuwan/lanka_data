from lanka_data.datasets.what_label.WhatLabel import WhatLabel


class TestWhatLabel:
    def test_list_returns_what_labels(self):
        labels = WhatLabel.list()
        assert len(labels) > 0
        assert all(isinstance(x, WhatLabel) for x in labels)

    def test_list_includes_shared_census_labels(self):
        names = {str(x) for x in WhatLabel.list()}
        assert "Religion" in names
        assert "Ethnicity" in names

    def test_labels_have_description_and_categories(self):
        for what_label in WhatLabel.list():
            assert what_label.description
            assert len(what_label.category_labels) > 0

    def test_from_label_resolves_known_label(self):
        religion = WhatLabel.from_label("Religion")
        assert religion is not None
        assert religion.category_labels == ["census"]

    def test_from_label_returns_none_for_unknown(self):
        assert WhatLabel.from_label("DoesNotExist") is None

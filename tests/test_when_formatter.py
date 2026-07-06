from lanka_data.visual.formatters.WhenFormatter import WhenFormatter


class TestWhenFormatter:
    def test_empty(self):
        assert WhenFormatter("").format() is None

    def test_year(self):
        assert WhenFormatter("2024").format() == "2024"

    def test_interval(self):
        assert WhenFormatter("2011-2024").format() == "2011\u20132024"

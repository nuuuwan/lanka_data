import numpy as np

from lanka_data.visual.plot.QRCode import QRCode
from lanka_data.visual.plot.Style import Style


class TestQRCode:
    def test_url_points_to_code_repo(self):
        assert QRCode.URL == Style.BRAND_URL

    def test_matrix_is_square_and_bordered(self):
        matrix = QRCode(None)._matrix()
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]
        assert np.all(matrix[0] == 0)
        assert np.all(matrix[-1] == 0)

    def test_matrix_encodes_url(self):
        try:
            from PIL import Image
            from pyzbar.pyzbar import decode
        except Exception:
            return

        scale = 8
        matrix = QRCode(None)._matrix()
        image = Image.fromarray(np.uint8((1 - matrix) * 255)).resize(
            (matrix.shape[1] * scale, matrix.shape[0] * scale),
            Image.NEAREST,
        )
        results = decode(image)
        assert any(r.data.decode() == Style.BRAND_URL for r in results)

from lanka_data.visual.plot.Style import Style


class InnerSquare:
    top = Style.BODY_TOP
    bottom = Style.BODY_BOTTOM
    height = top - bottom
    width = height
    left = (1 - width) / 2
    right = left + width

from lanka_data.what.What import What


class BasicWhat(What):

    def get_result(self, regions):
        return regions.get_result()

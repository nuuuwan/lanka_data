NAMES = {
    "41399660": "Mahaweli Ganga",
    "41407423": "Kalu Ganga",
    "41403190": "Deduru Oya",
    "41405515": "Kelani Ganga",
    "41409740": "Walawe Ganga",
    "41397865": "Malwathu Oya",
    "41400542": "Kala Oya",
    "41404427": "Maha Oya",
    "41402045": "Maduru Oya",
    "41404379": "Gal Oya",
    "41410526": "Nilwala Ganga",
    "41397237": "Yan Oya",
    "41409948": "Gin Ganga",
    "41402604": "Mundeni Aru",
    "41408473": "Menik Ganga",
    "41409262": "Kirindi Oya",
    "41401373": "Mi Oya",
    "41408119": "Benthara Ganga",
    "41407753": "Kumbukkan Oya",
    "41397058": "Ma Oya",
    "41399342": "Modaragam Aru",
    "41393543": "Kanagarayan Aru",
    "41403587": "Navakiri Aru",
    "41406267": "Heda Oya",
    "41406571": "Bolgoda",
    "41396222": "Thiruppu Moddai Aru",
    "41409952": "Uruboku Oya",
}


class RiverNames:
    @staticmethod
    def get_name(main_river_id):
        return NAMES.get(str(main_river_id), f"River {main_river_id}")

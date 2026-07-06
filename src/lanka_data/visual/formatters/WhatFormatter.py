class WhatFormatter:
    def __init__(self, what_cmd):
        self.what_cmd = what_cmd

    def format(self) -> str:
        if self.what_cmd == "Empty":
            return None

        return self.what_cmd.replace("+", " & ")

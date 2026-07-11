from pylatex import NoEscape


class ResearchPaperIntroMixin:
    def _add_intro_body(self, sec):
        sec.append(
            NoEscape(
                "The goal of Lanka Data is ``one API to rule them all'': a "
                r'single interface that can express \emph{any} query, rather '
                'than a proliferation of endpoints, methods, libraries, and '
                'parameter sets that each answer one narrow question. Most '
                'data libraries grow by accretion; every new question adds '
                'another function, endpoint, or flag, and no single mental '
                'model survives contact with the result. We wanted the '
                r'opposite: a fixed, minimal grammar that a user learns '
                r'\emph{once} and can then aim at anything, and that a '
                'non-technical user can read and write without learning to '
                r'program. The current domain is public Sri Lankan data---'
                'census measurements, election results, and administrative '
                r'geography---but nothing in the grammar is specific to Sri '
                r'Lanka: \emph{what}, \emph{when}, \emph{where}, and '
                r'\emph{how} are the dimensions of essentially any factual '
                'query about the world.'
            )
        )
        sec.append(
            NoEscape(
                r'Lanka Data is an open-source Python library '
                r'available at \url{' + self.REPO_URL + r'}.'
            )
        )

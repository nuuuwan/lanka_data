import hashlib
import os
import shutil

from pylatex import Figure, NoEscape, Subsection


class ResearchPaperExamplesMixin:
    EXAMPLES = [
        (
            'Empty/2024/LK:province/Map',
            'Base map: provinces',
            'The nine provinces with no measurement bound. '
            'Serves as reference geometry; the Where field '
            'alone controls the region type.',
        ),
        (
            'Gender/2024/LK:dsd/Map',
            'Gender by Divisional Secretariat Division',
            'Dominant gender at DSD level in 2024. '
            'Changing Where from LK:province to LK:dsd '
            'reveals finer spatial structure at no other cost.',
        ),
        (
            'Ethnicity/2024/LK-1127025@4/Map',
            'Proximity query: ethnicity',
            'Ethnic composition of GNDs within 4 km of '
            'GND LK-1127025. The @distance syntax in Where '
            'selects regions by proximity, not by boundary.',
        ),
        (
            'Fuel/2012-2024/LK:district/HexMap',
            'The change in most common cooking fuel',
            'by district across the 2012--2024 interval, '
            'rendered as a hexmap.',
        ),
        (
            'Presidential/2015/LK:ed/HexMap',
            'Presidential election 2015 by electoral division',
            'Winning party in each electoral division in the '
            '2015 presidential election. HexMap normalises '
            'area, giving equal visual weight to each division.',
        ),
        (
            'Religion/2024/LK:district/Map:DiversityPew',
            'Religious diversity index',
            'Pew Research diversity index per district '
            'for 2024. DiversityPew computes the summary '
            'statistic directly, without post-processing.',
        ),
        (
            'Religion/2012-2024/LK-11:dsd/BarChart',
            'Religion in Western Province DSDs (bar chart)',
            'Religious composition across DSDs in Western '
            'Province (LK-11) over 2012--2024. BarChart '
            'shows full distributions, not just a top rank.',
        ),
        (
            'Toilet/2024/LK:district/SquareMap',
            'Toilet facility type by district',
            'Dominant toilet facility type per district '
            'in 2024, as a square map where cell area is '
            'proportional to district population.',
        ),
        (
            'Water/2024/LK:district/TriangleMap',
            'Water source by district (triangle map)',
            'Primary water source per district in 2024, '
            'visualised as a triangle map---a ternary chart '
            'showing proportional distribution across '
            'three water source categories.',
        ),
        (
            'Local/2025/LK:district/Map',
            'Local government elections, 2025',
            'Winning party per district in the 2025 local '
            'elections. Switching What from census to '
            'elections requires no change to the other fields.',
        ),
    ]

    def _add_examples_section(self, sec):
        sec.append(
            NoEscape(
                'The following ten examples each span all four '
                'fields and are produced by a single '
                r'slash-delimited string. The library is at \url{'
                + self.REPO_URL
                + r'}.'
            )
        )
        self._add_examples_figures(sec)

    def _img_path(self, cmd):
        return hashlib.md5(cmd.encode()).hexdigest() + '.png'

    def copy_images(self, tex_dir):
        for cmd, _, _ in self.EXAMPLES:
            src = os.path.join('_output', cmd, 'Image.png')
            dst = os.path.join(tex_dir, self._img_path(cmd))
            shutil.copy(src, dst)

    def _add_examples_figures(self, sec):
        for i, (cmd, heading, description) in enumerate(self.EXAMPLES):
            with sec.create(Subsection(heading)) as ss:
                img = self._img_path(cmd)
                ss.append(NoEscape(r'\path{' + cmd + r'}. ' + description))
                with ss.create(Figure(position='H')) as fig:
                    fig.add_image(img, width=NoEscape(r'\linewidth'))
                    fig.add_caption(NoEscape(heading))
                    fig.append(
                        NoEscape(r'\label{fig:example-' + str(i + 1) + r'}')
                    )

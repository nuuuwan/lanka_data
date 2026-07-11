import os

from pylatex import Command, Document, NoEscape, Package, Section
from pylatex.base_classes import Environment

from lanka_data.latex.ResearchPaperAbstractMixin import \
    ResearchPaperAbstractMixin
from lanka_data.latex.ResearchPaperDatasetsMixin import \
    ResearchPaperDatasetsMixin
from lanka_data.latex.ResearchPaperDesignMixin import ResearchPaperDesignMixin
from lanka_data.latex.ResearchPaperExamplesMixin import \
    ResearchPaperExamplesMixin
from lanka_data.latex.ResearchPaperGrammarMixin import \
    ResearchPaperGrammarMixin
from lanka_data.latex.ResearchPaperIntroMixin import ResearchPaperIntroMixin
from lanka_data.latex.ResearchPaperPreambleMixin import \
    ResearchPaperPreambleMixin
from utils_future import Log

log = Log("ResearchPaper")


class ResearchPaper(
    ResearchPaperPreambleMixin,
    ResearchPaperDesignMixin,
    ResearchPaperAbstractMixin,
    ResearchPaperIntroMixin,
    ResearchPaperGrammarMixin,
    ResearchPaperExamplesMixin,
    ResearchPaperDatasetsMixin,
):
    DEFAULT_PATH = "latex/lanka_data.tex"

    class Abstract(Environment):
        _latex_name = 'abstract'
        escape = False

    def __init__(self, path=None):
        self.path = path or self.DEFAULT_PATH

    def build(self):
        tex_dir = os.path.dirname(self.path)
        if tex_dir and not os.path.exists(tex_dir):
            os.makedirs(tex_dir)
        self._create_bibliography_file(tex_dir)
        self._create_bst_file(tex_dir)
        doc = self._build_document()
        tex_no_ext = os.path.splitext(self.path)[0]
        doc.generate_tex(tex_no_ext)
        log.info(f"Wrote {self.path}")
        return self.path

    def _build_document(self):
        doc = Document(
            documentclass='article',
            document_options=['10pt', 'a4paper'],
            fontenc='T1',
            inputenc='utf8',
            lmodern=True,
            textcomp=True,
            page_numbers=False,
        )
        for pkg in [
            'lastpage',
            'times',
            'natbib',
            'hyperref',
            'graphicx',
            'float',
            'booktabs',
            'amsmath',
            'url',
            'acl',
        ]:
            doc.packages.append(Package(pkg))

        doc.preamble.append(Command('title', NoEscape(self.TITLE)))
        doc.preamble.append(
            Command(
                "author",
                NoEscape(
                    r"Nuwan I. Senaratna\\"
                    r"Independent Researcher\\"
                    r"\vspace{0.25em}\texttt{\href{%s}{%s}}"
                    % (self.AUTHOR_URL, self.AUTHOR_URL)
                ),
            )
        )
        doc.preamble.append(Command("date", NoEscape(r"\today")))
        doc.append(NoEscape(r"\maketitle"))
        self._add_body(doc)
        return doc

    def _add_body(self, doc):
        self._add_abstract(doc)
        with doc.create(Section('Introduction')) as sec:
            self._add_intro_body(sec)
        with doc.create(Section('The Command Grammar')) as sec:
            self._add_grammar_body(sec)
        with doc.create(Section('Datasets and Sources')) as sec:
            self._add_datasets_body(sec)
        with doc.create(Section('Examples')) as sec:
            self._add_examples_section(sec)
        doc.append(NoEscape(r'\bibliographystyle{acl_natbib}'))
        doc.append(NoEscape(r'\bibliography{lanka_data}'))

    def _create_bibliography_file(self, tex_dir):
        """Create BibTeX file in output directory."""
        bib_path = os.path.join(tex_dir, 'lanka_data.bib')
        lines = self._get_bib_content()
        with open(bib_path, 'w') as f:
            f.write('\n'.join(lines))

    def _create_bst_file(self, tex_dir):
        """Create BibTeX style file in output directory."""
        import shutil

        bst_path = os.path.join(tex_dir, 'acl_natbib.bst')
        if os.path.exists(bst_path):
            return
        src = '/usr/local/texlive/2025/texmf-dist/bibtex/bst/natbib/'
        src += 'plainnat.bst'
        if os.path.exists(src):
            shutil.copy(src, bst_path)
        else:
            log.warning(f"Could not find {src}")

    def _get_census_bib_entries(self):
        """Return census bibliography entries."""
        return [
            '@misc{census_2001,',
            '  author={{Department of Census and Statistics}},',
            '  title={Census of Population and Housing 2001},',
            '  howpublished={\\url{https://www.statistics.gov.lk/',
            '    Population/StaticalInformation/CPH2001}},',
            '  year={2001},',
            '  address={Colombo, Sri Lanka},',
            '  urldate={2024-01-01}',
            '}',
            '',
            '@misc{census_2012,',
            '  author={{Department of Census and Statistics}},',
            '  title={Census of Population and Housing 2012},',
            '  howpublished={\\url{https://www.statistics.gov.lk/',
            '    Resource/en/Population/CPH_2011/',
            '    CPH_2012_5Per_Rpt.pdf}},',
            '  year={2012},',
            '  address={Colombo, Sri Lanka},',
            '  note={Enumeration 2011; report published 2012},',
            '  urldate={2024-01-01}',
            '}',
            '',
            '@misc{census_2024,',
            '  author={{Department of Census and Statistics}},',
            '  title={Census of Population and Housing 2024},',
            '  howpublished={\\url{https://www.statistics.gov.lk/',
            '    Population/StaticalInformation/CPH2024}},',
            '  year={2024},',
            '  address={Colombo, Sri Lanka},',
            '  urldate={2024-01-01}',
            '}',
            '',
        ]

    def _get_organizations_bib_entries(self):
        """Return organization bibliography entries."""
        return [
            '@misc{dcs_sri_lanka,',
            '  author={{Department of Census and Statistics}},',
            '  title={Department of Census and Statistics},',
            '  howpublished={\\url{https://www.statistics.gov.lk/}},',
            '  year={2024},',
            '  note={Official statistics portal, Government},',
            '  urldate={2024-01-01}',
            '}',
            '',
            '@misc{elections_commission,',
            '  author={{Election Commission of Sri Lanka}},',
            '  title={Election Commission of Sri Lanka},',
            '  howpublished={\\url{https://www.elections.gov.lk/}},',
            '  year={2024},',
            '  note={Official elections portal, Government},',
            '  urldate={2024-01-01}',
            '}',
            '',
        ]

    def _get_dataset_bib_entries(self):
        """Return dataset bibliography entries."""
        return [
            '@misc{hydrorivers,',
            '  author={Lehner, Bernhard and Grill, Gunther},',
            '  title={{HydroRIVERS}: A Global Vector River Network},',
            '  howpublished={\\url{https://www.hydrosheds.org/',
            '    products/hydrorivers}},',
            '  year={2013},',
            '  note={Part of HydroSHEDS; Sri Lanka subset via',
            '    \\url{https://github.com/nuuuwan/lk_rivers}},',
            '  urldate={2024-01-01}',
            '}',
            '',
        ]

    def _get_bib_content(self):
        """Return content for BibTeX bibliography file."""
        entries = []
        entries.extend(self._get_organizations_bib_entries())
        entries.extend(self._get_census_bib_entries())
        entries.extend(self._get_dataset_bib_entries())
        return entries

from pylatex import NoEscape


class ResearchPaperAbstractMixin:
    def _add_abstract(self, doc):
        with doc.create(self.Abstract()) as abs_env:
            abs_env.append(
                NoEscape(
                    'Lanka Data is a software library that provides a single, '
                    'uniform interface to public data about Sri Lanka. Rather '
                    'than exposing a growing collection of endpoints, methods, '
                    'and parameters, it reduces every query to one string of '
                    r'four positional fields---\emph{what} is measured, '
                    r'\emph{when}, \emph{where}, and \emph{how} it is '
                    'presented---delimited by slashes. The same string serves '
                    'unchanged as a Python argument, '
                    'a command-line argument, a '
                    'URL path, and a file path. This paper describes the '
                    'motivation for the design, '
                    'specifies the four-field command '
                    'grammar and its single intentional coupling, argues that '
                    'the grammar spans the target query space by composition '
                    'rather than by accretion, and catalogues the datasets---'
                    'census, election, administrative geography, and '
                    'hydrology---and their sources that the library exposes. '
                    r'The library is available at \url{'
                    + self.REPO_URL
                    + r'}.'
                )
            )

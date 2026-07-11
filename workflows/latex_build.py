import os
import shutil
import subprocess
from lanka_data.latex.ResearchPaper import ResearchPaper
from utils_future import Log

log = Log("latex_build")


def check_pdflatex():
    """Check if pdflatex is available."""
    if not shutil.which("pdflatex"):
        log.error("pdflatex not found. Install it to build PDFs.")
        return False
    return True


def check_bibtex():
    """Check if bibtex is available."""
    if not shutil.which("bibtex"):
        log.warning("bibtex not found. Bibliography won't be processed.")
        return False
    return True


def log_pdflatex_errors(result, pass_num):
    """Log pdflatex output and errors."""
    if result.returncode != 0:
        msg = f"pdflatex pass {pass_num} exit code: {result.returncode}"
        log.warning(msg)

    if result.stderr:
        log.error(f"stderr: {result.stderr[:200]}")

    for line in (result.stdout or '').split('\n'):
        if 'error' in line.lower() or '!' in line:
            log.error(f"  {line}")


def run_pdflatex(tex_abs_path, tex_dir, pass_num):
    """Run pdflatex once."""
    tex_name = os.path.basename(tex_abs_path)
    log.info(f"Running pdflatex (pass {pass_num}) on {tex_name}")
    env = os.environ.copy()
    env['TEXINPUTS'] = tex_dir + ':'
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_name],
        capture_output=True,
        text=True,
        cwd=tex_dir,
        env=env,
    )
    log_pdflatex_errors(result, pass_num)


def run_bibtex(tex_abs_path, tex_dir):
    """Run bibtex to process bibliography."""
    tex_name = os.path.basename(tex_abs_path).replace(".tex", "")
    log.info(f"Running bibtex on {tex_name}")
    result = subprocess.run(
        ["bibtex", tex_name],
        capture_output=True,
        text=True,
        cwd=tex_dir,
    )
    if result.returncode != 0:
        log.warning(f"bibtex exit code: {result.returncode}")
    if result.stderr:
        log.error(f"bibtex stderr: {result.stderr[:200]}")


def build_pdf(tex_path):
    """Convert LaTeX file to PDF using pdflatex and bibtex."""
    if not check_pdflatex():
        return None

    path_str = str(tex_path).split(' (')[0]
    tex_abs_path = os.path.abspath(path_str)
    tex_dir = os.path.dirname(tex_abs_path)

    run_pdflatex(tex_abs_path, tex_dir, 1)

    if check_bibtex():
        run_bibtex(tex_abs_path, tex_dir)

    run_pdflatex(tex_abs_path, tex_dir, 2)
    run_pdflatex(tex_abs_path, tex_dir, 3)

    pdf_path = tex_abs_path.replace(".tex", ".pdf")
    if os.path.exists(pdf_path):
        log.info(f"PDF created: {pdf_path}")
        return pdf_path

    log.warning("PDF was not created")
    return None


if __name__ == "__main__":
    tex_file = ResearchPaper().build()
    build_pdf(str(tex_file))

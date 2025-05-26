from sphinx_pyproject import SphinxConfig

config = SphinxConfig("../../pyproject.toml", globalns=globals(), style="pep621")

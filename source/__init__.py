__all__ = ['pySurf','dataIO','pyProfile','thermal','utilities']
#__all__ = ['pySurf','dataIO','pyProfile','thermal','utilities']


from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed
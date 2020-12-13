
def pytest_configure(config):
    import traceback
    t = traceback.extract_stack()
    if 'pytestworker.py' in t[0][0]:
        import matplotlib
        matplotlib.use('template')  # suppress plots when Spyder unit-testing

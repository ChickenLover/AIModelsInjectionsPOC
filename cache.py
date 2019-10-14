CACHE_FILE = 'cache.txt'


def cache_results(results):
    with open(CACHE_FILE, 'w') as f:
        f.write(str(results))


def read_cache():
    with open(CACHE_FILE) as f:
        return f.read()

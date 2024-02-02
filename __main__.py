
# Ibrahim Najmudin
# University of Oxford

# This is the main script, and will call the function I desire.
# Will usually be the test functions.

# Import all run files
from unused.test_orion import test_orion
from unused.iterate_fin import iterate_fin
from unused.iterate_avgd import iterate_avgd
from unused.iterate_avgd_full import iterate_avgd_full
from test.orion_tests import iterate_v4
from test.simple_tests import simple_v1, simple_v2, simple_v3
from test.amica_tests import iterate_amica


from common.setup import setup_cross

if __name__ == "__main__":
    iterate_amica()
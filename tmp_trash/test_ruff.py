import os
import sys

def test_function():
    # Unused variable (should trigger F841)
    unused_var = "This is unused"
    
    # Line too long (should trigger E501 if we were to run with the default rules, but we've set line-length to 88)
    very_long_string = "This string is not actually too long for our configuration since we set line-length to 88"
    
    # Import sorting issue (will be flagged by I rules)
    
    return True 
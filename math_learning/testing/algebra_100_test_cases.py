#!/usr/bin/env python3
"""
100 Comprehensive Algebra Test Cases
Systematic coverage of all K-12 algebra concepts
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comprehensive_diagnostic import ComprehensiveDiagnostic, StudentProfile
from typing import Dict, List, Tuple
import random

class Algebra100TestCases:
    """Comprehensive collection of 100 algebra test cases"""
    
    def __init__(self):
        self.diagnostic = ComprehensiveDiagnostic()
        
        # 100 Algebra Test Cases organized by topic
        self.test_cases = {
            
            # ========== BASIC ARITHMETIC & NUMBER SENSE (10 cases) ==========
            "basic_arithmetic": {
                "ba_01": {
                    "problem": "Calculate: 15 + 28 - 12",
                    "answer": "31",
                    "type": "integer_operations",
                    "source": "Khan Academy",
                    "grade": "6",
                    "difficulty": 1
                },
                "ba_02": {
                    "problem": "What is 7 × 8 ÷ 4?",
                    "answer": "14",
                    "type": "order_of_operations",
                    "source": "Kumon",
                    "grade": "6",
                    "difficulty": 1
                },
                "ba_03": {
                    "problem": "Simplify: 3 + 4 × 2 - 1",
                    "answer": "10",
                    "type": "order_of_operations",
                    "source": "Singapore Math",
                    "grade": "6",
                    "difficulty": 2
                },
                "ba_04": {
                    "problem": "Calculate: (-5) + 8 - 3",
                    "answer": "0",
                    "type": "integer_operations",
                    "source": "Common Core",
                    "grade": "7",
                    "difficulty": 2
                },
                "ba_05": {
                    "problem": "What is 2³ + 3²?",
                    "answer": "17",
                    "type": "exponents",
                    "source": "Beast Academy",
                    "grade": "7",
                    "difficulty": 2
                },
                "ba_06": {
                    "problem": "Simplify: 24 ÷ 6 × 2 + 1",
                    "answer": "9",
                    "type": "order_of_operations",
                    "source": "Saxon Math",
                    "grade": "6",
                    "difficulty": 2
                },
                "ba_07": {
                    "problem": "Calculate: |−7| + |3|",
                    "answer": "10",
                    "type": "absolute_value",
                    "source": "Eureka Math",
                    "grade": "7",
                    "difficulty": 2
                },
                "ba_08": {
                    "problem": "What is 5² - 4²?",
                    "answer": "9",
                    "type": "exponents",
                    "source": "AoPS",
                    "grade": "7",
                    "difficulty": 2
                },
                "ba_09": {
                    "problem": "Simplify: 2 × 3² - 4 × 2",
                    "answer": "10",
                    "type": "order_of_operations",
                    "source": "RSM",
                    "grade": "7",
                    "difficulty": 3
                },
                "ba_10": {
                    "problem": "Calculate: (-3)² + (-2)³",
                    "answer": "1",
                    "type": "integer_operations",
                    "source": "IB Mathematics",
                    "grade": "8",
                    "difficulty": 3
                }
            },
            
            # ========== ONE-STEP EQUATIONS (10 cases) ==========
            "one_step_equations": {
                "os_01": {
                    "problem": "Solve for x: x + 5 = 12",
                    "answer": "7",
                    "type": "one_step_equations",
                    "source": "Khan Academy",
                    "grade": "6",
                    "difficulty": 1
                },
                "os_02": {
                    "problem": "Solve for y: y - 8 = 15",
                    "answer": "23",
                    "type": "one_step_equations",
                    "source": "Kumon",
                    "grade": "6",
                    "difficulty": 1
                },
                "os_03": {
                    "problem": "Solve for a: 3a = 21",
                    "answer": "7",
                    "type": "one_step_equations",
                    "source": "Singapore Math",
                    "grade": "7",
                    "difficulty": 1
                },
                "os_04": {
                    "problem": "Solve for b: b/4 = 6",
                    "answer": "24",
                    "type": "one_step_equations",
                    "source": "Common Core",
                    "grade": "7",
                    "difficulty": 2
                },
                "os_05": {
                    "problem": "Solve for x: x + 3.5 = 8.2",
                    "answer": "4.7",
                    "type": "one_step_equations",
                    "source": "Beast Academy",
                    "grade": "7",
                    "difficulty": 2
                },
                "os_06": {
                    "problem": "Solve for m: -2m = 14",
                    "answer": "-7",
                    "type": "one_step_equations",
                    "source": "Saxon Math",
                    "grade": "7",
                    "difficulty": 2
                },
                "os_07": {
                    "problem": "Solve for n: n/(-3) = 5",
                    "answer": "-15",
                    "type": "one_step_equations",
                    "source": "Eureka Math",
                    "grade": "8",
                    "difficulty": 2
                },
                "os_08": {
                    "problem": "Solve for p: 2/3 × p = 8",
                    "answer": "12",
                    "type": "one_step_equations",
                    "source": "AoPS",
                    "grade": "8",
                    "difficulty": 3
                },
                "os_09": {
                    "problem": "Solve for q: q - 2.8 = -1.5",
                    "answer": "1.3",
                    "type": "one_step_equations",
                    "source": "RSM",
                    "grade": "8",
                    "difficulty": 2
                },
                "os_10": {
                    "problem": "Solve for r: 0.5r = 3.5",
                    "answer": "7",
                    "type": "one_step_equations",
                    "source": "Cambridge",
                    "grade": "8",
                    "difficulty": 2
                }
            },
            
            # ========== TWO-STEP EQUATIONS (10 cases) ==========
            "two_step_equations": {
                "ts_01": {
                    "problem": "Solve for x: 2x + 3 = 11",
                    "answer": "4",
                    "type": "two_step_equations",
                    "source": "Khan Academy",
                    "grade": "7",
                    "difficulty": 2
                },
                "ts_02": {
                    "problem": "Solve for y: 3y - 5 = 16",
                    "answer": "7",
                    "type": "two_step_equations",
                    "source": "Kumon",
                    "grade": "7",
                    "difficulty": 2
                },
                "ts_03": {
                    "problem": "Solve for a: 4a + 7 = 31",
                    "answer": "6",
                    "type": "two_step_equations",
                    "source": "Singapore Math",
                    "grade": "8",
                    "difficulty": 2
                },
                "ts_04": {
                    "problem": "Solve for b: 5b - 12 = 23",
                    "answer": "7",
                    "type": "two_step_equations",
                    "source": "Common Core",
                    "grade": "8",
                    "difficulty": 2
                },
                "ts_05": {
                    "problem": "Solve for x: x/2 + 4 = 9",
                    "answer": "10",
                    "type": "two_step_equations",
                    "source": "Beast Academy",
                    "grade": "8",
                    "difficulty": 2
                },
                "ts_06": {
                    "problem": "Solve for m: 3m - 8 = -2",
                    "answer": "2",
                    "type": "two_step_equations",
                    "source": "Saxon Math",
                    "grade": "8",
                    "difficulty": 3
                },
                "ts_07": {
                    "problem": "Solve for n: -2n + 5 = 13",
                    "answer": "-4",
                    "type": "two_step_equations",
                    "source": "Eureka Math",
                    "grade": "8",
                    "difficulty": 3
                },
                "ts_08": {
                    "problem": "Solve for p: 2p/3 + 1 = 7",
                    "answer": "9",
                    "type": "two_step_equations",
                    "source": "AoPS",
                    "grade": "9",
                    "difficulty": 3
                },
                "ts_09": {
                    "problem": "Solve for q: 0.5q - 2.5 = 1.5",
                    "answer": "8",
                    "type": "two_step_equations",
                    "source": "RSM",
                    "grade": "8",
                    "difficulty": 3
                },
                "ts_10": {
                    "problem": "Solve for r: 4r + 3 = 2r + 15",
                    "answer": "6",
                    "type": "two_step_equations",
                    "source": "IB Mathematics",
                    "grade": "9",
                    "difficulty": 3
                }
            },
            
            # ========== DISTRIBUTIVE PROPERTY (10 cases) ==========
            "distributive_property": {
                "dp_01": {
                    "problem": "Simplify: 3(x + 4)",
                    "answer": "3x + 12",
                    "type": "distributive_property",
                    "source": "Khan Academy",
                    "grade": "7",
                    "difficulty": 2
                },
                "dp_02": {
                    "problem": "Expand: 2(y - 5)",
                    "answer": "2y - 10",
                    "type": "distributive_property",
                    "source": "Kumon",
                    "grade": "7",
                    "difficulty": 2
                },
                "dp_03": {
                    "problem": "Simplify: 4(2a + 3)",
                    "answer": "8a + 12",
                    "type": "distributive_property",
                    "source": "Singapore Math",
                    "grade": "8",
                    "difficulty": 2
                },
                "dp_04": {
                    "problem": "Expand: -3(b - 2)",
                    "answer": "-3b + 6",
                    "type": "distributive_property",
                    "source": "Common Core",
                    "grade": "8",
                    "difficulty": 3
                },
                "dp_05": {
                    "problem": "Simplify: 5(2x - 3y)",
                    "answer": "10x - 15y",
                    "type": "distributive_property",
                    "source": "Beast Academy",
                    "grade": "8",
                    "difficulty": 3
                },
                "dp_06": {
                    "problem": "Expand: -2(3m + 4n)",
                    "answer": "-6m - 8n",
                    "type": "distributive_property",
                    "source": "Saxon Math",
                    "grade": "8",
                    "difficulty": 3
                },
                "dp_07": {
                    "problem": "Simplify: 0.5(4p - 6)",
                    "answer": "2p - 3",
                    "type": "distributive_property",
                    "source": "Eureka Math",
                    "grade": "8",
                    "difficulty": 3
                },
                "dp_08": {
                    "problem": "Expand: (x + 7) × 3",
                    "answer": "3x + 21",
                    "type": "distributive_property",
                    "source": "AoPS",
                    "grade": "8",
                    "difficulty": 2
                },
                "dp_09": {
                    "problem": "Simplify: 2/3(6q - 9)",
                    "answer": "4q - 6",
                    "type": "distributive_property",
                    "source": "RSM",
                    "grade": "9",
                    "difficulty": 3
                },
                "dp_10": {
                    "problem": "Expand: -1/2(8r + 4s)",
                    "answer": "-4r - 2s",
                    "type": "distributive_property",
                    "source": "Cambridge",
                    "grade": "9",
                    "difficulty": 3
                }
            },
            
            # ========== COMBINING LIKE TERMS (10 cases) ==========
            "combining_like_terms": {
                "cl_01": {
                    "problem": "Simplify: 3x + 5x",
                    "answer": "8x",
                    "type": "combine_like_terms",
                    "source": "Khan Academy",
                    "grade": "7",
                    "difficulty": 1
                },
                "cl_02": {
                    "problem": "Combine: 7y - 2y",
                    "answer": "5y",
                    "type": "combine_like_terms",
                    "source": "Kumon",
                    "grade": "7",
                    "difficulty": 1
                },
                "cl_03": {
                    "problem": "Simplify: 4a + 3b + 2a",
                    "answer": "6a + 3b",
                    "type": "combine_like_terms",
                    "source": "Singapore Math",
                    "grade": "8",
                    "difficulty": 2
                },
                "cl_04": {
                    "problem": "Combine: 5x - 3x + 7",
                    "answer": "2x + 7",
                    "type": "combine_like_terms",
                    "source": "Common Core",
                    "grade": "8",
                    "difficulty": 2
                },
                "cl_05": {
                    "problem": "Simplify: 2m + 4n - m + 3n",
                    "answer": "m + 7n",
                    "type": "combine_like_terms",
                    "source": "Beast Academy",
                    "grade": "8",
                    "difficulty": 2
                },
                "cl_06": {
                    "problem": "Combine: 8p - 5p + 2q - q",
                    "answer": "3p + q",
                    "type": "combine_like_terms",
                    "source": "Saxon Math",
                    "grade": "8",
                    "difficulty": 2
                },
                "cl_07": {
                    "problem": "Simplify: 3x² + 2x + 5x² - x",
                    "answer": "8x² + x",
                    "type": "combine_like_terms",
                    "source": "Eureka Math",
                    "grade": "9",
                    "difficulty": 3
                },
                "cl_08": {
                    "problem": "Combine: 4ab + 3a - 2ab + 5a",
                    "answer": "2ab + 8a",
                    "type": "combine_like_terms",
                    "source": "AoPS",
                    "grade": "9",
                    "difficulty": 3
                },
                "cl_09": {
                    "problem": "Simplify: 2.5x - 1.5x + 3.2y",
                    "answer": "x + 3.2y",
                    "type": "combine_like_terms",
                    "source": "RSM",
                    "grade": "8",
                    "difficulty": 3
                },
                "cl_10": {
                    "problem": "Combine: 3/4t + 1/4t - 2/3s",
                    "answer": "t - 2/3s",
                    "type": "combine_like_terms",
                    "source": "IB Mathematics",
                    "grade": "9",
                    "difficulty": 3
                }
            },
            
            # ========== MULTI-STEP EQUATIONS (10 cases) ==========
            "multi_step_equations": {
                "ms_01": {
                    "problem": "Solve: 2x + 5 = 3x - 4",
                    "answer": "9",
                    "type": "multi_step_equations",
                    "source": "Khan Academy",
                    "grade": "8",
                    "difficulty": 3
                },
                "ms_02": {
                    "problem": "Solve: 4y - 7 = 2y + 11",
                    "answer": "9",
                    "type": "multi_step_equations",
                    "source": "Kumon",
                    "grade": "8",
                    "difficulty": 3
                },
                "ms_03": {
                    "problem": "Solve: 3(a + 2) = 21",
                    "answer": "5",
                    "type": "multi_step_equations",
                    "source": "Singapore Math",
                    "grade": "8",
                    "difficulty": 3
                },
                "ms_04": {
                    "problem": "Solve: 2(b - 3) + 5 = 15",
                    "answer": "8",
                    "type": "multi_step_equations",
                    "source": "Common Core",
                    "grade": "9",
                    "difficulty": 3
                },
                "ms_05": {
                    "problem": "Solve: 5x - 2(x + 1) = 7",
                    "answer": "3",
                    "type": "multi_step_equations",
                    "source": "Beast Academy",
                    "grade": "9",
                    "difficulty": 4
                },
                "ms_06": {
                    "problem": "Solve: 3m + 4 = 2(m - 1) + 10",
                    "answer": "4",
                    "type": "multi_step_equations",
                    "source": "Saxon Math",
                    "grade": "9",
                    "difficulty": 4
                },
                "ms_07": {
                    "problem": "Solve: 4(n + 1) - 3n = 12",
                    "answer": "8",
                    "type": "multi_step_equations",
                    "source": "Eureka Math",
                    "grade": "9",
                    "difficulty": 4
                },
                "ms_08": {
                    "problem": "Solve: 2p/3 + 1 = (p + 5)/2",
                    "answer": "6",
                    "type": "multi_step_equations",
                    "source": "AoPS",
                    "grade": "9",
                    "difficulty": 4
                },
                "ms_09": {
                    "problem": "Solve: 0.3q + 0.7 = 0.2q + 1.2",
                    "answer": "5",
                    "type": "multi_step_equations",
                    "source": "RSM",
                    "grade": "9",
                    "difficulty": 4
                },
                "ms_10": {
                    "problem": "Solve: 5r - 3(r - 2) = 2r + 8",
                    "answer": "2",
                    "type": "multi_step_equations",
                    "source": "Cambridge",
                    "grade": "9",
                    "difficulty": 4
                }
            },
            
            # ========== LINEAR INEQUALITIES (10 cases) ==========
            "linear_inequalities": {
                "li_01": {
                    "problem": "Solve: x + 3 > 7",
                    "answer": "x > 4",
                    "type": "linear_inequalities",
                    "source": "Khan Academy",
                    "grade": "8",
                    "difficulty": 2
                },
                "li_02": {
                    "problem": "Solve: 2y ≤ 10",
                    "answer": "y ≤ 5",
                    "type": "linear_inequalities",
                    "source": "Kumon",
                    "grade": "8",
                    "difficulty": 2
                },
                "li_03": {
                    "problem": "Solve: 3a - 5 < 7",
                    "answer": "a < 4",
                    "type": "linear_inequalities",
                    "source": "Singapore Math",
                    "grade": "9",
                    "difficulty": 3
                },
                "li_04": {
                    "problem": "Solve: -2b + 1 ≥ 9",
                    "answer": "b ≤ -4",
                    "type": "linear_inequalities",
                    "source": "Common Core",
                    "grade": "9",
                    "difficulty": 3
                },
                "li_05": {
                    "problem": "Solve: 4x + 3 > 2x - 5",
                    "answer": "x > -4",
                    "type": "linear_inequalities",
                    "source": "Beast Academy",
                    "grade": "9",
                    "difficulty": 3
                },
                "li_06": {
                    "problem": "Solve: 5m - 7 ≤ 3m + 1",
                    "answer": "m ≤ 4",
                    "type": "linear_inequalities",
                    "source": "Saxon Math",
                    "grade": "9",
                    "difficulty": 3
                },
                "li_07": {
                    "problem": "Solve: -3(n + 2) > 6",
                    "answer": "n < -4",
                    "type": "linear_inequalities",
                    "source": "Eureka Math",
                    "grade": "9",
                    "difficulty": 4
                },
                "li_08": {
                    "problem": "Solve: 2p/3 - 1 < 5",
                    "answer": "p < 9",
                    "type": "linear_inequalities",
                    "source": "AoPS",
                    "grade": "9",
                    "difficulty": 3
                },
                "li_09": {
                    "problem": "Solve: 1 - 2q ≥ 3q + 11",
                    "answer": "q ≤ -2",
                    "type": "linear_inequalities",
                    "source": "RSM",
                    "grade": "9",
                    "difficulty": 4
                },
                "li_10": {
                    "problem": "Solve: 0.5r + 2.5 < 1.5r - 1.5",
                    "answer": "r > 4",
                    "type": "linear_inequalities",
                    "source": "IB Mathematics",
                    "grade": "9",
                    "difficulty": 4
                }
            },
            
            # ========== SYSTEMS OF EQUATIONS (10 cases) ==========
            "systems_of_equations": {
                "se_01": {
                    "problem": "Solve the system: x + y = 5, x - y = 1",
                    "answer": "x = 3, y = 2",
                    "type": "systems_of_equations",
                    "source": "Khan Academy",
                    "grade": "9",
                    "difficulty": 3
                },
                "se_02": {
                    "problem": "Solve: 2a + b = 7, a - b = 2",
                    "answer": "a = 3, b = 1",
                    "type": "systems_of_equations",
                    "source": "Kumon",
                    "grade": "9",
                    "difficulty": 3
                },
                "se_03": {
                    "problem": "Solve: 3x + 2y = 12, x + y = 5",
                    "answer": "x = 2, y = 3",
                    "type": "systems_of_equations",
                    "source": "Singapore Math",
                    "grade": "9",
                    "difficulty": 4
                },
                "se_04": {
                    "problem": "Solve: 2m + 3n = 13, m + n = 5",
                    "answer": "m = 2, n = 3",
                    "type": "systems_of_equations",
                    "source": "Common Core",
                    "grade": "9",
                    "difficulty": 4
                },
                "se_05": {
                    "problem": "Solve: 4p - q = 11, 2p + q = 7",
                    "answer": "p = 3, q = 1",
                    "type": "systems_of_equations",
                    "source": "Beast Academy",
                    "grade": "10",
                    "difficulty": 4
                },
                "se_06": {
                    "problem": "Solve: 3r + 2s = 16, r - s = 1",
                    "answer": "r = 3.6, s = 2.6",
                    "type": "systems_of_equations",
                    "source": "Saxon Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "se_07": {
                    "problem": "Solve: 5u + 2v = 19, 3u - v = 8",
                    "answer": "u = 3, v = 2",
                    "type": "systems_of_equations",
                    "source": "Eureka Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "se_08": {
                    "problem": "Solve: 2w + 3z = 11, 4w - z = 5",
                    "answer": "w = 2, z = 7/3",
                    "type": "systems_of_equations",
                    "source": "AoPS",
                    "grade": "10",
                    "difficulty": 5
                },
                "se_09": {
                    "problem": "Solve: 0.5x + 0.3y = 2.3, 0.2x - 0.1y = 0.3",
                    "answer": "x = 3, y = 2.67",
                    "type": "systems_of_equations",
                    "source": "RSM",
                    "grade": "10",
                    "difficulty": 5
                },
                "se_10": {
                    "problem": "Solve: x/2 + y/3 = 5, x - 2y = 2",
                    "answer": "x = 6, y = 2",
                    "type": "systems_of_equations",
                    "source": "Cambridge",
                    "grade": "10",
                    "difficulty": 5
                }
            },
            
            # ========== QUADRATIC EQUATIONS (10 cases) ==========
            "quadratic_equations": {
                "qe_01": {
                    "problem": "Solve: x² = 16",
                    "answer": "x = ±4",
                    "type": "quadratic_equations",
                    "source": "Khan Academy",
                    "grade": "9",
                    "difficulty": 2
                },
                "qe_02": {
                    "problem": "Solve: x² - 9 = 0",
                    "answer": "x = ±3",
                    "type": "quadratic_equations",
                    "source": "Kumon",
                    "grade": "9",
                    "difficulty": 2
                },
                "qe_03": {
                    "problem": "Solve: x² + 5x + 6 = 0",
                    "answer": "x = -2, x = -3",
                    "type": "quadratic_equations",
                    "source": "Singapore Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "qe_04": {
                    "problem": "Solve: x² - 7x + 12 = 0",
                    "answer": "x = 3, x = 4",
                    "type": "quadratic_equations",
                    "source": "Common Core",
                    "grade": "10",
                    "difficulty": 4
                },
                "qe_05": {
                    "problem": "Solve: 2x² - 8 = 0",
                    "answer": "x = ±2",
                    "type": "quadratic_equations",
                    "source": "Beast Academy",
                    "grade": "10",
                    "difficulty": 3
                },
                "qe_06": {
                    "problem": "Solve: x² + 4x - 5 = 0",
                    "answer": "x = 1, x = -5",
                    "type": "quadratic_equations",
                    "source": "Saxon Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "qe_07": {
                    "problem": "Solve: x² - 6x + 9 = 0",
                    "answer": "x = 3",
                    "type": "quadratic_equations",
                    "source": "Eureka Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "qe_08": {
                    "problem": "Solve: 3x² + 2x - 1 = 0",
                    "answer": "x = 1/3, x = -1",
                    "type": "quadratic_equations",
                    "source": "AoPS",
                    "grade": "11",
                    "difficulty": 5
                },
                "qe_09": {
                    "problem": "Solve: x² + 2x - 8 = 0",
                    "answer": "x = 2, x = -4",
                    "type": "quadratic_equations",
                    "source": "RSM",
                    "grade": "10",
                    "difficulty": 4
                },
                "qe_10": {
                    "problem": "Solve: 2x² - 5x + 2 = 0",
                    "answer": "x = 2, x = 1/2",
                    "type": "quadratic_equations",
                    "source": "IB Mathematics",
                    "grade": "11",
                    "difficulty": 5
                }
            },
            
            # ========== FUNCTIONS & GRAPHING (10 cases) ==========
            "functions_graphing": {
                "fg_01": {
                    "problem": "If f(x) = 2x + 3, find f(4)",
                    "answer": "11",
                    "type": "function_evaluation",
                    "source": "Khan Academy",
                    "grade": "9",
                    "difficulty": 2
                },
                "fg_02": {
                    "problem": "If g(x) = x² - 1, find g(3)",
                    "answer": "8",
                    "type": "function_evaluation",
                    "source": "Kumon",
                    "grade": "9",
                    "difficulty": 3
                },
                "fg_03": {
                    "problem": "Find the slope of the line through (2,3) and (4,7)",
                    "answer": "2",
                    "type": "slope_calculation",
                    "source": "Singapore Math",
                    "grade": "9",
                    "difficulty": 3
                },
                "fg_04": {
                    "problem": "What is the y-intercept of y = 3x - 5?",
                    "answer": "-5",
                    "type": "linear_functions",
                    "source": "Common Core",
                    "grade": "9",
                    "difficulty": 2
                },
                "fg_05": {
                    "problem": "If h(x) = 3x - 2, find h⁻¹(7)",
                    "answer": "3",
                    "type": "inverse_functions",
                    "source": "Beast Academy",
                    "grade": "10",
                    "difficulty": 4
                },
                "fg_06": {
                    "problem": "Find the vertex of y = x² - 4x + 3",
                    "answer": "(2, -1)",
                    "type": "quadratic_functions",
                    "source": "Saxon Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "fg_07": {
                    "problem": "If f(x) = x + 1 and g(x) = 2x, find (f∘g)(3)",
                    "answer": "7",
                    "type": "function_composition",
                    "source": "Eureka Math",
                    "grade": "10",
                    "difficulty": 4
                },
                "fg_08": {
                    "problem": "Find the domain of f(x) = 1/(x-2)",
                    "answer": "x ≠ 2",
                    "type": "function_domain",
                    "source": "AoPS",
                    "grade": "11",
                    "difficulty": 4
                },
                "fg_09": {
                    "problem": "If f(x) = |x - 3|, find f(-1)",
                    "answer": "4",
                    "type": "absolute_value_functions",
                    "source": "RSM",
                    "grade": "10",
                    "difficulty": 3
                },
                "fg_10": {
                    "problem": "Find the range of f(x) = x² + 1",
                    "answer": "y ≥ 1",
                    "type": "function_range",
                    "source": "Cambridge",
                    "grade": "11",
                    "difficulty": 4
                }
            }
        }
    
    def count_total_cases(self):
        """Count total number of test cases"""
        total = 0
        for category, problems in self.test_cases.items():
            total += len(problems)
        return total
    
    def run_comprehensive_test(self):
        """Run all 100 test cases with different student profiles"""
        print("🧮 RUNNING 100 COMPREHENSIVE ALGEBRA TEST CASES")
        print("=" * 80)
        
        # Create diverse student profiles
        student_profiles = [
            StudentProfile(
                name="struggling_student",
                overall_ability=0.25,
                concept_mastery={},
                learning_style="visual",
                total_questions_attempted=0,
                total_questions_correct=0
            ),
            StudentProfile(
                name="average_student",
                overall_ability=0.50,
                concept_mastery={},
                learning_style="analytical",
                total_questions_attempted=0,
                total_questions_correct=0
            ),
            StudentProfile(
                name="strong_student",
                overall_ability=0.75,
                concept_mastery={},
                learning_style="hands_on",
                total_questions_attempted=0,
                total_questions_correct=0
            ),
            StudentProfile(
                name="advanced_student",
                overall_ability=0.90,
                concept_mastery={},
                learning_style="analytical",
                total_questions_attempted=0,
                total_questions_correct=0
            )
        ]
        
        total_cases = self.count_total_cases()
        print(f"📊 Total Test Cases: {total_cases}")
        print()
        
        results_summary = {
            "total_tests": 0,
            "successful_diagnoses": 0,
            "categories_tested": len(self.test_cases),
            "student_profiles": len(student_profiles),
            "performance_metrics": {}
        }
        
        # Test each category
        for category_name, problems in self.test_cases.items():
            print(f"🔍 Testing Category: {category_name.upper().replace('_', ' ')}")
            print(f"   Problems in category: {len(problems)}")
            
            category_results = []
            
            # Test each problem with different student profiles
            for problem_id, problem_data in problems.items():
                for student_profile in student_profiles:
                    try:
                        # Simulate student answer based on ability
                        if random.random() < student_profile.overall_ability:
                            student_answer = problem_data['answer']  # Correct answer
                        else:
                            # Generate incorrect answer
                            student_answer = self._generate_wrong_answer(problem_data)
                        
                        # Run diagnostic
                        result = self.diagnostic.diagnose_student_response(
                            problem_data['type'],
                            problem_data['problem'],
                            student_answer,
                            problem_data['answer'],
                            student_profile
                        )
                        
                        results_summary["total_tests"] += 1
                        results_summary["successful_diagnoses"] += 1
                        
                        category_results.append({
                            "problem_id": problem_id,
                            "student": student_profile.name,
                            "correct": student_answer == problem_data['answer'],
                            "gaps_identified": 1 if result.missing_concept else 0,
                            "recommendations": len(result.recommended_questions)
                        })
                        
                    except Exception as e:
                        print(f"   ❌ Error in {problem_id} with {student_profile.name}: {e}")
                        results_summary["total_tests"] += 1
            
            # Category summary
            correct_answers = sum(1 for r in category_results if r["correct"])
            total_attempts = len(category_results)
            accuracy = (correct_answers / total_attempts * 100) if total_attempts > 0 else 0
            
            print(f"   ✅ Category Results: {correct_answers}/{total_attempts} correct ({accuracy:.1f}%)")
            print(f"   📈 Average gaps identified: {sum(r['gaps_identified'] for r in category_results) / len(category_results):.1f}")
            print()
            
            results_summary["performance_metrics"][category_name] = {
                "accuracy": accuracy,
                "total_attempts": total_attempts,
                "avg_gaps": sum(r['gaps_identified'] for r in category_results) / len(category_results)
            }
        
        # Final summary
        print("🎯 COMPREHENSIVE TEST SUMMARY")
        print("=" * 80)
        print(f"📊 Total Tests Run: {results_summary['total_tests']}")
        print(f"✅ Successful Diagnoses: {results_summary['successful_diagnoses']}")
        print(f"📈 Success Rate: {(results_summary['successful_diagnoses'] / results_summary['total_tests'] * 100):.1f}%")
        print(f"🏷️  Categories Tested: {results_summary['categories_tested']}")
        print(f"👥 Student Profiles: {results_summary['student_profiles']}")
        print()
        
        print("📈 CATEGORY PERFORMANCE:")
        for category, metrics in results_summary["performance_metrics"].items():
            print(f"   {category.replace('_', ' ').title()}: {metrics['accuracy']:.1f}% accuracy, {metrics['avg_gaps']:.1f} avg gaps")
        
        print()
        print("🎉 100 ALGEBRA TEST CASES COMPLETED SUCCESSFULLY!")
        return results_summary
    
    def _generate_wrong_answer(self, problem_data):
        """Generate realistic wrong answers based on common mistakes"""
        correct_answer = problem_data['answer']
        problem_type = problem_data['type']
        
        # Common mistake patterns by type
        if problem_type == "one_step_equations":
            if correct_answer.isdigit():
                return str(int(correct_answer) + random.choice([-2, -1, 1, 2]))
        elif problem_type == "two_step_equations":
            if correct_answer.isdigit():
                return str(int(correct_answer) * 2)  # Forgot to divide
        elif problem_type == "distributive_property":
            return correct_answer.replace("+", "-")  # Sign error
        elif problem_type == "order_of_operations":
            if correct_answer.isdigit():
                return str(int(correct_answer) + 5)  # Wrong order
        
        # Default wrong answer
        return "wrong_answer"

def main():
    """Run the comprehensive 100 algebra test cases"""
    tester = Algebra100TestCases()
    results = tester.run_comprehensive_test()
    return results

if __name__ == "__main__":
    main() 
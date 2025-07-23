# Knowledge Graph Tests - Detailed Explanation

## Graph Structure Overview

The knowledge graph tests use several different graph structures to test various algorithms and functionality. Here's a detailed breakdown:

## 1. Main Test Graph (Algorithm Testing)

### Graph Structure
```
        A (Root)
       / \
      B   C
     /|   |\
    D |   | E
   /  |   |  \
  F   |   |   G
      |   |
       \ /
        E
```

### Nodes (10 total)
| Node ID | Name | Difficulty | Time to Master | Category | Description |
|---------|------|------------|----------------|----------|-------------|
| A | Concept A | 1 | 10 min | Foundation | Root concept - starting point |
| B | Concept B | 2 | 15 min | Basic | Branch 1 level 1 |
| C | Concept C | 2 | 15 min | Basic | Branch 2 level 1 |
| D | Concept D | 3 | 20 min | Intermediate | Branch 1 level 2 |
| E | Concept E | 3 | 20 min | Intermediate | Branch 2 level 2 |
| F | Concept F | 4 | 25 min | Advanced | Branch 1 terminus |
| G | Concept G | 4 | 25 min | Advanced | Branch 2 terminus |
| H | Concept H | 2 | 15 min | Other | Separate component root |
| I | Concept I | 3 | 20 min | Other | Separate component leaf |
| J | Concept J | 2 | 15 min | Isolated | Completely isolated node |

### Edges (9 total)
| From | To | Strength | Type | Description |
|------|-------|----------|------|-------------|
| A | B | 0.9 | Main Path | Strong prerequisite relationship |
| A | C | 0.9 | Main Path | Strong prerequisite relationship |
| B | D | 0.8 | Main Path | Medium prerequisite relationship |
| C | E | 0.8 | Main Path | Medium prerequisite relationship |
| D | F | 0.7 | Main Path | Weaker prerequisite relationship |
| E | G | 0.7 | Main Path | Weaker prerequisite relationship |
| B | E | 0.6 | Cross Connection | Cross-branch connection |
| C | D | 0.6 | Cross Connection | Cross-branch connection |
| H | I | 0.8 | Separate Component | Isolated component |

### Graph Components
1. **Main Component**: A, B, C, D, E, F, G (7 nodes, 8 edges)
2. **Separate Component**: H, I (2 nodes, 1 edge)  
3. **Isolated Node**: J (1 node, 0 edges)

## 2. Test Categories and Expected Results

### A. Graph Traversal Tests

#### Breadth-First Search (BFS)
- **Starting Node**: A
- **Expected Order**: Level-by-level traversal
  - Level 0: A
  - Level 1: B, C
  - Level 2: D, E
  - Level 3: F, G
- **Test Assertion**: B and C should come before D, E, F, G
- **Expected Result**: `[A, B, C, D, E, F, G]` (order may vary within levels)

#### Depth-First Search (DFS)
- **Starting Node**: A
- **Expected Behavior**: Deep traversal before backtracking
- **Possible Orders**: 
  - `[A, B, D, F, E, G, C]` or
  - `[A, C, E, G, D, F, B]`
- **Test Assertion**: Should start with A and visit all reachable nodes

#### Topological Sorting
- **Purpose**: Order nodes so prerequisites come before dependents
- **Expected Property**: For any edge (X → Y), X appears before Y in the sorted list
- **Valid Orders**: Multiple valid topological orders exist
- **Example**: `[A, B, C, D, E, F, G]` or `[A, C, B, E, D, G, F]`

### B. Connected Components Analysis

#### Test Graph Structure
```
Component 1: A → B → C    (3 nodes, 2 edges)
Component 2: D → E        (2 nodes, 1 edge)  
Component 3: F            (1 node, 0 edges)
```

- **Total Nodes**: 6
- **Total Edges**: 3
- **Expected Components**: 3 separate components
- **Component Sizes**: [1, 2, 3] (sorted)

### C. Path Finding Tests

#### Shortest Learning Path
- **Test Cases**:
  1. Linear path: A → B → C → D
  2. Diamond structure: A → {B,C} → D
  3. Multiple paths with different difficulties

#### Path Optimization
- **By Difficulty**: Prefer easier concepts (lower difficulty numbers)
- **By Time**: Prefer concepts with shorter time_to_master
- **Expected**: Algorithm should find valid paths between start and end nodes

### D. Cycle Detection Tests

#### Acyclic Graph Test
- **Structure**: A → B → C → D (linear chain)
- **Expected**: No cycles detected
- **Test Result**: `has_cycles()` should return `False`

#### Cycle Prevention Test
- **Structure**: A → B → C, then attempt C → A
- **Expected**: Either prevent cycle creation or detect existing cycle
- **Self-Prerequisite**: Concept depending on itself should be prevented

### E. Graph Metrics Tests

#### Graph Density
- **Formula**: `actual_edges / max_possible_edges`
- **Max Possible**: n × (n-1) for directed graph
- **Expected Range**: 0.0 to 1.0
- **Main Graph**: 9 edges / (10 × 9) = 0.1 density

#### Centrality Analysis
- **Degree Centrality**: Count of incoming + outgoing connections
- **Expected Results**:
  - Node A: High centrality (connects to B, C)
  - Node E: High centrality (receives from B, C and connects to G)
  - Node J: Zero centrality (isolated)

## 3. Test Validation Logic

### Core Functionality Tests
- **Graph Creation**: Verify graph initializes correctly
- **Concept Addition**: Ensure concepts are stored and retrievable
- **Relationship Management**: Test prerequisite relationships work
- **Error Handling**: Validate proper error responses

### Algorithm Tests
- **Traversal Correctness**: Verify algorithms visit nodes in expected order
- **Path Validity**: Ensure found paths respect prerequisite relationships
- **Component Detection**: Correctly identify disconnected graph parts
- **Cycle Detection**: Properly identify circular dependencies

### Performance Tests
- **Large Scale**: Test with 100+ concepts
- **Relationship Density**: Test graphs with many interconnections
- **Query Speed**: Ensure operations complete within reasonable time

## 4. Expected Test Outcomes

### Success Criteria
- ✅ All 63 tests pass
- ✅ Graph algorithms produce valid results
- ✅ Error conditions are handled gracefully
- ✅ Performance remains acceptable for realistic graph sizes

### Key Assertions
1. **Graph Structure**: Nodes and edges are correctly stored and accessible
2. **Algorithm Correctness**: Traversals follow mathematical definitions
3. **Path Validity**: All returned paths respect prerequisite ordering
4. **Component Analysis**: Correctly identifies connected/disconnected parts
5. **Error Robustness**: Handles invalid inputs without crashing

The test suite comprehensively validates that the knowledge graph can effectively model educational concept dependencies and support learning path generation for the math tutoring system. 
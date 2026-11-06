# HW3: Advanced Python

Author: Jiajing Wang

Use Python 3.12 or later. From this folder, run:

```bash
python test_hw3.py
```

Part 1: Memoization with expiry

Cache each result for three seconds from the time the function returns. Cache hits do not reset the timer.

Part 2: Binary tree traversal

Test pre-order/ in-order/ post-order by both recursive method and iterative method.

Return the new lists of node values.

The iterative traversals use a stack for pending nodes. Pre-order pushes the right child first. In-order saves nodes while moving left. Post-order marks a node for recording after both children.

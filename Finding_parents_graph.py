# Python implementation for
# the above approach
import COPDGen_normal_graph

sz = 10 ** 5

# Adjacency list representation
# of the tree
tree = [[] for _ in range(sz + 1)]

# Boolean array to mark all the
# vertices which are visited
vis = [0] * (sz + 1)

# Array of vector where ith index
# stores the path from the root
# node to the ith node
ans = [0] * (sz + 1)


# Function to create an
# edge between two vertices
def addEdge(a, b):
    # Add a to b's list
    tree[a].append(b)

    # Add b to a's list
    tree[b].append(a)


# Modified Breadth-First Function
def bfs(node):
    # Create a queue of child, parent
    qu = []

    # Push root node in the front of
    qu.append([node, 0])

    while (len(qu)):
        p = qu[0]

        # Dequeue a vertex from queue
        qu.pop(0)
        ans[p[0]] = p[1]
        vis[p[0]] = True

        # Get all adjacent vertices of the dequeued
        # vertex s. If any adjacent has not
        # been visited then enqueue it
        for child in tree[p[0]]:
            if (not vis[child]):
                qu.append([child, p[0]])


# Driver code

# Number of vertices
n = 6

addEdge(0, 1)
addEdge(0, 2)
addEdge(1, 3)
addEdge(2, 4)
addEdge(2, 5)

Combined_graph = COPDGen_normal_graph.avgBranchRadius(RmergedGraph, new_dic)
# Calling modified bfs function
bfs(0)

q = [2, 3]

for i in range(2):
    print(ans[q[i]])

# This code is contributed by SHUBHAMSINGH10
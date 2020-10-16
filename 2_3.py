import numpy as np
from Tree import Tree
from Tree import Node
import ipdb

def calculate_likelihood(tree_topology, theta, beta):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: numpy array. Dimensions: (num_nodes, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.
    """
    print("Calculating the likelihood...")
    
    num_nodes = np.shape(theta)[0]
    K = np.shape(theta)[1] #number of possible states
    theta = np.asarray(theta)
    S = np.zeros(np.shape(theta))

    # Set S for all the leaves
    for node in range(num_nodes):
        try:
            i = int(beta[node])
            S[node, i] = 1
        except:
            pass
            # At inner node, not leaf
    
    children = [0]*num_nodes
    
    # tree_topology[i] = parent of i
    # we want children[i] = children of i as well

    for i, n in enumerate(tree_topology):
        try:
            n = int(n)
            if children[n] == 0:
                children[n] = i
            else:
                children[n] = (children[n], i)
        except:
            pass
            # At root, root's no one's child
    
    for node in range(num_nodes-1, -1, -1): # Go from leaves to root 
        for i in range(K):
            if children[node] != 0:
                uv_sum = 0
                uw_sum = 0
                v = children[node][0]
                w = children[node][1]
                for j in range(K):
                    uv_sum += theta[v, i][j] * S[v, j]   
                    uw_sum += theta[w, i][j] * S[w, j]
                S[node, i] = uv_sum * uw_sum
            else:
                pass
                # At leaf, already know S, it's one-hot!

    p = 0
    # The probability for the entire tree
    for i in range(K):
        p += S[0, i] * theta[0,i]

    likelihood = p
    return likelihood


def main():

    filename ="data/q2_3_large_tree.pkl"#"#  "data/q2_3_large_tree.pkl" "data/q2_3_small_tree.pkl"# 
    t = Tree()
    t.load_tree(filename)

    leaves = 0
    for b in t.filtered_samples[0]:
        try:
            b = int(b)
            leaves += 1
        except:
            pass
    print("Number of leaves:", leaves)

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx)
        #print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        top = t.get_topology_array()
        theta = t.get_theta_array()
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()

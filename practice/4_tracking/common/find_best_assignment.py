import logging as log
from scipy.optimize import linear_sum_assignment

def convert_affinity_matrix_to_cost_matrix(affinity_matrix):
    cost_matrix = []
    for affinity_row in affinity_matrix:
        cost_row = []
        for aff in affinity_row:
            cost_row.append(-aff)
        cost_matrix.append(cost_row)
    return cost_matrix

def solve_assignment_problem(affinity_matrix, affinity_threshold):
    """
    This method receives an affinity matrix and returns the decision as a map
    {row_index => column_index}
    Also this method returns best_affinity -- affinity of the assignment for each row.
    Note that best_affinity is used mostly for logging / algorithm debugging
    """
    if len(affinity_matrix) == 0:
        log.debug("No active tracks at the moment -- return empty decision")
        return {}, {}

    cost_matrix = convert_affinity_matrix_to_cost_matrix(affinity_matrix)

    decision = {}
    best_affinity = {}
    num_rows = len(affinity_matrix)
    for i in range(num_rows):
        decision[i] = None
        best_affinity[i] = None

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    for i, j in zip(row_ind, col_ind):
        decision[i] = j
        best_affinity[i] = affinity_matrix[i][j]

        if best_affinity[i] < affinity_threshold:
            # this match is too bad -- remove it
            log.debug("remove match for row_index={}, since best_affinity={:.3f} < {}".format(
                      i, best_affinity[i], affinity_threshold))
            decision[i] = None

    return decision, best_affinity

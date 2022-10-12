# CIV102 assignment 4, problem 3 and 4
# Write a script to solve the loads on all members of a truss structure

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import lsqr


def solve_truss(joints, members, loads):
    """Solves for forces on each member and prints the answer"""
    m = len(joints)*2  # number of equations
    n = len(members)  # number of unknowns
    assert n < m
    joints_i = {}
    for key in joints.keys():
        joints[key] = np.array(joints[key], dtype=np.float64)
        joints_i[key] = len(joints_i)

    # matrix: transforms member loads to joint loads
    A = []
    for i in range(len(members)):
        j1, j2 = members[i]
        dp = joints[j2]-joints[j1]
        dp /= np.linalg.norm(dp)
        i1, i2 = joints_i[j1], joints_i[j2]
        A += [
            (dp[0], 2*i1, i),
            (dp[1], 2*i1+1, i),
            (-dp[0], 2*i2, i),
            (-dp[1], 2*i2+1, i)
        ]
    A = scipy.sparse.coo_matrix(
        ([a[0] for a in A], ([a[1] for a in A], [a[2] for a in A])),
        shape=(m, n))

    # vector: at each joint, member force + load = 0
    b = np.zeros(m)
    for (j, load) in loads.items():
        i = joints_i[j]
        b[2*i] -= load[0]
        b[2*i+1] -= load[1]

    # solve the linear system
    x, istop, itn, normr = lsqr(A, b)[:4]
    print("Numerical error:", normr)
    for (member, load) in zip(members, x):
        print(''.join(member), "{:.4g}".format(load))


def problem_3():
    """Warren truss"""
    l = 5  # 5m each
    dx = l * 0.5
    dy = l * np.sin(np.pi/3)
    joints = {
        'A': (0, 0),
        'B': (dx, dy),
        'C': (2*dx, 0),
        'D': (3*dx, dy),
        'E': (4*dx, 0),
        'F': (5*dx, dy),
        'G': (6*dx, 0),
        "F'": (7*dx, dy),
        "E'": (8*dx, 0),
        "D'": (9*dx, dy),
        "C'": (10*dx, 0),
        "B'": (11*dx, dy),
        "A'": (12*dx, 0)
    }
    members = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
        ('B', 'D'),
        ('C', 'D'),
        ('C', 'E'),
        ('D', 'E'),
        ('D', 'F'),
        ('E', 'F'),
        ('E', 'G'),
        ('F', 'G'),
        ('F', "F'"),
        ('G', "F'"),
        ('G', "E'"),
        ("F'", "E'"),
        ("F'", "D'"),
        ("E'", "D'"),
        ("E'", "C'"),
        ("D'", "C'"),
        ("D'", "B'"),
        ("C'", "B'"),
        ("C'", "A'"),
        ("B'", "A'")
    ]
    loads = {
        'A': (0, 105-17.5),
        'C': (0, -35),
        'E': (0, -35),
        'G': (0, -35),
        "E'": (0, -35),
        "C'": (0, -35),
        "A'": (0, 105-17.5)
    }
    solve_truss(joints, members, loads)


def problem_4():
    """Pratt truss"""
    l = 5  # 5m each
    joints = {
        'A': (0, 0),
        'B': (l, 0),
        'C': (l, -l),
        'D': (2*l, 0),
        'E': (2*l, -l),
        'F': (3*l, 0),
        'G': (3*l, -l),
        "D'": (4*l, 0),
        "E'": (4*l, -l),
        "B'": (5*l, 0),
        "C'": (5*l, -l),
        "A'": (6*l, 0)
    }
    members = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
        ('B', 'D'),
        ('B', 'E'),
        ('C', 'E'),
        ('D', 'E'),
        ('D', 'F'),
        ('D', 'G'),
        ('E', 'G'),
        ('F', 'G'),
        ("E'", 'G'),
        ("D'", 'F'),
        ("D'", 'G'),
        ("D'", "E'"),
        ("C'", "E'"),
        ("B'", "E'"),
        ("B'", "D'"),
        ("B'", "C'"),
        ("A'", "C'"),
        ("A'", "B'")
    ]
    loads = {
        'A': (0, 105-17.5),
        'C': (0, -35),
        'E': (0, -35),
        'G': (0, -35),
        "E'": (0, -35),
        "C'": (0, -35),
        "A'": (0, 105-17.5)
    }
    solve_truss(joints, members, loads)
    

if __name__ == "__main__":
    #problem_3()
    problem_4()

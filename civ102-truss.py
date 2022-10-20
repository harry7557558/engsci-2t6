# CIV102 truss problems on assignments
# Write a script to solve the forces on all members of a truss structure
# I'm lazy doing assignments, but I'm not lazy writing code :D

import numpy as np
import scipy.sparse
from scipy.sparse.linalg import eigs, lsqr


def format_float(x):
    """Slide-rule precision"""
    sigfig = 4 if np.log10(abs(x))%1.0 < np.log10(2) else 3
    return f"{{:.{sigfig}g}}".format(x)


def solve_truss(joints, members, loads):
    """Solves for forces on each member and prints the answer"""
    m = len(joints)*2  # number of equations
    n = len(members)  # number of unknowns
    print(m, "equations,", n, "unknowns")
    assert n <= m
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
    print("Minimum eigenvalue:", sorted(abs(eigs(A.T@A)[0]))[0])

    # vector: at each joint, member force + load = 0
    net_load = np.zeros(2)
    net_moment = 0.0
    b = np.zeros(m)
    for (j, load) in loads.items():
        i = joints_i[j]
        b[2*i] -= load[0]
        b[2*i+1] -= load[1]
        pos = joints[j]
        net_load += load
        net_moment += pos[0]*load[1]-pos[1]*load[0]
    print("Net force/moment:", net_load, net_moment)

    # solve the linear system
    #A, b = A.T@A, A.T*b
    x, istop, itn, normr = lsqr(A, b)[:4]
    print("Numerical error:", normr)
    for (member, load) in zip(members, x):
        name = ''.join(member)
        if abs(load) < min(1e-6, 10.0*normr):
            print(name, 0)
            continue
        print(name, format_float(load))


def assignment_4_problem_3():
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


def assignment_4_problem_4():
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


def assignment_5_problem_1():
    """Four equilateral triangles"""
    dx = 8000
    dy = dx * np.sin(np.pi/3)
    Fy = 900 * np.sin(np.pi/3)
    joints = {
        'A': (dx, 2*dy),
        'B': (0.5*dx, dy),
        'C': (1.5*dx, dy),
        'D': (0, 0),
        'E': (dx, 0),
        'F': (2*dx, 0)
    }
    members = [
        ('A', 'B'),
        ('A', 'C'),
        ('B', 'C'),
        ('B', 'D'),
        ('B', 'E'),
        ('C', 'E'),
        ('C', 'F'),
        ('D', 'E'),
        ('D', 'F')
    ]
    loads = {
        'A': (900, 0),
        'D': (-450, -Fy),
        'F': (-450, Fy)
    }
    solve_truss(joints, members, loads)


def assignment_5_problem_2():
    """A long Warren truss"""
    # joints
    dx = 3000
    dy = 4000
    joints = {}
    for i in range(1, 11+1):
        joints[f"T{i}"] = (i*dx, dy)
    for i in range(1, 13+1):
        joints[f"B{i}"] = ((i-1)*dx, 0)
    # members
    members = []
    for i in range(1, 6+1):
        members += [
            (f"B{i}", f"T{i}"),
            (f"B{i}", f"B{i+1}")
        ]
        if i != 6:
            members += [
                (f"T{i}", f"T{i+1}"),
                (f"T{i}", f"B{i+1}")
            ]
    for i in range(7, 12+1):
        if i != 12:
            members += [
                (f"T{i-1}", f"T{i}")
            ]
        members += [
            (f"T{i-1}", f"B{i}"),
            (f"T{i-1}", f"B{i+1}"),
            (f"B{i}", f"B{i+1}")
        ]
    # loads
    loads = {
        "B1": (0, 5.5),
        "B13": (0, 5.5)
    }
    for i in range(2, 12+1):
        loads[f"B{i}"] = (0, -1)
    solve_truss(joints, members, loads)


def assignment_5_problem_4b(n, fe, fm):
    """Wind forces on top/bottom braces
        n: number of "chambers"
        fe: load for the two"""
    # joints
    dx = 5000
    dy = 3000
    joints = {}
    for i in range(1, n+2):
        joints[f"A{i}"] = ((i-1)*dx, dy)
        joints[f"B{i}"] = ((i-1)*dx, 0)
    # memebrs
    members = []
    for i in range(1, n+2):
        members += [
            (f"A{i}", f"B{i}")
        ]
        if i > n:
            continue
        members += [
            (f"A{i}", f"A{i+1}"),
            (f"B{i}", f"B{i+1}")
        ]
        if i%2 == 1:
            members += [
                (f"B{i}", f"A{i+1}")
            ]
        else:
            members += [
                (f"A{i}", f"B{i+1}")
            ]
    # loads
    fr = 0.25*(2*fe+(n-1)*fm)
    loads = {
        f"B{1}": (0, fe-fr),
        f"B{n+1}": (0, fe-fr),
        f"A{1}": (0, -fr),
        f"A{n+1}": (0, -fr)
    }
    print("A1.y =", format_float(loads["A1"][1]))
    print("B1.y =", format_float(loads["B1"][1]))
    for i in range(2, n+1):
        loads[f"B{i}"] = (0, fm)
    solve_truss(joints, members, loads)


def assignment_5_problem_6():
    """A weird truss bridge"""
    dx = 3000
    dy = 2500
    joints = {
        'A': (dx, 2*dy),
        'B': (2*dx, 2*dy),
        'C': (3*dx, 2*dy),
        'D': (4*dx, 2*dy),
        'E': (5*dx, 2*dy),
        'F': (dx, dy),
        'G': (2*dx, dy),
        'H': (4*dx, dy),
        'I': (5*dx, dy),
        'J': (0, 0),
        'K': (dx, 0),
        'L': (2*dx, 0),
        'M': (3*dx, 0),
        'N': (4*dx, 0),
        'O': (5*dx, 0),
        'P': (6*dx, 0)
    }
    members = [tuple(ab) for ab in [
        'AB', 'BC', 'CD', 'DE',
        'AJ', 'AF', 'FK', 'BF', 'FL',
        'BG', 'GL', 'CG', 'GM', 'CM',
        'CH', 'HM', 'DH', 'HN',
        'DI', 'IN', 'EI', 'IO', 'EP',
        'JK', 'KL', 'LM', 'MN', 'NO', 'OP'
    ]]
    loads = {
        'J': (0, 25),
        'K': (0, -10),
        'L': (0, -10),
        'M': (0, -10),
        'N': (0, -10),
        'O': (0, -10),
        'P': (0, 25)
    }
    solve_truss(joints, members, loads)


def assignment_5_problem_7():
    """2/3 of a regular hexagon hanging on the wall"""
    r = 3000
    s = np.sin(np.pi/3)
    joints = {
        'O': (0, 0),
        'A': (0.5*r, s*r),
        'B': (r, 0),
        'C': (0.5*r, -s*r),
        'D': (-0.5*r, -s*r),
        'E': (-r, 0)
    }
    members = [tuple(ab) for ab in [
        'OA', 'OB', 'OC', 'OD', 'OE',
        'AB', 'BC', 'CD', 'DE'
    ]]
    loads = {
        'A': (8, 0),
        'B': (-6, 4*s),
        'E': (-2, -4*s)
    }
    solve_truss(joints, members, loads)


def quiz_6():
    """WARNING:
        If you did all your truss homework using this script (like me),
        you won't end well on CIV102 quiz 6.
    """
    joints = {
        'A': (-2, 2),
        'B': (-1, 2),
        'C': (-1, 3),
        'D': (0, 2),
        'E': (0, 3),
        'F': (1, 2),
        'G': (0, 1),
        'H': (1, 1),
        'I': (0, 0),
        'J': (1, 0)
    }
    members = [tuple(ab) for ab in [
        'AB', 'AC', 'BC', 'CE', 'BE', 'BD', 'DE', 'EF',
        'DF', 'DG', 'FG', 'FH', 'GH', 'GI', 'HI', 'HJ', 'IJ'
    ]]
    loads = {
        'A': (0, -10),
        'B': (0, -20),
        'C': (15, 0),
        'F': (-2.5, 0),
        'I': (-12.5, 30)
    }
    solve_truss(joints, members, loads)

if __name__ == "__main__":
    #assignment_4_problem_3()
    #assignment_4_problem_4()
    #assignment_5_problem_1()
    #assignment_5_problem_2()
    #assignment_5_problem_4b(5, 3.01, 3.77)
    #assignment_5_problem_4b(6, 5.80, 11.61)
    #assignment_5_problem_6()
    #assignment_5_problem_7()
    quiz_6()

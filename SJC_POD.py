def get_POD(Data_Mat):
    from numpy import load, diag
    from numpy.linalg import svd
    U, s, V = svd(Data_Mat, full_matrices=0, compute_uv=1)
    S = diag(s)
    return U, S, V

def get_PODMode(U, S, V, ModeI):
    from numpy import dot, asmatrix
    PODMode = dot( \
        asmatrix(U[:, ModeI] * S[ModeI, ModeI]).T, \
        asmatrix(V[ModeI, :]) \
        )
    return PODMode

def get_PODTimeCoe(V, ModeI):
    PODTimeCoe = V[ModeI, :]
    return PODTimeCoe

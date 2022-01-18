import heterocl as hcl
import os

def top_2mm(P=16, Q=22, R=18, S=24, alpha=0.1, beta=0.1, dtype=hcl.Int(), target=None):

    hcl.init(dtype)
    A = hcl.placeholder((P, Q), "A")
    B = hcl.placeholder((Q, R), "B")
    C = hcl.placeholder((R, S), "C")
    D = hcl.placeholder((P, S), "D")

    def kernel_2mm(A, B, C, D):
        
        r = hcl.reduce_axis(0, Q, "r")
        out_AB = hcl.compute((P, R), 
                         lambda x, y: hcl.sum(A[x, r] * B[r, y], 
                         axis=r, 
                         dtype=dtype
                         ), 
                         name="out_AB"
                         )

        k = hcl.reduce_axis(0, R, "k")
        out_ABC = hcl.compute((P, S), 
                         lambda x, y: hcl.sum(out_AB[x, k] * C[k, y], 
                         axis=k, 
                         dtype=dtype
                         ), 
                         name="out_ABC"
                         )
        hcl.update(D,
                   lambda x, y: (alpha * out_ABC[x, y] + beta * D[x, y]),
                   name="D"
                   )
        
    s = hcl.create_schedule([A, B, C, D], kernel_2mm)

    #### Applying customizations ####

    AB = kernel_2mm.out_AB
    ABC = kernel_2mm.out_ABC
    D = kernel_2mm.D
    
    ## N Buggy 1
    s[AB].compute_at(s[ABC], ABC.axis[0])
    s[ABC].compute_at(s[D], D.axis[0])

    ## N Buggy 2
    #s[AB].compute_at(s[ABC], ABC.axis[0])
    #s[ABC].compute_at(s[D], D.axis[1])

    ## N Buggy 3
    #s[D].reorder(D.axis[1], D.axis[0])
    #s[AB].compute_at(s[ABC], ABC.axis[0])

    ## Buggy 1
    #s[AB].compute_at(s[ABC], ABC.axis[1])
    #s[ABC].compute_at(s[D], D.axis[0])

    ## Buggy 2
    #s[AB].compute_at(s[ABC], ABC.axis[1])
    #s[ABC].compute_at(s[D], D.axis[1])

    ## Buggy 3
    #s[D].reorder(D.axis[1], D.axis[0])
    #s[AB].compute_at(s[ABC], ABC.axis[1])

    #s[D].parallel(D.axis[1])

    #### Applying customizations ####

    return hcl.build(s, target=target)
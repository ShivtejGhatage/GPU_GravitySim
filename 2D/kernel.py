import numpy as np
from galaxy import Galaxy
import config

deltaT = config.deltaT
G = config.G
eps = config.eps
e = config.e
L = config.L
BITS = config.BITS
GRID = config.GRID


def update(gx):
    N = gx.N
    acc = np.zeros((N, 2), dtype=np.float32)

    # ------------------------------
    # 1. Morton ordering
    # ------------------------------
    ix = ((gx.pos[:,0] + L) / (2*L) * (GRID-1)).astype(np.uint32)
    iy = ((gx.pos[:,1] + L) / (2*L) * (GRID-1)).astype(np.uint32)

    ix = np.clip(ix, 0, GRID-1)
    iy = np.clip(iy, 0, GRID-1)

    codes = morton2D(ix, iy)
    order = np.argsort(codes)
    gx.morton = order

    pos = gx.pos[order]
    mass = gx.masses[order]

    # ------------------------------
    # 2. Build implicit nodes (ranges)
    # ------------------------------
    nodes = [(0, N)]
    for level in range(BITS):
        new_nodes = []
        mask = 1 << (2*(BITS-1-level))

        for s, e in nodes:
            if e - s <= 1:
                new_nodes.append((s, e))
                continue

            split = s
            for i in range(s+1, e):
                if (codes[order[i]] & mask) != (codes[order[i-1]] & mask):
                    new_nodes.append((split, i))
                    split = i
            new_nodes.append((split, e))

        nodes = new_nodes

    # ------------------------------
    # 3. Compute node mass + COM
    # ------------------------------
    node_mass = []
    node_com  = []

    for s, e in nodes:
        ids = slice(s, e)
        m = mass[ids].sum()
        if m > 0:
            com = (pos[ids] * mass[ids,None]).sum(axis=0) / m
        else:
            com = np.zeros(2)
        node_mass.append(m)
        node_com.append(com)

    # ------------------------------
    # 4. Barnes–Hut force evaluation
    # ------------------------------
    theta = 0.5
    box_size = 2 * L

    for ii in range(N):
        i = order[ii]
        pi = gx.pos[i]
        ai = np.zeros(2)

        for (s,e), m, com in zip(nodes, node_mass, node_com):
            if m == 0:
                continue

            size = box_size * (e - s) / N
            d = com - pi
            dist = np.linalg.norm(d) + eps

            if (size / dist) < theta or (e - s) == 1:
                ai += G * m * d / (dist**3)

        acc[i] = ai

    # ------------------------------
    # 5. Leapfrog integration
    # ------------------------------
    gx.vel += 0.5 * acc * deltaT
    gx.pos += gx.vel * deltaT
    gx.vel += 0.5 * acc * deltaT

    # ------------------------------
    # 6. Periodic boundaries
    # ------------------------------
    gx.pos[:,0] = (gx.pos[:,0] + L) % (2*L) - L
    gx.pos[:,1] = (gx.pos[:,1] + L) % (2*L) - L


    # print(np.max(np.linalg.norm(gx.pos, axis=1)))


    # for i in range(N):
    #     for j in range(i+1,N):

    #         # if (gx.active[i] and gx.active[j]):
    #             dr = gx.pos[j]-gx.pos[i]
    #             dist = np.linalg.norm(dr)
    #             if ( dist  <  (np.sqrt(gx.masses[i])+np.sqrt(gx.masses[j]))/8  ):
    #                 collision(i,j,gx)




def morton2D(x, y):
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555

    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555

    return x | (y << 1)




# def collision(i ,j,gx):
#     if (gx.masses[i] > gx.masses[j]): collision(j,i,gx)
#     m1 = min(gx.masses[i],gx.masses[j])
#     m2 = max(gx.masses[i],gx.masses[j])
#     if(m1*3 < m2):
#         merge(j,i,gx)
#     else:
#         inelastic_collision(i,j,gx)
    

# def inelastic_collision(i ,j,gx):
#     m1 = gx.masses[i]
#     m2 = gx.masses[j]
#     v1 = gx.vel[i]
#     v2 = gx.vel[j]
#     gx.vel[i] = (m1-e*m2)/(m1+m2)*v1 + ((1+e)*m2)/(m1+m2)*v2
#     gx.vel[j] = ((1+e)*m1)/(m1+m2)*v1 +  (m2-e*m1)/(m1+m2)*v2

# def merge(i,j,gx):
#     m1 = gx.masses[i]
#     m2 = gx.masses[j]
#     v1 = gx.vel[i]
#     v2 = gx.vel[j]

#     gx.masses[i] = m1+m2
#     gx.pos[i] = (m1 * gx.pos[i] + m2 * gx.pos[j])/(m1+m2)
#     gx.vel[i] = (m1*v1+m2*v2)/(m1+m2)
    
#     gx.masses[j] = np.random.uniform(5,10)
#     addOnMerge(j,gx)
    
#     # gx.active[j] = 0



# def addOnMerge(j,gx):
#     gx.masses[j] = np.random.uniform(5,10)
#     edge = np.random.randint(4)
#     speedx = np.random.uniform(0,2)
#     speedy = np.random.uniform(0,2)

#     if (edge == 0):
#         y = np.random.uniform(-100,100)
#         gx.pos[j] = (-100,y)
#         gx.vel[j] = (speedx, (speedy if (y<0) else -speedy))
#     elif (edge == 1):
#         x = np.random.uniform(-100,100)
#         gx.pos[j] = (x,100)
#         gx.vel[j] = ((speedx if (x<0) else -speedx), -speedy)
#     elif (edge == 2):
#         y = np.random.uniform(-100,100)
#         gx.pos[j] = (100,y)
#         gx.vel[j] = (-speedx, (speedy if (y<0) else -speedy))
#     else:
#         x = np.random.uniform(-100,100)
#         gx.pos[j] = (x,-100)
#         gx.vel[j] = ((speedx if (x<0) else -speedx), speedy)



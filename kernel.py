import numpy as np
from galaxy import Galaxy

deltaT = 0.1
G = 0.1
eps = 2
e = 0.7


def collision(i ,j,gx):
    m1 = min(gx.masses[i],gx.masses[j])
    m2 = max(gx.masses[i],gx.masses[j])
    if(m1*3 < m2):
        merge(i,j,gx)
    else:
        inelastic_collision(i,j,gx)
    

def inelastic_collision(i ,j,gx):
    m1 = gx.masses[i]
    m2 = gx.masses[j]
    v1 = gx.vel[i]
    v2 = gx.vel[j]
    gx.vel[i] = (m1-e*m2)/(m1+m2)*v1 + ((1+e)*m2)/(m1+m2)*v2
    gx.vel[j] = ((1+e)*m1)/(m1+m2)*v1 +  (m2-e*m1)/(m1+m2)*v2

def merge(i,j,gx):
    m1 = gx.masses[i]
    m2 = gx.masses[j]
    v1 = gx.vel[i]
    v2 = gx.vel[j]

    gx.masses[i] = m1+m2
    gx.pos[i] = (m1 * gx.pos[i] + m2 * gx.pos[j])/(m1+m2)
    gx.vel[i] = (m1*v1+m2*v2)/(m1+m2)
    
    gx.masses[j] = 0
    gx.vel[j] = 0
    gx.active[j] = 0


def update(gx):
    N = gx.N
    acc = np.zeros((N, 2), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):

            if (gx.active[i] and gx.active[j]):
                dr = gx.pos[j]-gx.pos[i]
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + eps*eps
                invR3 = 1.0/(np.sqrt(r2)*r2)

                acc[i] += G * gx.masses[j] * dr * invR3
                acc[j] -= G * gx.masses[i] * dr * invR3

    gx.vel += acc*deltaT
    gx.pos += gx.vel*deltaT


    for i in range(N):
        for j in range(i+1,N):

            if (gx.active[i] and gx.active[j]):
                dr = gx.pos[j]-gx.pos[i]
                dist = np.linalg.norm(dr)
                if ( dist  <  (np.sqrt(gx.masses[i])+np.sqrt(gx.masses[j]))/8  ):
                    collision(i,j,gx)

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
    
    gx.masses[j] = np.random.uniform(5,10)
    addOnMerge(j,gx)
    
    # gx.active[j] = 0



def addOnMerge(j,gx):
    gx.masses[j] = np.random.uniform(5,10)
    edge = np.random.randint(4)
    speedx = np.random.uniform(0,2)
    speedy = np.random.uniform(0,2)

    if (edge == 0):
        y = np.random.uniform(-100,100)
        gx.pos[j] = (-100,y)
        gx.vel[j] = (speedx, (speedy if (y<0) else -speedy))
    elif (edge == 1):
        x = np.random.uniform(-100,100)
        gx.pos[j] = (x,100)
        gx.vel[j] = ((speedx if (x<0) else -speedx), -speedy)
    elif (edge == 2):
        y = np.random.uniform(-100,100)
        gx.pos[j] = (100,y)
        gx.vel[j] = (-speedx, (speedy if (y<0) else -speedy))
    else:
        x = np.random.uniform(-100,100)
        gx.pos[j] = (x,-100)
        gx.vel[j] = ((speedx if (x<0) else -speedx), speedy)




def update(gx):
    N = gx.N
    acc = np.zeros((N, 2), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):

            # if (gx.active[i] and gx.active[j]):
                dr = gx.pos[j]-gx.pos[i]
                r2 = dr[0]*dr[0] + dr[1]*dr[1] + eps*eps
                invR3 = 1.0/(np.sqrt(r2)*r2)

                acc[i] += G * gx.masses[j] * dr * invR3
                acc[j] -= G * gx.masses[i] * dr * invR3

    gx.vel += acc*deltaT
    gx.pos += gx.vel*deltaT


    for i in range(N):
        for j in range(i+1,N):

            # if (gx.active[i] and gx.active[j]):
                dr = gx.pos[j]-gx.pos[i]
                dist = np.linalg.norm(dr)
                if ( dist  <  (np.sqrt(gx.masses[i])+np.sqrt(gx.masses[j]))/8  ):
                    collision(i,j,gx)

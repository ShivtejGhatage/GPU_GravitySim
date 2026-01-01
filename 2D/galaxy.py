import numpy as np
import config

deltaT = config.deltaT
G = config.G
eps = config.eps
e = config.e
M = config.M


class Galaxy:
    def __init__(self, N = 10):
        self.N = N
        self.vel = np.zeros((N,2), dtype=np.float32)
        self.pos = np.zeros((N,2), dtype=np.float32)
        self.masses = np.zeros(N, dtype=np.float32)
        # self.active = np.ones(N, dtype=np.float32)
        self.new = N-1
        self.morton = np.arange(N, dtype=np.int32)


    def __str__(self):
        N = self.N
        pos_min = self.pos.min(axis=0) 
        pos_max = self.pos.max(axis=0)
        vel_mag = np.linalg.norm(self.vel, axis=1)

        return (
            f"Galaxy(\n"
            f"  particles : {N}\n"
            f"  mass      : min={self.masses.min():.2f}, "
            f"max={self.masses.max():.2f}, "
            f"mean={self.masses.mean():.2f}\n"
            f"  position  : x∈[{pos_min[0]:.2f}, {pos_max[0]:.2f}], "
            f"y∈[{pos_min[1]:.2f}, {pos_max[1]:.2f}] \n"
            f"  speed     : min={vel_mag.min():.2f}, "
            f"max={vel_mag.max():.2f}, "
            f"mean={vel_mag.mean():.2f}\n"
            f")"
        )


    def add(self, mass, pos, vel):
        if(self.new<0): 
            self.new = self.N
        self.masses[self.new] = mass
        self.pos[self.new] = pos
        self.vel[self.new] = vel
        self.new -= 1


    def rando(self):

        theta = np.random.uniform(0, 2*np.pi, self.N)
        # r = np.sqrt(np.random.uniform(10**2, 50**2, self.N))
        r = np.random.uniform(10, 100, self.N)

        self.pos[:,0] = r * np.cos(theta)
        self.pos[:,1] = r * np.sin(theta)

        self.masses[:] = np.abs(np.random.normal(2, 0.1, self.N))

        for i in range(self.N):
            rvec = self.pos[i]
            dist = np.linalg.norm(rvec) + eps

            # circular orbit speed
            vmag = np.sqrt(G * M / dist)

            # 2D perpendicular (tangential direction)
            t = np.array([-rvec[1], rvec[0]]) / dist

            self.vel[i] = vmag * t



    def big_bang(self):
        # random positions in small sphere
        self.pos[:] = np.random.normal(0, 1.0, (self.N, 2))
        r = np.linalg.norm(self.pos, axis=1)
        self.pos /= r[:,None]
        self.pos *= np.random.uniform(1, 5, (self.N,1))

        # nearly uniform masses
        self.masses[:] = np.abs(np.random.normal(1.0, 0.05, self.N))

        # outward Hubble velocity
        H0 = 1.0
        self.vel[:] = H0 * self.pos





    
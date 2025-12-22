import numpy as np

class Galaxy:
    def __init__(self, N = 10):
        self.N = N
        self.vel = np.zeros((N,2), dtype=np.float32)
        self.pos = np.zeros((N,2), dtype=np.float32)
        self.masses = np.zeros(N, dtype=np.float32)
        # self.active = np.ones(N, dtype=np.float32)
        self.new = N-1

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
            f"y∈[{pos_min[1]:.2f}, {pos_max[1]:.2f}]\n"
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
        center = np.array([0.0, 0.0])
        M = 500.0          # central mass
        G = 0.1
        eps = 2.0


        theta = np.random.uniform(-1, 1, self.N) * np.pi
        r = np.random.normal(50,20,size=(self.N))

        self.pos[:, 0] = r * np.cos(theta)
        self.pos[:, 1] = r * np.sin(theta)


        self.masses[:] = np.random.uniform(1, 10, self.N)

        for i in range(self.N):
            r = self.pos[i] - center
            dist = np.linalg.norm(r) + eps

            # circular orbit speed
            vmag = np.sqrt(G * M / dist)

            # perpendicular direction
            perp = np.array([-r[1], r[0]]) / dist

            self.vel[i] = vmag * perp



    
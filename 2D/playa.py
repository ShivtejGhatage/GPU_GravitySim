import numpy as np
import glfw
import moderngl
import os

# ---------------- LOAD DATA ----------------

HERE = os.path.dirname(__file__)
history = np.load(os.path.join(HERE, "history.npy"))
masses = np.load(os.path.join(HERE, "mass.npy"))
T, N, _ = history.shape

SPEED = 5   # larger = slower
frame = 0

# ---------------- SHADERS ----------------

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
in float in_mass;
in vec3 in_color;

uniform float scale;
out vec3 v_color;

void main() {
    gl_Position = vec4(in_pos / scale, 0.0, 1.0);
    gl_PointSize = 2.0 * sqrt(in_mass);
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_color;
out vec4 f_color;

void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c,c) > 0.25) discard;
    f_color = vec4(v_color, 1.0);
}
"""

# ---------------- MAIN ----------------

def main():
    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 800, "N-Body Replay", None, None)
    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    prog = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER,
    )

    data = np.zeros(
        N,
        dtype=[('pos','f4',2), ('mass','f4'), ('color','f4',3)]
    )

    data['pos']   = history[0]
    data['mass']  = masses
    data['color'] = np.random.uniform(0.4, 1.0, (N,3))

    vbo = ctx.buffer(data.tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo, "2f 1f 3f", "in_pos", "in_mass", "in_color")]
    )

    prog["scale"].value = 100

    running = True
    t = 0

    def key_cb(win, key, sc, action, mods):
        nonlocal running
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            running = not running

    glfw.set_key_callback(window, key_cb)

    # ---------------- LOOP ----------------

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if running:
            data['pos'] = history[t]
            vbo.write(data.tobytes())
            t = (t + 1) % T

        ctx.clear(0, 0, 0)
        vao.render(moderngl.POINTS)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()

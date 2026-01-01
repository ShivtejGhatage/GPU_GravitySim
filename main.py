import numpy as np
from galaxy import Galaxy
import glfw
import moderngl
from kernel import update

N = 100

# ================= SHADERS =================

VERTEX_SHADER = """
#version 330

in vec3 in_pos;
in float in_mass;
in vec3 in_color;

uniform mat4 MVP;

out vec3 v_color;

void main() {
    gl_Position = MVP * vec4(in_pos, 1.0);
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
    f_color = vec4(v_color, 0.6);
}
"""

# ================= CAMERA =================

def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / np.tan(fovy / 2)
    return np.array([
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (zfar+znear)/(znear-zfar), -1],
        [0, 0, (2*zfar*znear)/(znear-zfar), 0],
    ], dtype=np.float32)

# ================= MAIN =================

def main():
    gx = Galaxy(N)
    gx.rando()
    # gx.big_bang()
    gx.add(3, (0,0,0), (0,0,0))

    glfw.init()
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 800, "StarForge 3D", None, None)
    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    prog = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER,
    )

    data = np.zeros(
        N,
        dtype=[('pos','f4',3), ('mass','f4'), ('color','f4',3)]
    )
    data['pos'] = gx.pos
    data['mass'] = gx.masses
    data['color'] = (0.1,0.7,1)
    # data['color'] = (1,1,0.7)
    # data['color'][-1] = (1,1,1)

    vbo = ctx.buffer(data.tobytes())
    vao = ctx.vertex_array(
        prog,
        [(vbo, "3f 1f 3f", "in_pos", "in_mass", "in_color")]
    )

    fbw, fbh = glfw.get_framebuffer_size(window)
    proj = perspective(np.radians(60), fbw/fbh, 0.1, 2000)

    view = np.eye(4, dtype=np.float32)
    view[2,3] = -100  # camera back

    MVP = proj @ view
    prog["MVP"].write(MVP.tobytes())

    running = False

    def key_callback(window, key, scancode, action, mods):
        nonlocal running
        if key == glfw.KEY_SPACE and action == glfw.PRESS:
            running = not running

    glfw.set_key_callback(window, key_callback)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        if running:
            update(gx)

        data['pos'] = gx.pos
        data['mass'] = gx.masses
        vbo.write(data.tobytes())

        ctx.clear(0,0,0)
        vao.render(moderngl.POINTS)
        glfw.swap_buffers(window)


    glfw.terminate()

if __name__ == "__main__":
    main()

import numpy as np
from galaxy import Galaxy
import glfw
import moderngl
from kernel import update

N = 100

# ---------------- OpenGL ----------------

VERTEX_SHADER = """
#version 330

in vec2 in_pos;
in float in_mass;


uniform float scale;

void main() {
    gl_Position = vec4(in_pos / scale, 0.0, 1.0);

    // map mass -> size (tune these numbers)
    gl_PointSize = 2.0 * sqrt(in_mass);
}

"""

FRAGMENT_SHADER = """
#version 330
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord - vec2(0.5);
    if (dot(c, c) > 0.25) discard;
    f_color = vec4(1.0, 1.0, 1.0, 1.0);
}
"""
def main():
    # init simulation
    gx = Galaxy(N)
    gx.rando()

    gx.add(500, (0,0), (0,0))

    # centers = [(-50,-50),(50,-50),(-50,50),(50,50)]
    # M_star = 500

    # for c in centers:
    #     gx.add(M_star, c, (0,0))

    print(gx)

    # init window
    glfw.init()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)

    window = glfw.create_window(800, 800, "StarForge", None, None)
    if not window:
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, fb_width, fb_height)



    prog = ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=FRAGMENT_SHADER,
    )

    data = np.zeros(N, dtype=[('pos', 'f4', 2), ('mass', 'f4')])
    data['pos'] = gx.pos
    data['mass'] = gx.masses

    vbo = ctx.buffer(data.tobytes())
    vao = ctx.vertex_array(
    prog,
    [(vbo, "2f 1f", "in_pos", "in_mass")]
)
    prog["scale"].value = 100


    # main loop
    while not glfw.window_should_close(window):
        glfw.poll_events()

        update(gx)
        data['pos'] = gx.pos
        data['mass'] = gx.masses
        vbo.write(data.tobytes())
        ctx.clear(0.0, 0.0, 0.0)
        vao.render(mode=moderngl.POINTS)
        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
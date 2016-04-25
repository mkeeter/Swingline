#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include <png.h>
#include <epoxy/gl.h>
#include <GLFW/glfw3.h>

/******************************************************************************/

#define GLSL(src) "#version 330 core\n" #src
const char* voronoi_vert_src = GLSL(
    layout(location=0) in vec3 pos;
    layout(location=1) in vec2 offset;

    out vec3 color_;

    void main()
    {
        gl_Position = vec4(pos.xy + offset, pos.z, 1.0f);

        // Pick color based on instance ID
        int r = gl_InstanceID           % 256;
        int g = (gl_InstanceID / 256)   % 256;
        int b = (gl_InstanceID / 65536) % 256;
        color_ = vec3(r / 255.0f, g / 255.0f, b / 255.0f);
    }
);

const char* voronoi_frag_src = GLSL(
    in vec3 color_;
    layout (location=0) out vec4 color;

    void main()
    {
        color = vec4(color_, 1.0f);
    }
);

/******************************************************************************/

const char* quad_vert_src = GLSL(
    layout(location=0) in vec2 pos;

    void main()
    {
        gl_Position = vec4(pos, 0.0f, 1.0f);
    }
);

const char* sum_frag_src = GLSL(
    layout (location=0) out vec4 color;
    layout (pixel_center_integer) in vec4 gl_FragCoord;

    uniform sampler2D voronoi;

    void main()
    {
        int my_index = int(gl_FragCoord.x);
        ivec2 tex_size = textureSize(voronoi, 0);
        color = vec4(0.0f);

        // Iterate over all columns of the source image, accumulating a
        // weighted sum of the pixels that match our index
        for (int x=0; x < tex_size.x; x++)
        {
            vec4 t = texelFetch(voronoi, ivec2(x, gl_FragCoord.y), 0);
            int i = int(255.0f * (t.r + (t.g * 255.0f) + (t.b * 65535.0f)));
            if (i == my_index)
            {
                float wx = 1.0f; // Replace these with weights later
                float wy = 1.0f;

                color.x += x * wx;
                color.y += gl_FragCoord.y * wy;
                color.z += wx;
                color.w += wy;
            }
        }
    }
);

const char* feedback_src = GLSL(
    layout (location=0) in uint index;
    out vec2 pos;

    uniform sampler2D summed;
    void main()
    {
        ivec2 tex_size = textureSize(summed, 0);
        pos = vec2(0.0f, 0.0f);
        vec2 divisor = vec2(0.0f, 0.0f);
        for (int y=0; y < tex_size.y; ++y)
        {
            vec4 t = texelFetch(summed, ivec2(index, y), 0);
            pos += t.xy;
            divisor += t.zw;
        }
        pos /= divisor;
        pos = vec2(1.0f, 3.5);
    }
);


const char* blit_frag_src = GLSL(
    layout (location=0) out vec4 color;
    layout (pixel_center_integer) in vec4 gl_FragCoord;

    uniform sampler2D tex;

    void main()
    {
        vec4 t = texelFetch(tex, ivec2(gl_FragCoord.x, gl_FragCoord.y), 0);
        color = vec4(t.xyz, 1.0f);
    }
);

/******************************************************************************/

void check_shader(GLuint shader)
{
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint log_length;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);

        GLchar* info_log = (GLchar*)malloc((log_length + 1) * sizeof(GLchar));
        glGetShaderInfoLog(shader, log_length, NULL, info_log);
        fprintf(stderr, "Error: shader failed with error '%s'\n", info_log);
        exit(-1);
    }
}

GLuint build_shader(GLenum type, const GLchar* src)
{
    assert(type == GL_VERTEX_SHADER || type == GL_FRAGMENT_SHADER);

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    check_shader(shader);
    return shader;
}

void check_program(GLuint program)
{
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint log_length;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);

        GLchar* info_log = (GLchar*)malloc((log_length + 1) * sizeof(GLchar));
        glGetProgramInfoLog(program, log_length, NULL, info_log);
        fprintf(stderr, "Error: linking failed with error '%s'\n", info_log);
        exit(-1);
    }
}

GLuint build_program(GLuint vert, GLuint frag)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    check_program(program);
    return program;
}

/******************************************************************************/

/*
 *  Builds a vertex buffer to draw a single cone
 *  Must be called with a bound VAO; binds the cone into vertex attribute
 *  slot 0
 */
void build_cone(size_t n)
{
    GLuint vbo;
    size_t bytes = (n + 2) * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);

    /* This is the tip of the cone */
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = -1;

    for (size_t i=0; i <= n; ++i)
    {
        float angle = 2 * M_PI * i / n;
        buf[i*3 + 3] = cos(angle);
        buf[i*3 + 4] = sin(angle);
        buf[i*3 + 5] = 1;
    }

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes, buf, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

    free(buf);
}

/*
 *  Builds and returns the VBO for cone instances, binding it to vertex
 *  attribute slot 1
 */
GLuint build_instances(size_t n)
{
    GLuint vbo;
    size_t bytes = n * 2 * sizeof(float);
    float* buf = (float*)malloc(bytes);

    /*  Fill the buffer with random numbers between -1 and 1 */
    for (size_t i=0; i < 2*n; ++i)
    {
        buf[i] = ((float)rand() / RAND_MAX - 0.5) * 2;
    }

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes, buf, GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribDivisor(1, 1);

    free(buf);
    return vbo;
}

/******************************************************************************/

/*
 *  Builds a quad covering the viewport, returning the relevant VAO
 */
GLuint build_quad()
{
    GLfloat verts[] = {-1.0f, -1.0f,     1.0f, -1.0f,
                        1.0f,  1.0f,    -1.0f,  1.0f};
    GLuint vbo;
    GLuint vao;

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBindVertexArray(0);
    return vao;
}

/******************************************************************************/

GLuint build_feedback(size_t point_count)
{
    GLuint vao;
    GLuint vbo;
    size_t bytes = sizeof(GLuint) * point_count;
    GLuint* indices = (GLuint*)malloc(bytes);

    for (size_t i=0; i < point_count; ++i)
    {
        indices[i] = i;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, bytes, indices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 1, GL_UNSIGNED_INT, GL_FALSE, 0, 0);
    glBindVertexArray(0);

    free(indices);

    return vao;
}

/******************************************************************************/

/*
 *  Creates an OpenGL context (3.3 or higher)
 *  Returns a window pointer; the context is made current
 */
GLFWwindow* make_context(size_t width, size_t height)
{
    if (!glfwInit())
    {
        fprintf(stderr, "Error: Failed to initialize GLFW!\n");
        exit(-1);
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* const window = glfwCreateWindow(
            width, height, "swingline", NULL, NULL);

    if (!window)
    {
        fprintf(stderr, "Error:  Failed to create window");
        exit(-1);
    }

    glfwMakeContextCurrent(window);
    {   /* Check that the OpenGL version is new enough */
        const GLubyte* ver = glGetString(GL_VERSION);
        const size_t major = ver[0] - '0';
        const size_t minor = ver[2] - '0';
        if (major * 10 + minor < 33)
        {
            fprintf(stderr, "Error: OpenGL context is too old"
                            " (require 3.3, got %lu.%lu)\n", major, minor);
            exit(-1);
        }
    }
    return window;
}

/******************************************************************************/

GLuint new_texture()
{
    GLuint tex;
    glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    return tex;
}

/******************************************************************************/

void check_fbo(const char* description)
{   /*  Check to see if the framebuffer is complete  */
    int status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        fprintf(stderr, "Error: %s framebuffer is incomplete (%i)\n",
                description, status);
        exit(-1);
    }
}

void render_voronoi(GLuint program, GLuint fbo, GLuint vao,
                    size_t cone_res, size_t point_count)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glEnable(GL_DEPTH_TEST);
        glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
        glClearDepth(1.0f);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
            glBindVertexArray(vao);
                glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, cone_res + 2, point_count);
            glBindVertexArray(0);
        glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void render_sum(GLuint program, GLuint fbo, GLuint vao, GLuint tex,
                size_t point_count, size_t height)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glPushAttrib(GL_VIEWPORT_BIT);
            glViewport(0, 0, point_count, height);

            glClearColor(1.0f, 1.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glUseProgram(program);
                glBindVertexArray(vao);
                    glActiveTexture(GL_TEXTURE0);
                    glBindTexture(GL_TEXTURE_2D, tex);
                    glUniform1i(glGetUniformLocation(program, "tex"), 0);
                    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
                glBindVertexArray(0);
            glUseProgram(0);
        glPopAttrib();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void render_feedback(GLuint vao, GLuint vbo, GLuint tex,
                     GLuint program, size_t point_count)
{
    glEnable(GL_RASTERIZER_DISCARD);
    glBindVertexArray(vao);
        glUseProgram(program);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(glGetUniformLocation(program, "tex"), 0);
            glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo);
            glBeginTransformFeedback(GL_POINTS);
                glDrawArrays(GL_POINTS, 0, point_count);
            glEndTransformFeedback();
        glUseProgram(0);
    glBindVertexArray(0);
    glDisable(GL_RASTERIZER_DISCARD);
    glFlush();

    {
        GLfloat feedback[point_count*2];
        glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof(feedback), feedback);
        for (unsigned i=0; i < point_count*2; i += 2)
        {
            printf("%f %f\n", feedback[i], feedback[i+1]);
        }
    }
}

/******************************************************************************/

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    static const size_t cone_res = 64;
    static const size_t point_count = 10;
    static const size_t width = 10;
    static const size_t height = 10;

    GLFWwindow* win = make_context(width, height);

    /*************************************************************************/
    /*  Generate all of the parts used in the voronoi rendering step         */
    GLuint voronoi_vao;
    glGenVertexArrays(1, &voronoi_vao);

    glBindVertexArray(voronoi_vao);
        build_cone(cone_res);           /* Uses bound VAO   */
        GLuint voronoi_vbo = build_instances(point_count);   /* (same) */
    glBindVertexArray(0);

    GLuint voronoi_program = build_program(
        build_shader(GL_VERTEX_SHADER, voronoi_vert_src),
        build_shader(GL_FRAGMENT_SHADER, voronoi_frag_src));

    GLuint voronoi_tex = new_texture();
    GLuint voronoi_depth = new_texture();

    glBindTexture(GL_TEXTURE_2D, voronoi_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,
                     0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, voronoi_depth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height,
                     0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint voronoi_fbo;
    glGenFramebuffers(1, &voronoi_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, voronoi_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, voronoi_tex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                               GL_TEXTURE_2D, voronoi_depth, 0);
    check_fbo("voronoi");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    /*************************************************************************/
    /*  Build everything needed for the summing stage                        */
    GLuint quad_vao = build_quad();

    GLuint sum_tex = new_texture();
    glBindTexture(GL_TEXTURE_2D, sum_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, point_count, height,
                     0, GL_RGB, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint sum_fbo;
    glGenFramebuffers(1, &sum_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sum_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, sum_tex, 0);
    check_fbo("sum");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLuint sum_program = build_program(
        build_shader(GL_VERTEX_SHADER, quad_vert_src),
        build_shader(GL_FRAGMENT_SHADER, sum_frag_src));

    /*************************************************************************/
    /*  Build everything needed for the transform feedback stage             */
    GLuint feedback_shader = build_shader(GL_VERTEX_SHADER, feedback_src);
    GLuint feedback_program = glCreateProgram();
    GLuint feedback_vao = build_feedback(point_count);
    glAttachShader(feedback_program, feedback_shader);
    const GLchar* feedback_varying[] = { "pos" };
    glTransformFeedbackVaryings(feedback_program, 1, feedback_varying,
                                GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(feedback_program);
    check_program(feedback_program);
    /*************************************************************************/

    render_voronoi(voronoi_program, voronoi_fbo, voronoi_vao,
                   cone_res, point_count);
    render_sum(sum_program, sum_fbo, quad_vao, voronoi_tex,
               point_count, height);
    render_feedback(feedback_vao, voronoi_vbo, sum_tex, feedback_program, point_count);

    /*************************************************************************/

    glUseProgram(sum_program);
    glBindVertexArray(quad_vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, voronoi_tex);
    glUniform1i(glGetUniformLocation(sum_program, "tex"), 0);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);

    while (!glfwWindowShouldClose(win))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        // Swap front and back buffers
        glfwSwapBuffers(win);

        // Poll for and process events
        glfwPollEvents();
    }

    return 0;
}

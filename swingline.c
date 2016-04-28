#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include <png.h>
#include <epoxy/gl.h>
#include <GLFW/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

/******************************************************************************/

#define GLSL(src) "#version 330 core\n" #src

const char* voronoi_vert_src = GLSL(
    layout(location=0) in vec3 pos;     /*  Absolute coordinates  */
    layout(location=1) in vec2 offset;  /*  0 to 1 */
    uniform vec2 scale;

    out vec3 color_;

    void main()
    {
        gl_Position = vec4(pos.xy*scale + 2.0f*offset - 1.0f, pos.z, 1.0f);

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
    out vec2 pos_;

    void main()
    {
        gl_Position = vec4(pos, 0.0f, 1.0f);
        pos_ = vec2((pos + 1.0f) / 2.0f);
    }
);

const char* blit_frag_src = GLSL(
    layout (location=0) out vec4 color;
    in vec2 pos_;  /* 0 to 1 range */

    uniform sampler2D tex;

    float rand(vec2 co)
    {
        return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
    }

    void main()
    {
        vec4 t = texture(tex, pos_);
        vec3 rgb = vec3(rand(vec2(t.x, t.y)),
                        rand(vec2(t.y, t.x)),
                        rand(vec2(t.x - t.y, t.x)));
        color = vec4(0.9f + 0.1f*rgb, 1.0f);
    }
);

const char* sum_frag_src = GLSL(
    layout (location=0) out vec4 color;
    layout (pixel_center_integer) in vec4 gl_FragCoord;

    uniform sampler2D voronoi;
    uniform sampler2D img;

    void main()
    {
        int my_index = int(gl_FragCoord.x);
        ivec2 tex_size = textureSize(voronoi, 0);
        color = vec4(0.0f);

        // Iterate over all columns of the source image, accumulating a
        // weighted sum of the pixels that match our index
        for (int x=0; x < tex_size.x; x++)
        {
            ivec2 coord = ivec2(x, gl_FragCoord.y);
            vec4 t = texelFetch(voronoi, coord, 0);
            int i = int(255.0f * (t.r + (t.g * 256.0f) + (t.b * 65536.0f)));
            if (i == my_index)
            {
                float weight = 1 - texelFetch(img, coord, 0)[0];
                color.xy += (coord + 0.5f) * weight;
                color.w += weight;
                color.z += 1.0f;
            }
        }

        // Normalize to the 0 - 1 range
        color.x = color.x / tex_size.x;
        color.y = color.y / tex_size.y;
    }
);

const char* feedback_src = GLSL(
    layout (location=0) in uint index;
    out vec3 pos;

    uniform sampler2D summed;
    void main()
    {
        ivec2 tex_size = textureSize(summed, 0);
        pos = vec3(0.0f, 0.0f, 0.0f);
        float weight = 0.0f;
        float count = 0;
        for (int y=0; y < tex_size.y; ++y)
        {
            vec4 t = texelFetch(summed, ivec2(index, y), 0);
            pos.xy += t.xy;
            weight += t.w;
            count += t.z;
        }
        if (weight != 0)
        {
            pos.xy /= weight;
        }
        pos.z = weight / count;
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

void teardown(GLint* viewport)
{
    glBindVertexArray(0);
    glUseProgram(0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    if (viewport)
    {
        glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    }
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
    size_t bytes = n * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);

    /*  Fill the buffer with random numbers between -1 and 1 */
    for (size_t i=0; i < n; ++i)
    {
        buf[3*i]     = (float)rand() / RAND_MAX;
        buf[3*i + 1] = (float)rand() / RAND_MAX;
        buf[3*i + 2] = 0.0f;
    }

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bytes, buf, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 3*sizeof(float), 0);
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

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

////////////////////////////////////////////////////////////////////////////////

typedef struct Config_ {
    size_t width, height;   /* Image size   */
    size_t samples;         /*  Number of Voronoi cells */
    size_t resolution;      /*   Resolution of Voronoi cones  */

    float sx, sy;           /*  Scale (used to adjust for aspect ratio) */
} Config;

Config* config_new(size_t width, size_t height, size_t samples,
                   size_t resolution)
{
    Config* c = (Config*)calloc(1, sizeof(Config));

    (*c) = (Config){
        .width = width,
        .height = height,
        .samples = samples,
        .resolution = resolution};

    if (width > height)
    {
        c->sx = 1;
        c->sy = (float)width / (float)height;
    }
    else
    {
        c->sx = (float)height / (float)width;
        c->sy = 1;
    }
    return c;
}

////////////////////////////////////////////////////////////////////////////////

typedef struct Voronoi_ {
    GLuint vao;     /*  VAO with bound cone and offsets */
    GLuint pts;     /*  VBO containing point locations  */
    GLuint prog;    /*  Shader program (compiled)       */
    GLuint img;     /*  Target image texture            */

    GLuint tex;     /*  RGB texture (bound to fbo)          */
    GLuint depth;   /*  Depth texture (bound to fbo)        */
    GLuint fbo;     /*  Framebuffer for render-to-texture   */
} Voronoi;

Voronoi* voronoi_new(const Config* cfg, uint8_t* img)
{
    Voronoi* v = (Voronoi*)calloc(1, sizeof(Voronoi));
    glGenVertexArrays(1, &v->vao);

    glBindVertexArray(v->vao);
        build_cone(cfg->resolution);           /* Uses bound VAO   */
        v->pts = build_instances(cfg->samples);   /* (same) */
    glBindVertexArray(0);

    v->prog = build_program(
        build_shader(GL_VERTEX_SHADER, voronoi_vert_src),
        build_shader(GL_FRAGMENT_SHADER, voronoi_frag_src));

    v->tex   = new_texture();
    v->depth = new_texture();
    v->img   = new_texture();

    glBindTexture(GL_TEXTURE_2D, v->tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, cfg->width, cfg->height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, v->depth);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, cfg->width, cfg->height,
                 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, v->img);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, cfg->width, cfg->height,
                 0, GL_RED, GL_UNSIGNED_BYTE, img);

    glGenFramebuffers(1, &v->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, v->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, v->tex, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, v->depth, 0);
    check_fbo("voronoi");

    teardown(NULL);
    return v;
}


void voronoi_draw(Config* cfg, Voronoi* v)
{
    glBindFramebuffer(GL_FRAMEBUFFER, v->fbo);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(0, 0, cfg->width, cfg->height);

    glEnable(GL_DEPTH_TEST);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glUseProgram(v->prog);
    glBindVertexArray(v->vao);
    glUniform2f(glGetUniformLocation(v->prog, "scale"), cfg->sx, cfg->sy);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, cfg->resolution+2, cfg->samples);

    teardown(viewport);
}

////////////////////////////////////////////////////////////////////////////////

typedef struct Sum_
{
    GLuint prog;
    GLuint fbo;
    GLuint tex;
    GLuint vao;
} Sum;

Sum* sum_new(Config* config)
{
    Sum* sum = (Sum*)calloc(1, sizeof(Sum));
    sum->vao = build_quad();
    sum->tex = new_texture();
    glBindTexture(GL_TEXTURE_2D, sum->tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, config->samples, config->height,
                     0, GL_RGB, GL_FLOAT, 0);

    glGenFramebuffers(1, &sum->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sum->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, sum->tex, 0);
    check_fbo("sum");

    sum->prog = build_program(
        build_shader(GL_VERTEX_SHADER, quad_vert_src),
        build_shader(GL_FRAGMENT_SHADER, sum_frag_src));

    teardown(NULL);
    return sum;
}

void sum_draw(Config* cfg, Voronoi* v, Sum* s)
{
    glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);

    // Save viewport size and restore it later
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    glViewport(0, 0, cfg->samples, cfg->height);

    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(s->prog);
    glBindVertexArray(s->vao);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, v->tex);
    glUniform1i(glGetUniformLocation(s->prog, "voronoi"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, v->img);
    glUniform1i(glGetUniformLocation(s->prog, "img"), 1);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    teardown(viewport);
}

////////////////////////////////////////////////////////////////////////////////

typedef struct Feedback_
{
    GLuint vao;
    GLuint prog;
} Feedback;

GLuint feedback_indices(size_t samples)
{
    GLuint vao;
    GLuint vbo;
    size_t bytes = sizeof(GLuint) * samples;
    GLuint* indices = (GLuint*)malloc(bytes);

    for (size_t i=0; i < samples; ++i)
    {
        indices[i] = i;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, bytes, indices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribIPointer(0, 1, GL_UNSIGNED_INT, 0, 0);
    glBindVertexArray(0);

    free(indices);

    return vao;
}

Feedback* feedback_new(GLuint samples)
{
    Feedback* f = (Feedback*)calloc(1, sizeof(Feedback));

    f->prog = glCreateProgram();
    GLuint shader = build_shader(GL_VERTEX_SHADER, feedback_src);
    glAttachShader(f->prog, shader);
    const GLchar* varying[] = { "pos" };
    glTransformFeedbackVaryings(f->prog, 1, varying, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(f->prog);
    check_program(f->prog);

    f->vao = feedback_indices(samples);

    return f;
}

void feedback_draw(Config* cfg, Voronoi* v, Sum* s, Feedback* f)
{
    glEnable(GL_RASTERIZER_DISCARD);
    glBindVertexArray(f->vao);
    glUseProgram(f->prog);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, s->tex);
    glUniform1i(glGetUniformLocation(f->prog, "tex"), 0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, v->pts);

    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, cfg->samples);
    glEndTransformFeedback();

    glDisable(GL_RASTERIZER_DISCARD);
    teardown(NULL);
}

/******************************************************************************/

const char* stipples_vert_src = GLSL(
    layout(location=0) in vec2 pos;     /*  Absolute coordinates  */
    layout(location=1) in vec3 offset;  /*  0 to 1 */

    /*  Seperate radii to compensate for window aspect ratio  */
    uniform vec2 radius;

    void main()
    {
        float r = 0.2f + 0.8f * offset.z;
        vec2 scaled = vec2(pos.x * radius.x, pos.y * radius.y) * r;
        gl_Position = vec4(scaled + 2.0f*offset.xy - 1.0f, 0.0f, 1.0f);
    }
);

const char* stipples_frag_src = GLSL(
    layout (location=0) out vec4 color;

    void main()
    {
        color = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
);

typedef struct Stipples_
{
    GLuint vao;
    GLuint prog;
} Stipples;

Stipples* stipples_new(Config* cfg, Voronoi* v)
{
    Stipples* s = (Stipples*)calloc(1, sizeof(Stipples));

    glGenVertexArrays(1, &s->vao);
    glBindVertexArray(s->vao);

    {   // Make and bind a VBO that draws a simple circle
        GLuint vbo;
        size_t bytes = (2 + cfg->resolution) * 2 * sizeof(float);
        float* buf = (float*)malloc(bytes);

        buf[0] = 0;
        buf[1] = 0;
        for (size_t i=0; i <= cfg->resolution; ++i)
        {
            float angle = 2 * M_PI * i / cfg->resolution;
            buf[i*2 + 2] = cos(angle);
            buf[i*2 + 3] = sin(angle);
        }

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, bytes, buf, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

        free(buf);
    }

    // Bind the Voronoi points array to location 1 in the VAO
    glBindBuffer(GL_ARRAY_BUFFER, v->pts);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glVertexAttribDivisor(1, 1);

    s->prog = build_program(
        build_shader(GL_VERTEX_SHADER, stipples_vert_src),
        build_shader(GL_FRAGMENT_SHADER, stipples_frag_src));

    teardown(NULL);
    return s;
}

void stipples_draw(Config* cfg, Stipples* s)
{
    glUseProgram(s->prog);

    glUniform2f(glGetUniformLocation(s->prog, "radius"),
                0.01 * cfg->sx, 0.01 * cfg->sy);
    glBindVertexArray(s->vao);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, cfg->resolution+2, cfg->samples);

    teardown(NULL);
}

/******************************************************************************/

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Usage: swingline image.jpg\n");
        exit(-1);
    }

    int x, y;
    stbi_set_flip_vertically_on_load(true);
    stbi_uc* img = stbi_load(argv[1], &x, &y, NULL, 1);
    if (img == NULL)
    {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        exit(-1);
    }

    Config* c = config_new(x, y, 100, 64);
    GLFWwindow* win = make_context(c->width, c->height);

    /*  These are the three stages in the stipple update loop   */
    Voronoi* v = voronoi_new(c, img);
    Sum* s = sum_new(c);
    Feedback* f = feedback_new(c->samples);

    /*  These are used for rendering to the screen  */
    GLuint quad_vao = build_quad();
    GLuint blit_program = build_program(
        build_shader(GL_VERTEX_SHADER, quad_vert_src),
        build_shader(GL_FRAGMENT_SHADER, blit_frag_src));
    Stipples* stipples = stipples_new(c, v);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);

    while (!glfwWindowShouldClose(win))
    {
        /*  Render the current voronoi diagram's state  */
        voronoi_draw(c, v);

        /*  Calculate the centroids and write them to v->pts  */
        sum_draw(c, v, s);
        feedback_draw(c, v, s, f);

        /*  Then draw the quad   */
        glBindVertexArray(quad_vao);
        glUseProgram(blit_program);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, v->tex);
        glUniform1i(glGetUniformLocation(blit_program, "tex"), 0);

        glDisable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT);

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

        /*  Render cell centroids as white dots  */
        stipples_draw(c, stipples);

        /*  Draw and poll   */
        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    return 0;
}

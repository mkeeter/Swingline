/*
    Swingline - Weighted Voronoi Stippling on the GPU
    Copyright (C) 2016 Matthew Keeter

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

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

    float rand(float a, float b)
    {
        return fract(sin(a*12.9898 + b*78.233) * 43758.5453);
    }

    void main()
    {
        vec4 t = texture(tex, pos_);
        vec3 rgb = vec3(rand(t.x, t.y), rand(t.y, t.x), rand(t.x - t.y, t.x));
        color = vec4(0.9f + 0.1f*rgb, 1.0f);
    }
);

const char* sum_frag_src = GLSL(
    layout (pixel_center_integer) in vec4 gl_FragCoord;
    out vec4 color;

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
                float weight = 1.0f - texelFetch(img, coord, 0)[0];
                weight = 0.01f + 0.99f * weight;

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
        pos.xy /= weight;
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

GLuint shader_compile(GLenum type, const GLchar* src)
{
    assert(type == GL_VERTEX_SHADER || type == GL_FRAGMENT_SHADER);

    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    check_shader(shader);
    return shader;
}

void program_check(GLuint program)
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

GLuint program_link(GLuint vert, GLuint frag)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    program_check(program);
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

/******************************************************************************/

/*
 *  Builds a quad covering the viewport, returning the relevant VAO
 */
GLuint quad_new()
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
GLFWwindow* make_context(uint16_t width, uint16_t height, bool hide)
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
    glfwWindowHint(GLFW_VISIBLE, !hide);

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
        const uint8_t major = ver[0] - '0';
        const uint8_t minor = ver[2] - '0';
        if (major * 10 + minor < 33)
        {
            fprintf(stderr, "Error: OpenGL context is too old"
                            " (require 3.3, got %u.%u)\n", major, minor);
            exit(-1);
        }
    }
    return window;
}

/******************************************************************************/

GLuint texture_new()
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

void fbo_check(const char* description)
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
    stbi_uc* img;           /*  Pointer to raw image data  */

    uint16_t width, height; /*  Image size   */
    uint16_t samples;       /*  Number of Voronoi cells */
    uint16_t resolution;    /*  Resolution of Voronoi cones  */

    float sx, sy;           /*  Scale (used to adjust for aspect ratio) */
    float radius;           /*  Stipple radius (in arbitrary units)     */

    int iter;               /*  Number of iterations; -1 if interactive */
    const char* out;        /*  Output file name  */
} Config;

void config_set_aspect_ratio(Config* c)
{
    if (c->width > c->height)
    {
        c->sx = 1;
        c->sy = (float)c->width / (float)c->height;
    }
    else
    {
        c->sx = (float)c->height / (float)c->width;
        c->sy = 1;
    }
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

/*
 *  Builds a vertex buffer to draw a single cone
 *  Must be called with a bound VAO; binds the cone into vertex attribute
 *  slot 0
 */
void voronoi_cone_bind(uint16_t n)
{
    GLuint vbo;
    size_t bytes = (n + 2) * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);

    /* This is the tip of the cone */
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = -1;

    for (uint16_t i=0; i <= n; ++i)
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
GLuint voronoi_instances(const Config* c)
{
    GLuint vbo;
    size_t bytes = c->samples * 3 * sizeof(float);
    float* buf = (float*)malloc(bytes);

    /*  Fill the buffer with values between 0 and 1, using        *
     *  rejection sampling to create a good initial distribution  */
    uint16_t i=0;
    while (i < c->samples)
    {
        int x = rand() % c->width;
        int y = rand() % c->height;
        uint8_t p = c->img[y*c->width + x];

        if ((rand() % 256) > p)
        {
            buf[3*i]     = (x + 0.5f) / c->width;
            buf[3*i + 1] = (y + 0.5f) / c->height;
            buf[3*i + 2] = 0.0f;
            i++;
        }
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

Voronoi* voronoi_new(const Config* cfg, uint8_t* img)
{
    Voronoi* v = (Voronoi*)calloc(1, sizeof(Voronoi));
    glGenVertexArrays(1, &v->vao);

    glBindVertexArray(v->vao);
        voronoi_cone_bind(cfg->resolution);         /* Uses bound VAO   */
        v->pts = voronoi_instances(cfg);            /* (same) */
    glBindVertexArray(0);

    v->prog = program_link(
        shader_compile(GL_VERTEX_SHADER, voronoi_vert_src),
        shader_compile(GL_FRAGMENT_SHADER, voronoi_frag_src));

    v->tex   = texture_new();
    v->depth = texture_new();
    v->img   = texture_new();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
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
    fbo_check("voronoi");

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
    sum->vao = quad_new();
    sum->tex = texture_new();
    glBindTexture(GL_TEXTURE_2D, sum->tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, config->samples, config->height,
                     0, GL_RGB, GL_FLOAT, 0);

    glGenFramebuffers(1, &sum->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sum->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, sum->tex, 0);
    fbo_check("sum");

    sum->prog = program_link(
        shader_compile(GL_VERTEX_SHADER, quad_vert_src),
        shader_compile(GL_FRAGMENT_SHADER, sum_frag_src));

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

GLuint feedback_indices(uint16_t samples)
{
    GLuint vao;
    GLuint vbo;
    size_t bytes = sizeof(GLuint) * samples;
    GLuint* indices = (GLuint*)malloc(bytes);

    for (uint16_t i=0; i < samples; ++i)
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
    GLuint shader = shader_compile(GL_VERTEX_SHADER, feedback_src);
    glAttachShader(f->prog, shader);
    const GLchar* varying[] = { "pos" };
    glTransformFeedbackVaryings(f->prog, 1, varying, GL_INTERLEAVED_ATTRIBS);
    glLinkProgram(f->prog);
    program_check(f->prog);

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
        vec2 scaled = vec2(pos.x * radius.x, pos.y * radius.y) * sqrt(offset.z);
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

    s->prog = program_link(
        shader_compile(GL_VERTEX_SHADER, stipples_vert_src),
        shader_compile(GL_FRAGMENT_SHADER, stipples_frag_src));

    teardown(NULL);
    return s;
}

void stipples_draw(Config* cfg, Stipples* s)
{
    glUseProgram(s->prog);

    glUniform2f(glGetUniformLocation(s->prog, "radius"),
                cfg->radius * cfg->sx, cfg->radius * cfg->sy);
    glBindVertexArray(s->vao);
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, cfg->resolution+2, cfg->samples);

    teardown(NULL);
}

/******************************************************************************/

void print_usage(char* prog)
{
    fprintf(stderr, "Usage: %s [-n samples] [-r radius] [-o output] "
                              "[-i iterations] image\n", prog);
}

Config* parse_args(int argc, char** argv)
{
    unsigned n = 1000;
    float r = 0.01f;
    int iter = -1;
    const char* out = NULL;

    while (true)
    {
        char c = getopt(argc, argv, "r:n:o:i:");
        if (c == -1) {  break; }

        switch (c)
        {
            case 'n':
                n = atoi(optarg);
                break;
            case 'i':
                iter = atoi(optarg);
                break;
            case 'o':
                out = optarg;
                break;
            case 'r':
                r = 0.01f * atof(optarg);
                break;
            default:
                print_usage(argv[0]);
                exit(EXIT_FAILURE);
        };
    }

    if (optind >= argc)
    {
        fprintf(stderr, "%s: expected filename after options\n", argv[0]);
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    else if (n > UINT16_MAX)
    {
        fprintf(stderr, "Error: too many points (%i)\n", n);
        exit(-1);
    }

    int x, y;
    stbi_set_flip_vertically_on_load(true);
    stbi_uc* img = stbi_load(argv[optind], &x, &y, NULL, 1);

    if (img == NULL)
    {
        fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
        exit(-1);
    }
    else if ((unsigned)x > UINT16_MAX || (unsigned)y > UINT16_MAX)
    {
        fprintf(stderr, "Error: image is too large (%i x %i)\n", x, y);
        exit(-1);
    }

    if (out)
    {
        size_t len = strlen(out);
        if (len >= 4 && strcmp(out + len - 4, ".svg"))
        {
            fprintf(stderr, "Error: output file should end in .svg (%s)\n", out);
            exit(-1);
        }
    }

    Config* c = (Config*)calloc(1, sizeof(Config));
    (*c) = (Config){
        .img = img,
        .width = (uint16_t)x,
        .height = (uint16_t)y,
        .samples = (uint16_t)n,
        .resolution = 256,
        .radius = r,
        .iter = iter,
        .out = out};

    config_set_aspect_ratio(c);
    return c;
}

int main(int argc, char** argv)
{
    Config* c = parse_args(argc, argv);
    GLFWwindow* win = make_context(c->width, c->height, c->iter != -1);

    /*  These are the three stages in the stipple update loop   */
    Voronoi* v = voronoi_new(c, c->img);
    Sum* s = sum_new(c);
    Feedback* f = feedback_new(c->samples);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClearDepth(1.0f);

    if (c->iter == -1)  /* Interactive mode */
    {
        /*  These are used for rendering to the screen  */
        GLuint quad_vao = quad_new();
        GLuint blit_program = program_link(
            shader_compile(GL_VERTEX_SHADER, quad_vert_src),
            shader_compile(GL_FRAGMENT_SHADER, blit_frag_src));
        Stipples* stipples = stipples_new(c, v);

        while (!glfwWindowShouldClose(win))
        {
            /*  Render the current voronoi diagram's state to v->tex */
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
    }
    else    /* Non-interactive mode */
    {
        for (int i=0; i < c->iter; ++i)
        {
            printf("\r%s: %i / %i", argv[0], i + 1, c->iter);
            fflush(stdout);
            voronoi_draw(c, v);
            sum_draw(c, v, s);
            feedback_draw(c, v, s, f);
        }
        printf("\n");
    }

    if (c->out)
    {
        FILE* f = fopen(c->out, "w");
        if (!f)
        {
            perror("File opening failed");
            return EXIT_FAILURE;
        }

        fprintf(f,
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
            "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\"\n"
            "    viewBox=\"0 0 %u %u\" width=\"%u\" height=\"%u\" id=\"swingline\">\n",
            c->width, c->height, c->width, c->height);

        glBindBuffer(GL_ARRAY_BUFFER, v->pts);
        size_t bytes = 3 * sizeof(float) * c->samples;
        float (*pts)[3] = (float (*)[3])malloc(3 * sizeof(bytes) * c->samples);
        glGetBufferSubData(GL_ARRAY_BUFFER, 0, bytes, pts);

        for (int i=0; i < c->samples; ++i)
        {
            fprintf(f,
                "    <circle cx=\"%f\" cy=\"%f\" r=\"%f\" fill=\"black\" />\n",
                c->width*pts[i][0], c->height - c->height*pts[i][1],
                c->radius * fmin(c->sx, c->sy) * fmin(c->width, c->height) *
                    pts[i][2]);
        }

        free(pts);
        fprintf(f, "</svg>");
        fclose(f);
    }

    return 0;
}

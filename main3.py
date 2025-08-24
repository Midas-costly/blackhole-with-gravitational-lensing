#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Black Hole with Gravitational Lensing (pygame + PyOpenGL)
------------------------------------------------------------
Features:
- Orbit camera (LMB drag), scroll to zoom
- Space-time grid (x-z plane)
- Static luminous bodies (spheres)
- Screen-space postprocess lensing shader that warps the entire scene near the BH
- Black hole overlay (disc + soft glow)

Keys:
- ESC / Q : quit
- R       : reset camera
- G       : toggle grid
- L       : toggle lensing
- +/-     : change lens strength
- [ / ]   : change lens radius

Mouse:
- Left drag: orbit yaw/pitch
- Wheel: zoom

Requires:
- pygame
- PyOpenGL, PyOpenGL_accelerate (recommended)
"""

import sys, math, ctypes, random
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

WIN_W, WIN_H = 1280, 800
FOV = 60.0
NEAR, FAR = 0.1, 200.0

# Camera spherical coordinates (orbit)
cam_dist = 18.0
cam_yaw = 35.0
cam_pitch = -15.0

# Toggles
show_grid = True
enable_lensing = True

# Lensing params (shader)
lens_strength = 0.16     # higher -> stronger bending
lens_radius = 0.55       # fraction of screen half-min dimension (in NDC uv)
event_horizon_radius_px = 70  # black disc radius in pixels (overlay)

# Star bodies
STAR_POSITIONS = [
    ( 6.5,  2.2, -9.0, (1.0, 0.95, 0.2)),  # yellow
    (-7.0,  3.0, -12.0, (1.0, 0.4, 0.3)),  # red/orange
    ( 10.0, 1.0, -14.0, (0.6, 0.8, 1.0)),  # blue-white
    (-4.0, -1.0, -8.0, (1.0, 1.0, 1.0)),   # white
]

GRID_SIZE = 24
GRID_STEP = 1.0

def compile_shader(src, shader_type):
    sid = glCreateShader(shader_type)
    glShaderSource(sid, src)
    glCompileShader(sid)
    ok = glGetShaderiv(sid, GL_COMPILE_STATUS)
    if not ok:
        log = glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid

def link_program(vs_src, fs_src):
    vs = compile_shader(vs_src, GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, GL_FRAGMENT_SHADER)
    pid = glCreateProgram()
    glAttachShader(pid, vs)
    glAttachShader(pid, fs)
    glLinkProgram(pid)
    ok = glGetProgramiv(pid, GL_LINK_STATUS)
    if not ok:
        log = glGetProgramInfoLog(pid).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return pid

def create_fbo_tex(w, h):
    
    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
   
    rbo = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, rbo)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("FBO incomplete")
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    return fbo, tex, rbo

# Fullscreen quad (two triangles)
def create_fs_quad():
    verts = [
        -1.0, -1.0,  0.0, 0.0,
         1.0, -1.0,  1.0, 0.0,
         1.0,  1.0,  1.0, 1.0,
        -1.0,  1.0,  0.0, 1.0,
    ]
    idx = [0,1,2, 2,3,0]
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    array_type = (ctypes.c_float * len(verts))
    glBufferData(GL_ARRAY_BUFFER, len(verts)*4, array_type(*verts), GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    idx_type = (ctypes.c_uint * len(idx))
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, len(idx)*4, idx_type(*idx), GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
    return vbo, ebo

# ---------- Postprocess shader ----------

POST_VS = """
#version 120
attribute vec2 aPos;
attribute vec2 aUV;
varying vec2 vUV;
void main(){
    vUV = aUV;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

# Simple radial lensing. We treat BH at screen center.
# We warp UVs by adding a displacement inversely proportional to distance to BH center.
POST_FS = """
#version 120
uniform sampler2D uScene;
uniform vec2 uResolution;   // screen size in pixels
uniform float uStrength;    // lens strength
uniform float uRadius;      // influence radius in normalized space (0..1, where 1 ~ half-min dimension)
uniform float uEHpx;        // event horizon radius in pixels
varying vec2 vUV;

void main(){
    vec2 res = uResolution;
    vec2 uv = vUV;

    // convert to NDC-like centered coords where y up, x right
    vec2 p = (gl_FragCoord.xy - 0.5*res);
    float minHalf = min(res.x, res.y) * 0.5;

    float d = length(p);                        // pixels from center
    float dnorm = d / minHalf;                  // 0..~1 range
    // If inside the event horizon, render black
    if (d <= uEHpx){
        gl_FragColor = vec4(0.0,0.0,0.0,1.0);
        return;
    }

    // Only warp within influence radius (soft falloff)
    float r = uRadius;  // normalized
    vec2 dir = (d > 1e-5) ? (p / d) : vec2(0.0,0.0);

    float warp = 0.0;
    if (dnorm < r){
        // Bell-ish curve that grows as we approach center
        float t = 1.0 - (dnorm / r);
        // inverse-like term for bending (avoid singularities)
        warp = uStrength * t * (1.0 / (dnorm + 0.02));
    }

    vec2 disp = dir * warp * (minHalf / res);  // map back to UV space
    vec2 uv2 = uv + disp;

    // sample scene
    vec4 col = texture2D(uScene, uv2);

    // subtle vignette for drama
    float vig = 1.0 - pow(dnorm, 2.0) * 0.25;
    col.rgb *= clamp(vig, 0.8, 1.0);

    gl_FragColor = col;
}
"""

# ---------- Overlay (2D) drawing ----------
def draw_black_hole_overlay(screen_w, screen_h, eh_px):
    # Switch to simple ortho and draw a black disc + glow as screen-space overlay
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, screen_w, screen_h, 0, -1, 1)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    cx, cy = screen_w * 0.5, screen_h * 0.5

    def ring(r, a):
        glColor4f(0.25, 0.45, 0.9, a)
        glBegin(GL_TRIANGLE_FAN)
        glVertex2f(cx, cy)
        segments = 64
        for i in range(segments+1):
            th = 2.0 * math.pi * (i / segments)
            glVertex2f(cx + math.cos(th)*r, cy + math.sin(th)*r)
        glEnd()

    for i in range(5, 0, -1):
        ring(eh_px*(1.0 + 0.45*i), 0.08 + 0.03*i)

    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(cx, cy)
    segments = 96
    for i in range(segments+1):
        th = 2.0 * math.pi * (i / segments)
        glVertex2f(cx + math.cos(th)*eh_px, cy + math.sin(th)*eh_px)
    glEnd()

    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# ---------- Scene drawing (fixed pipeline for simplicity) ----------

def set_perspective(w, h):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV, float(w)/float(h), NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)

def set_camera():
    # Camera orbit around origin
    glLoadIdentity()
    yaw_r = math.radians(cam_yaw)
    pitch_r = math.radians(cam_pitch)
    x = cam_dist * math.cos(pitch_r) * math.sin(yaw_r)
    y = cam_dist * math.sin(pitch_r)
    z = cam_dist * math.cos(pitch_r) * math.cos(yaw_r)
    gluLookAt(x, y, z,   0, 0, 0,   0, 1, 0)

def draw_grid():
    if not show_grid: return
    glDisable(GL_LIGHTING)
    glLineWidth(1.0)
    glColor3f(0.25, 0.5, 0.9)
    glBegin(GL_LINES)
    # x/z plane grid at y=0
    s = GRID_SIZE
    step = GRID_STEP
    for i in range(-s, s+1):
        glVertex3f(i*step, 0.0, -s*step)
        glVertex3f(i*step, 0.0,  s*step)
        glVertex3f(-s*step, 0.0, i*step)
        glVertex3f( s*step, 0.0, i*step)
    glEnd()

def draw_sphere(radius, color=(1,1,1), slices=32, stacks=24):
    glColor3f(*color)
    quad = gluNewQuadric()
    gluSphere(quad, radius, slices, stacks)
    gluDeleteQuadric(quad)

def draw_scene_objects():
    # Stars (static luminous bodies)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (ctypes.c_float*4)(0.0, 5.0, 5.0, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (ctypes.c_float*4)(1.0, 1.0, 1.0, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, (ctypes.c_float*4)(0.5,0.5,0.5,1.0))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 12.0)

    # draw stars
    for (sx, sy, sz, col) in STAR_POSITIONS:
        glPushMatrix()
        glTranslatef(sx, sy, sz)
        draw_sphere(0.6, color=col)
        glPopMatrix()

    glDisable(GL_LIGHTING)

    # (Optional) faint accretion disk as a textured/colored ring (simple color here)
    glPushMatrix()
    glRotatef(90, 1, 0, 0)  # lie flat in x-z
    glColor4f(1.0, 0.8, 0.2, 0.7)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glBegin(GL_TRIANGLE_STRIP)
    segments = 128
    r1, r2 = 2.5, 6.5
    for i in range(segments+1):
        th = 2*math.pi * (i/segments)
        c, s = math.cos(th), math.sin(th)
        glVertex3f(c*r1, s*r1, 0.0)
        glVertex3f(c*r2, s*r2, 0.0)
    glEnd()
    glDisable(GL_BLEND)
    glPopMatrix()

# ---------- Main ----------

def main():
    global WIN_W, WIN_H
    global cam_dist, cam_yaw, cam_pitch
    global show_grid, enable_lensing
    global lens_strength, lens_radius, event_horizon_radius_px

    pygame.init()
    pygame.display.set_caption("Blackhole")
    pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)

    # GL setup
    glViewport(0, 0, WIN_W, WIN_H)
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.02, 0.05, 0.10, 1.0)

    set_perspective(WIN_W, WIN_H)

    # Create FBO for scene render
    fbo, scene_tex, rbo = create_fbo_tex(WIN_W, WIN_H)

    # Create fullscreen quad
    fs_vbo, fs_ebo = create_fs_quad()

    # Build postprocess shader
    post_prog = link_program(POST_VS, POST_FS)
    # Attribute locations (OpenGL 2.1 + #version 120 means we use glVertexAttribPointer by name)
    # We'll pack aPos and aUV interleaved manually via client arrays for simplicity
    # But to keep it short, we'll bind arrays ad-hoc when drawing.

    clock = pygame.time.Clock()
    dragging = False
    last_mx, last_my = 0, 0

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_r:
                    cam_dist, cam_yaw, cam_pitch = 18.0, 35.0, -15.0
                elif e.key == pygame.K_g:
                    show_grid = not show_grid
                elif e.key == pygame.K_l:
                    enable_lensing = not enable_lensing
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    lens_strength = min(0.75, lens_strength + 0.02)
                elif e.key == pygame.K_MINUS:
                    lens_strength = max(0.02, lens_strength - 0.02)
                elif e.key == pygame.K_LEFTBRACKET:
                    lens_radius = max(0.15, lens_radius - 0.02)
                elif e.key == pygame.K_RIGHTBRACKET:
                    lens_radius = min(0.95, lens_radius + 0.02)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    dragging = True
                    last_mx, last_my = e.pos
                elif e.button == 4:
                    cam_dist = max(3.5, cam_dist - 1.0)
                elif e.button == 5:
                    cam_dist = min(80.0, cam_dist + 1.0)
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1:
                    dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                mx, my = e.pos
                dx = mx - last_mx
                dy = my - last_my
                last_mx, last_my = mx, my
                cam_yaw = (cam_yaw + dx * 0.3) % 360.0
                cam_pitch = max(-85.0, min(85.0, cam_pitch + dy * 0.3))

            elif e.type == pygame.VIDEORESIZE:
                WIN_W, WIN_H = e.w, e.h
                pygame.display.set_mode((WIN_W, WIN_H), DOUBLEBUF | OPENGL)
                glViewport(0, 0, WIN_W, WIN_H)
                set_perspective(WIN_W, WIN_H)
                # re-create FBO at new size
                glDeleteFramebuffers(1, [fbo])
                glDeleteTextures(1, [scene_tex])
                glDeleteRenderbuffers(1, [rbo])
                fbo, scene_tex, rbo = create_fbo_tex(WIN_W, WIN_H)

        # -------- 1) render scene into FBO --------
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, WIN_W, WIN_H)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        set_perspective(WIN_W, WIN_H)
        set_camera()

        draw_grid()
        draw_scene_objects()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # -------- 2) postprocess lensing to screen --------
        glViewport(0, 0, WIN_W, WIN_H)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glDisable(GL_DEPTH_TEST)
        glUseProgram(post_prog)

        # Set uniforms
        uScene_loc = glGetUniformLocation(post_prog, "uScene")
        uRes_loc = glGetUniformLocation(post_prog, "uResolution")
        uStrength_loc = glGetUniformLocation(post_prog, "uStrength")
        uRadius_loc = glGetUniformLocation(post_prog, "uRadius")
        uEH_loc = glGetUniformLocation(post_prog, "uEHpx")

        glUniform1i(uScene_loc, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, scene_tex)

        glUniform2f(uRes_loc, float(WIN_W), float(WIN_H))
        glUniform1f(uStrength_loc, lens_strength if enable_lensing else 0.0)
        glUniform1f(uRadius_loc, lens_radius)
        glUniform1f(uEH_loc, float(event_horizon_radius_px))

        # Bind FS quad and draw (client arrays for simplicity)
        glBindBuffer(GL_ARRAY_BUFFER, fs_vbo)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fs_ebo)

        # aPos (xy), aUV (uv) interleaved: [x,y,u,v] per vertex
        stride = 4 * 4
        # query attrib locations
        aPos = glGetAttribLocation(post_prog, "aPos")
        aUV  = glGetAttribLocation(post_prog, "aUV")
        glEnableVertexAttribArray(aPos)
        glEnableVertexAttribArray(aUV)
        glVertexAttribPointer(aPos, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glVertexAttribPointer(aUV,  2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, ctypes.c_void_p(0))

        glDisableVertexAttribArray(aPos)
        glDisableVertexAttribArray(aUV)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        glUseProgram(0)
        glEnable(GL_DEPTH_TEST)

        # -------- 3) draw black hole overlay (disc + glow) --------
        draw_black_hole_overlay(WIN_W, WIN_H, event_horizon_radius_px)

        # -------- 4) UI overlay (tiny text via pygame) --------
        # You can draw text with pygame if desired by making a separate surface and blitting,
        # but mixing is more code; keeping it minimal here.

        pygame.display.flip()
        clock.tick(60)

    # Cleanup
    glDeleteFramebuffers(1, [fbo])
    glDeleteTextures(1, [scene_tex])
    glDeleteRenderbuffers(1, [rbo])
    glDeleteBuffers(1, [fs_vbo])
    glDeleteBuffers(1, [fs_ebo])
    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()

"""Diagnostic-only explicit MSAA sample readback; not a released renderer.

Keeps the native sample count and scene. A multisample texture permits observing
individual samples before the driver's resolve. CPU averaging isolates resolve
rounding from rasterization; either change still requires image validation.
"""
import hashlib

import numpy as np


class MultisampleDiagnostic:
    def __init__(self, context):
        from OpenGL import GL
        from OpenGL.GL.shaders import compileProgram, compileShader

        self.context = context
        self.samples = int(context.con.offSamples)
        if self.samples <= 0:
            raise ValueError('Requires a multisample native context')
        self.sample_hashes = []
        self.original_read = context.read_pixels
        self.texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture)
        GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, self.samples, GL.GL_RGBA8,
                                 context.con.offWidth, context.con.offHeight, True)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, context.con.offFBO)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                 GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture, 0)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Diagnostic sample framebuffer incomplete')
        self.fbo = GL.glGenFramebuffers(1)
        self.buffer = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.buffer)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, context.con.offWidth, context.con.offHeight)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.buffer)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Diagnostic readback framebuffer incomplete')
        vertex = '''#version 330 compatibility
        void main() {
            vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
            gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
        }'''
        fragment = '''#version 330 compatibility
        uniform sampler2DMS source_samples;
        uniform int sample_index;
        out vec4 color;
        void main() { color = texelFetch(source_samples, ivec2(gl_FragCoord.xy), sample_index); }
        '''
        self.program = compileProgram(compileShader(vertex, GL.GL_VERTEX_SHADER),
                                      compileShader(fragment, GL.GL_FRAGMENT_SHADER))
        self.vao = GL.glGenVertexArrays(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, context.con.offFBO)

    def read_pixels(self, width, height, depth=False, segmentation=False):
        from OpenGL import GL

        if depth or segmentation:
            raise ValueError('Sample diagnostic accepts RGB only')
        errors = []
        for _ in range(16):
            error = int(GL.glGetError())
            if error == GL.GL_NO_ERROR:
                break
            errors.append(error)
        self.last_pending_errors = errors
        old_program = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        old_vao = int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING))
        old_read = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        old_draw = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        GL.glPushAttrib(GL.GL_ALL_ATTRIB_BITS)
        try:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
            GL.glViewport(0, 0, width, height)
            for capability in (GL.GL_DEPTH_TEST, GL.GL_STENCIL_TEST, GL.GL_SCISSOR_TEST,
                               GL.GL_CULL_FACE, GL.GL_BLEND, GL.GL_DITHER, GL.GL_MULTISAMPLE):
                GL.glDisable(capability)
            GL.glColorMask(True, True, True, True)
            GL.glUseProgram(self.program)
            GL.glBindVertexArray(self.vao)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, self.texture)
            GL.glUniform1i(GL.glGetUniformLocation(self.program, 'source_samples'), 0)
            values = []
            for index in range(self.samples):
                GL.glUniform1i(GL.glGetUniformLocation(self.program, 'sample_index'), index)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
                value = np.empty((height, width, 4), dtype=np.uint8)
                GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, value)
                values.append(value[..., :3].copy())
            stack = np.stack(values)
            self.sample_hashes.append(hashlib.sha256(stack.tobytes()).hexdigest())
            return np.rint(stack.astype(np.float64).mean(axis=0)).astype(np.uint8)
        finally:
            GL.glUseProgram(old_program)
            GL.glBindVertexArray(old_vao)
            GL.glPopAttrib()
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, old_read)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, old_draw)

    def close(self):
        from OpenGL import GL

        self.context.read_pixels = self.original_read
        for _ in range(16):
            if GL.glGetError() == GL.GL_NO_ERROR:
                break
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.context.con.offFBO)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                     GL.GL_RENDERBUFFER, self.context.con.offColor)
        GL.glDeleteTextures([self.texture])
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteRenderbuffers(1, [self.buffer])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteProgram(self.program)

#version 400

layout(location=0) in vec3 in_Position;
layout(location=1) in vec4 in_Color;
out vec4 ex_Color;

uniform mat4 ModelMatrix;
uniform mat4 ViewMatrix;

void main(void)
{
  gl_Position = ViewMatrix * vec4(in_Position.xyz,1.0);
  ex_Color = in_Color;
}
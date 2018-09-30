#version 400

layout(location=0) in vec3 in_Position;
layout(location=1) in vec4 in_Color;

layout(location=2) in vec4 in_vec0;
layout(location=3) in vec4 in_vec1;
layout(location=4) in vec4 in_vec2;
layout(location=5) in vec4 in_vec3;


out VS_OUT{
	vec4 ex_Color;
	vec4 exvec0;
	vec4 exvec1;
	vec4 exvec2;
	vec4 exvec3;
} vs_out;

uniform mat4 ModelMatrix;


void main(void)
{
  gl_Position = vec4(in_Position.xyz,1.0);
  vs_out.ex_Color = in_Color;

  vs_out.exvec0=in_vec0;
  vs_out.exvec1=in_vec1;
  vs_out.exvec2=in_vec2;
  vs_out.exvec3=in_vec3;
  //ex_Rotation[0][0] = in_vec0.x; ex_Rotation[1][0] = in_vec0.y; ex_Rotation[2][0] = in_vec0.z; ex_Rotation[3][0] = in_vec0.w;
  //ex_Rotation[0][1] = in_vec1.x; ex_Rotation[1][1] = in_vec1.y; ex_Rotation[2][1] = in_vec1.z; ex_Rotation[3][1] = in_vec1.w;
  //ex_Rotation[0][2] = in_vec2.x; ex_Rotation[1][2] = in_vec2.y; ex_Rotation[2][2] = in_vec2.z; ex_Rotation[3][2] = in_vec2.w;
  //ex_Rotation[0][3] = in_vec3.x; ex_Rotation[1][3] = in_vec3.y; ex_Rotation[2][3] = in_vec3.z; ex_Rotation[3][3] = in_vec3.w;

  //ex_Rotation[0][0] = 1; ex_Rotation[1][0] = 0; ex_Rotation[2][0] = 0; ex_Rotation[3][0] = 0;
  //ex_Rotation[0][1] = 0; ex_Rotation[1][1] = 2; ex_Rotation[2][1] = 0; ex_Rotation[3][1] = 0;
  //ex_Rotation[0][2] = 0; ex_Rotation[1][2] = 0; ex_Rotation[2][2] = 3; ex_Rotation[3][2] = 0;
  //ex_Rotation[0][3] = 0; ex_Rotation[1][3] = 0; ex_Rotation[2][3] = 0; ex_Rotation[3][3] = 4;
}